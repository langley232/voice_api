import os
import torch
import torchaudio
import tempfile
import shutil
import logging
from typing import Optional, Tuple, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, WebSocket
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Language Support Configuration
SUPPORTED_LANGUAGES = {
    'speech_input': set(),  # Add 101 supported languages
    'text_input_output': set(),  # Add 96 supported languages
    'speech_output': set()  # Add 35 supported languages
}

# --- Custom Exceptions ---


class ModelNotLoadedError(Exception):
    """Raised when the model is not loaded."""
    pass


class AudioProcessingError(Exception):
    """Raised when there's an error processing audio data."""
    pass


class TranslationError(Exception):
    """Raised when there's an error during translation."""
    pass


class UnsupportedLanguageError(Exception):
    """Raised when language is not supported for the requested task."""
    pass

# --- Pydantic Models ---


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    source_language: str = Field(..., min_length=2, max_length=5)
    target_language: str = Field(..., min_length=2, max_length=5)
    preserve_voice_style: bool = Field(default=False)


class STTResponse(BaseModel):
    text: str
    language: str
    confidence: Optional[float] = None


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    source_language: str = Field(..., min_length=2, max_length=5)
    target_language: str = Field(..., min_length=2, max_length=5)


class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str
    processing_time: float


class ModelMetrics:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_processing_time = 0

    def update_metrics(self, success: bool, processing_time: float):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.average_processing_time = (
            (self.average_processing_time *
             (self.total_requests - 1) + processing_time)
            / self.total_requests
        )

# --- Model Management ---


class ModelManager:
    def __init__(self):
        self.model = None
        self.device = None
        self.dtype = None
        self.last_loaded = None
        self.version = "facebook/seamless-m4t-v2-large"
        self.metrics = ModelMetrics()

    def load_model(self) -> None:
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                self.dtype = torch.float16
                device_str = "GPU (cuda:0)"
            else:
                self.device = torch.device("cpu")
                self.dtype = torch.float32
                device_str = "CPU"

            logger.info(
                f"Loading model on {device_str} with dtype {self.dtype}")

            from transformers import SeamlessM4Tv2Model, AutoProcessor
            try:
                logger.info("Loading processor...")
                self.processor = AutoProcessor.from_pretrained(
                    self.version,
                    trust_remote_code=True
                )
                logger.info("Processor loaded successfully")

                logger.info("Loading model...")
                self.model = SeamlessM4Tv2Model.from_pretrained(
                    self.version,
                    device_map=self.device,
                    torch_dtype=self.dtype,
                    trust_remote_code=True
                )
                logger.info("Model loaded successfully")

                self.last_loaded = datetime.now()
                logger.info(f"Successfully loaded model on {device_str}")

            except Exception as e:
                logger.error(f"Error loading model components: {str(e)}")
                raise ModelNotLoadedError(
                    f"Model component loading failed: {str(e)}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelNotLoadedError(f"Model loading failed: {str(e)}")

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_model(self):
        if not self.is_loaded():
            raise ModelNotLoadedError("Model is not loaded")
        return self.model, self.processor


# Initialize model manager
model_manager = ModelManager()

# FastAPI app instance
app = FastAPI(
    title="Seamless M4T v2 API",
    description="API for Seamless M4T v2 Large model",
    version="1.0.0"
)

# --- Helper Functions ---


def validate_language_support(
    source_lang: str,
    target_lang: str,
    task_type: str
) -> Tuple[bool, str]:
    if task_type == 's2st':
        if source_lang not in SUPPORTED_LANGUAGES['speech_input']:
            return False, f"Language {source_lang} not supported for speech input"
        if target_lang not in SUPPORTED_LANGUAGES['speech_output']:
            return False, f"Language {target_lang} not supported for speech output"
    return True, ""


async def process_streaming_audio(audio_data: bytes, model, processor):
    try:
        # Process streaming audio using SeamlessM4T v2
        audio_inputs = processor(audios=audio_data, return_tensors="pt")
        outputs = model.generate(**audio_inputs)
        return outputs.cpu().numpy().squeeze()
    except Exception as e:
        logger.error(f"Error processing streaming audio: {str(e)}")
        raise AudioProcessingError(str(e))

# --- Dependencies ---


async def get_model():
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_manager.get_model()

# --- Startup Event ---


@app.on_event("startup")
async def startup_event():
    try:
        model_manager.load_model()
    except ModelNotLoadedError as e:
        logger.error(f"Failed to load model during startup: {str(e)}")

# --- Health Check ---


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_manager.is_loaded() else "unhealthy",
        "model_loaded": model_manager.is_loaded(),
        "last_loaded": model_manager.last_loaded.isoformat() if model_manager.last_loaded else None,
        "device": str(model_manager.device) if model_manager.device else None,
        "version": model_manager.version,
        "metrics": {
            "total_requests": model_manager.metrics.total_requests,
            "successful_requests": model_manager.metrics.successful_requests,
            "failed_requests": model_manager.metrics.failed_requests,
            "average_processing_time": model_manager.metrics.average_processing_time
        }
    }

# --- STT Endpoint ---


@app.post("/stt", response_model=STTResponse)
async def speech_to_text(
    audio_file: UploadFile = File(...),
    model_and_processor: Tuple = Depends(get_model)
):
    start_time = datetime.now()
    temp_audio_path = None
    model, processor = model_and_processor

    try:
        # Save uploaded file
        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        logger.info(f"Processing STT request for file: {audio_file.filename}")

        # Process audio using the processor
        audio_inputs = processor(audios=temp_audio_path, return_tensors="pt")
        outputs = model.generate(
            **audio_inputs, task="s2t_transcript", tgt_lang="eng")

        # Convert outputs to text
        transcribed_text = processor.decode(
            outputs[0].cpu(), skip_special_tokens=True)
        detected_language = outputs[1] if len(outputs) > 1 else "unknown"

        if not transcribed_text:
            raise AudioProcessingError("Transcription failed")

        processing_time = (datetime.now() - start_time).total_seconds()
        model_manager.metrics.update_metrics(True, processing_time)
        logger.info(f"STT completed in {processing_time:.2f}s")

        return STTResponse(
            text=transcribed_text,
            language=detected_language
        )

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        model_manager.metrics.update_metrics(False, processing_time)
        logger.error(f"STT Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# --- TTS Endpoint ---


@app.post("/tts")
async def text_to_speech(
    request: TTSRequest,
    model_and_processor: Tuple = Depends(get_model)
):
    start_time = datetime.now()
    model, processor = model_and_processor

    try:
        # Validate language support
        is_valid, error_message = validate_language_support(
            request.source_language,
            request.target_language,
            "t2st"
        )
        if not is_valid:
            raise UnsupportedLanguageError(error_message)

        # Process text to speech
        inputs = processor(
            text=request.text,
            src_lang=request.source_language,
            return_tensors="pt"
        )

        outputs = model.generate(
            **inputs,
            tgt_lang=request.target_language,
            preserve_voice_style=request.preserve_voice_style
        )

        # Save audio to temporary file
        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        torchaudio.save(
            temp_audio_path,
            outputs[0].cpu(),
            sample_rate=16000
        )

        processing_time = (datetime.now() - start_time).total_seconds()
        model_manager.metrics.update_metrics(True, processing_time)

        return FileResponse(
            temp_audio_path,
            media_type="audio/wav",
            background=BackgroundTask(lambda: os.remove(temp_audio_path))
        )

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        model_manager.metrics.update_metrics(False, processing_time)
        logger.error(f"TTS Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Translation Endpoint ---


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    model_and_processor: Tuple = Depends(get_model)
):
    start_time = datetime.now()
    model, processor = model_and_processor

    try:
        # Validate language support
        is_valid, error_message = validate_language_support(
            request.source_language,
            request.target_language,
            "t2tt"
        )
        if not is_valid:
            raise UnsupportedLanguageError(error_message)

        # Process translation
        inputs = processor(
            text=request.text,
            src_lang=request.source_language,
            return_tensors="pt"
        )

        outputs = model.generate(
            **inputs,
            task="t2tt",
            tgt_lang=request.target_language
        )

        translated_text = processor.decode(
            outputs[0].cpu(), skip_special_tokens=True)

        processing_time = (datetime.now() - start_time).total_seconds()
        model_manager.metrics.update_metrics(True, processing_time)

        return TranslationResponse(
            translated_text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language,
            processing_time=processing_time
        )

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        model_manager.metrics.update_metrics(False, processing_time)
        logger.error(f"Translation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Streaming Translation Endpoint ---


@app.websocket("/stream-translation")
async def stream_translation(websocket: WebSocket):
    await websocket.accept()
    try:
        model, processor = model_manager.get_model()
        while True:
            audio_data = await websocket.receive_bytes()
            result = await process_streaming_audio(audio_data, model, processor)
            await websocket.send_text(result)
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        await websocket.close(code=1000)
