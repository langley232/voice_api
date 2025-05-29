import os
import torch
import torchaudio  # For saving audio
import tempfile  # For creating temporary files
import shutil  # For saving uploaded file
import logging
from typing import Optional, Tuple, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask  # For cleanup
from pydantic import BaseModel, Field
from seamless_communication.inference import Translator
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to hold the model translator
model_translator: Translator = None

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

# --- Pydantic Models ---


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    source_language: str = Field(..., min_length=2, max_length=5)
    target_language: str = Field(..., min_length=2, max_length=5)


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


# --- Model Management ---
class ModelManager:
    def __init__(self):
        self.model: Optional[Translator] = None
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        self.last_loaded: Optional[datetime] = None
        self.version: str = "seamlessM4T_v2_large"

    def load_model(self) -> None:
        """Load the model with proper error handling and logging."""
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

            self.model = Translator(
                self.version,
                "vocoder_v2",
                device=self.device,
                dtype=self.dtype
            )
            self.last_loaded = datetime.now()
            logger.info(f"Successfully loaded model on {device_str}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelNotLoadedError(f"Model loading failed: {str(e)}")

    def is_loaded(self) -> bool:
        """Check if the model is loaded and healthy."""
        return self.model is not None

    def get_model(self) -> Translator:
        """Get the model instance with proper error handling."""
        if not self.is_loaded():
            raise ModelNotLoadedError("Model is not loaded")
        return self.model


# Initialize model manager
model_manager = ModelManager()

# FastAPI app instance
app = FastAPI(
    title="Seamless M4T API",
    description="API for Seamless M4T v2 Large model",
    version="1.0.0"
)

# --- Dependencies ---


async def get_model():
    """Dependency to get the model instance."""
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_manager.get_model()

# --- Startup Event ---


@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    try:
        model_manager.load_model()
    except ModelNotLoadedError as e:
        logger.error(f"Failed to load model during startup: {str(e)}")

# --- Health Check ---


@app.get("/health")
async def health_check():
    """Check the health of the API and model."""
    return {
        "status": "healthy" if model_manager.is_loaded() else "unhealthy",
        "model_loaded": model_manager.is_loaded(),
        "last_loaded": model_manager.last_loaded.isoformat() if model_manager.last_loaded else None,
        "device": str(model_manager.device) if model_manager.device else None,
        "version": model_manager.version
    }

# --- STT Endpoint ---


@app.post("/stt", response_model=STTResponse)
async def speech_to_text(
    audio_file: UploadFile = File(...),
    model: Translator = Depends(get_model)
):
    """Transcribe speech from an audio file to text."""
    temp_audio_path = None
    start_time = datetime.now()

    try:
        # Save uploaded file
        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        logger.info(f"Processing STT request for file: {audio_file.filename}")

        # Perform transcription
        transcribed_text, detected_language_code, _, _ = model.predict(
            input=temp_audio_path,
            task_str="s2t_transcript",
            tgt_lang="eng"
        )

        if not transcribed_text or not detected_language_code:
            raise AudioProcessingError("Transcription failed")

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"STT completed in {processing_time:.2f}s")

        return STTResponse(
            text=transcribed_text,
            language=detected_language_code
        )

    except Exception as e:
        logger.error(f"STT processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.debug(f"Cleaned up temporary file: {temp_audio_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")

# --- Translation Endpoint ---


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    model: Translator = Depends(get_model)
):
    """Translate text between languages."""
    start_time = datetime.now()

    try:
        logger.info(
            f"Processing translation request: {request.source_language} -> {request.target_language}")

        translated_text_raw, _ = model.predict(
            input=request.text,
            task_str="t2tt",
            src_lang=request.source_language,
            tgt_lang=request.target_language
        )

        if not translated_text_raw:
            raise TranslationError("Translation failed")

        # Handle different response types
        if isinstance(translated_text_raw, list) and translated_text_raw:
            final_text = str(translated_text_raw[0])
        else:
            final_text = str(translated_text_raw)

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Translation completed in {processing_time:.2f}s")

        return TranslationResponse(
            translated_text=final_text,
            source_language=request.source_language,
            target_language=request.target_language,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- TTS Endpoint ---


@app.post("/tts")
async def text_to_speech(
    request: TTSRequest,
    model: Translator = Depends(get_model)
):
    """Generate speech from text."""
    temp_audio_path = None
    start_time = datetime.now()

    try:
        logger.info(
            f"Processing TTS request: {request.source_language} -> {request.target_language}")

        result = model.predict(
            input=request.text,
            task_str="t2st",
            src_lang=request.source_language,
            tgt_lang=request.target_language,
        )

        # Log the raw result
        logger.info(f"Raw model output type: {type(result)}")
        if isinstance(result, tuple):
            logger.info(f"Result tuple length: {len(result)}")
            for i, item in enumerate(result):
                logger.info(f"Result item {i} type: {type(item)}")
                if hasattr(item, 'shape'):
                    logger.info(f"Result item {i} shape: {item.shape}")
                elif hasattr(item, '__len__'):
                    logger.info(f"Result item {i} length: {len(item)}")

        # Handle model output
        if isinstance(result, tuple):
            if len(result) == 3:
                output_text, audio_waveform, audio_sample_rate = result
                logger.info(
                    f"Got 3-tuple result: text={type(output_text)}, audio={type(audio_waveform)}, sample_rate={audio_sample_rate}")
            elif len(result) == 2:
                output_text, audio_waveform = result
                audio_sample_rate = 16000
                logger.info(
                    f"Got 2-tuple result: text={type(output_text)}, audio={type(audio_waveform)}")
            else:
                raise AudioProcessingError(
                    f"Unexpected number of return values: {len(result)}")
        else:
            raise AudioProcessingError(
                f"Unexpected return type: {type(result)}")

        # Process audio waveform
        if hasattr(audio_waveform, 'audio'):
            logger.info(
                f"Audio waveform has 'audio' attribute. Original type: {type(audio_waveform)}")
            audio_waveform = audio_waveform.audio
            logger.info(
                f"After accessing 'audio' attribute, type: {type(audio_waveform)}")
            if isinstance(audio_waveform, list):
                logger.info(
                    f"Audio waveform is a list of length {len(audio_waveform)}")
                audio_waveform = audio_waveform[0]
                logger.info(
                    f"After taking first element, type: {type(audio_waveform)}")

        # Convert to tensor if needed
        if not isinstance(audio_waveform, torch.Tensor):
            try:
                # Handle BatchedSpeechOutput directly
                if hasattr(audio_waveform, 'audio'):
                    logger.info(
                        "Converting BatchedSpeechOutput audio attribute")
                    audio_waveform = audio_waveform.audio
                    if isinstance(audio_waveform, list):
                        audio_waveform = audio_waveform[0]
                    logger.info(
                        f"After BatchedSpeechOutput conversion, type: {type(audio_waveform)}")

                # Convert to numpy array first
                if hasattr(audio_waveform, 'numpy'):
                    logger.info("Converting using numpy() method")
                    audio_numpy = audio_waveform.numpy()
                elif hasattr(audio_waveform, 'detach'):
                    logger.info("Converting using detach().cpu().numpy()")
                    audio_numpy = audio_waveform.detach().cpu().numpy()
                else:
                    logger.info(
                        f"Converting using np.array() from type: {type(audio_waveform)}")
                    audio_numpy = np.array(audio_waveform)

                logger.info(
                    f"Numpy array shape: {audio_numpy.shape}, dtype: {audio_numpy.dtype}")

                # Handle 0-d arrays
                if audio_numpy.ndim == 0:
                    logger.info("Converting 0-d array to 1-d array")
                    audio_numpy = np.array(
                        [audio_numpy.item()], dtype=np.float32)
                # Handle object dtype arrays
                elif audio_numpy.dtype == np.dtype('O'):
                    logger.info("Converting object dtype array")
                    if audio_numpy.size == 0:
                        raise AudioProcessingError("Empty audio data received")
                    # Try to convert each element to float32
                    try:
                        audio_numpy = np.array(
                            [np.array(x, dtype=np.float32) for x in audio_numpy], dtype=np.float32)
                        logger.info(
                            f"Converted object array shape: {audio_numpy.shape}, dtype: {audio_numpy.dtype}")
                    except Exception as e:
                        logger.error(
                            f"Failed to convert object array: {str(e)}")
                        raise AudioProcessingError(
                            f"Failed to convert audio data: {str(e)}")
                else:
                    audio_numpy = audio_numpy.astype(np.float32)

                # Ensure we have a 1D or 2D array
                if audio_numpy.ndim == 0:
                    logger.info("Reshaping 0-d array to 1-d")
                    audio_numpy = audio_numpy.reshape(1)
                elif audio_numpy.ndim > 2:
                    logger.info(f"Reshaping {audio_numpy.ndim}-d array to 2-d")
                    audio_numpy = audio_numpy.reshape(-1,
                                                      audio_numpy.shape[-1])

                logger.info(
                    f"Final numpy array shape: {audio_numpy.shape}, dtype: {audio_numpy.dtype}")

                # Convert to tensor
                audio_waveform = torch.from_numpy(audio_numpy)
                if torch.cuda.is_available():
                    audio_waveform = audio_waveform.cuda()

            except Exception as e:
                logger.error(f"Audio conversion error: {str(e)}")
                raise AudioProcessingError(
                    f"Failed to convert audio data: {str(e)}")

        # Add channel dimension if needed
        if audio_waveform.ndim == 1:
            logger.info("Adding channel dimension to audio data")
            audio_waveform = audio_waveform.unsqueeze(0)

        # Save audio file
        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        audio_data_cpu = audio_waveform.cpu()
        logger.info(
            f"Final audio tensor shape: {audio_data_cpu.shape}, dtype: {audio_data_cpu.dtype}")
        torchaudio.save(temp_audio_path, audio_data_cpu, audio_sample_rate)

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"TTS completed in {processing_time:.2f}s")

        return FileResponse(
            temp_audio_path,
            media_type="audio/wav",
            filename="tts_output.wav",
            background=BackgroundTask(os.remove, temp_audio_path)
        )

    except Exception as e:
        logger.error(f"TTS processing error: {str(e)}")
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
