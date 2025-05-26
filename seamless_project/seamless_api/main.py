import os
import torch
import torchaudio # For saving audio
import tempfile # For creating temporary files
import shutil # For saving uploaded file
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask # For cleanup
from pydantic import BaseModel
from seamless_communication.inference import Translator

# Global variable to hold the model translator
model_translator: Translator = None

# --- Pydantic Models ---
class TTSRequest(BaseModel):
    text: str
    target_language: str # e.g., 'eng', 'spa', 'fra'

class STTResponse(BaseModel):
    text: str
    language: str

class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

class TranslationResponse(BaseModel):
    translated_text: str

# FastAPI app instance
app = FastAPI()

@app.on_event("startup")
async def load_model_on_startup():
    """
    Loads the Seamless M4T v2 Large model on application startup.
    """
    global model_translator

    try:
        # Determine device and data type
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            dtype = torch.float16
            device_str = "GPU (cuda:0)"
        else:
            device = torch.device("cpu")
            dtype = torch.float32  # float16 is not typically used for CPU inference
            device_str = "CPU"

        print(f"Attempting to load model on {device_str} with dtype {dtype}...")

        # Model name and vocoder name (these are the actual identifiers)
        model_identifier = "seamlessM4T_v2_large"
        vocoder_identifier = "vocoder_v2"

        # Load the translator model
        # The 'model_name' (identifier) and 'vocoder_name' (identifier) are typically passed
        # as the first and second POSITIONAL arguments.
        translator_instance = Translator(
            model_identifier,    # First positional argument: model name/identifier
            vocoder_identifier,  # Second positional argument: vocoder name/identifier
            device=device,       # Keyword argument for device
            dtype=dtype          # Keyword argument for dtype
        )
        
        model_translator = translator_instance
        print(f"Successfully loaded Seamless M4T v2 Large model ('{model_identifier}') on {device_str}.")

    except Exception as e:
        print(f"Error loading model: {e}")
        # Depending on policy, you might want to raise the exception
        # or allow the app to start without the model for some endpoints.
        # For now, we'll let it start and log the error.
        model_translator = None 

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    if model_translator:
        return {"message": "Seamless API is running. Model loaded."}
    else:
        return {"message": "Seamless API is running. Model FAILED to load."}

# Placeholder for future endpoints that will use the model_translator
# For example:
# @app.post("/translate_speech_to_text")
# async def translate_s2t(audio_file: UploadFile = File(...)):
#     if not model_translator:
#         raise HTTPException(status_code=503, detail="Model not loaded")
#     # ... implementation ...
#     pass

if __name__ == "__main__":
    # This part is for local development/testing if not using Uvicorn directly via CMD
    # The Dockerfile CMD uses: ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.post("/stt", response_model=STTResponse)
async def speech_to_text(audio_file: UploadFile = File(...)):
    """
    Transcribes speech from an audio file to text.
    The model should automatically detect the language of the audio.
    """
    global model_translator

    if model_translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded or failed to load.")

    temp_audio_path = None
    try:
        # Save uploaded audio file to a temporary path
        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav") # Assuming WAV, or let model handle format
        os.close(fd) # Close descriptor, open with 'wb'

        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        print(f"Received STT request: Saved uploaded audio to temporary file: {temp_audio_path}")

        transcribed_text, detected_language_code, _, _ = model_translator.predict(
            input=temp_audio_path,
            task_str="s2t_transcript", # Speech-to-Text (transcript)
            tgt_lang="eng" # Required, but hoping `detected_language_code` gives the true language
        )

        if transcribed_text is None or detected_language_code is None:
            raise HTTPException(status_code=500, detail="Model failed to transcribe audio or detect language.")

        print(f"Transcribed text: '{transcribed_text}', Detected language: '{detected_language_code}'")
        
        return STTResponse(text=transcribed_text, language=detected_language_code)

    except HTTPException as e_http:
        raise e_http
    except Exception as e:
        print(f"Error during STT processing: {e}")
        raise HTTPException(status_code=500, detail=f"STT processing failed: {str(e)}")
    finally:
        # Clean up the temporary audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"Cleaned up temporary audio file: {temp_audio_path}")
            except Exception as e_cleanup:
                print(f"Error cleaning up temporary file {temp_audio_path}: {e_cleanup}")


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translates text from a source language to a target language.
    """
    global model_translator

    if model_translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded or failed to load.")

    try:
        print(f"Received Translation request: Text='{request.text}', Source='{request.source_language}', Target='{request.target_language}'")
        
        translated_text, _, _, _ = model_translator.predict(
            input=request.text,
            task_str="t2tt_text", # Text-to-Text Translation
            src_lang=request.source_language,
            tgt_lang=request.target_language
        )

        if translated_text is None:
            raise HTTPException(status_code=500, detail="Model failed to translate text.")

        print(f"Translated text: '{translated_text}'")
        
        return TranslationResponse(translated_text=translated_text)

    except HTTPException as e_http:
        raise e_http
    except Exception as e:
        print(f"Error during text translation: {e}")
        raise HTTPException(status_code=500, detail=f"Text translation failed: {str(e)}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Generates speech from text in the specified target language.
    """
    global model_translator

    if model_translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded or failed to load.")

    try:
        print(f"Received TTS request: Text='{request.text}', Target Language='{request.target_language}'")
        
        output_text, audio_waveform, audio_sample_rate = model_translator.predict(
            input=request.text,
            task_str="t2st_sync", # Text-to-Speech (and Text)
            tgt_lang=request.target_language,
        )

        if audio_waveform is None or audio_sample_rate is None:
            raise HTTPException(status_code=500, detail="Model failed to generate audio.")

        print(f"Generated audio: Waveform shape {audio_waveform.shape}, Sample rate {audio_sample_rate}")
        
        audio_data_cpu = audio_waveform.cpu()
        if audio_data_cpu.ndim == 1:
            audio_data_cpu = audio_data_cpu.unsqueeze(0)

        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            torchaudio.save(
                temp_audio_path,
                audio_data_cpu,
                audio_sample_rate
            )
            print(f"Saved generated audio to temporary file: {temp_audio_path}")

            response = FileResponse(
                temp_audio_path,
                media_type="audio/wav",
                filename="tts_output.wav",
                background=BackgroundTask(os.remove, temp_audio_path)
            )
            return response

        except Exception as e_save:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            print(f"Error saving audio to file: {e_save}")
            raise HTTPException(status_code=500, detail=f"Failed to save generated audio: {str(e_save)}")

    except HTTPException as e_http:
        raise e_http
    except Exception as e:
        print(f"Error during TTS generation: {e}")
        # Attempt to clean up temp file if it was created before the error in predict or saving
        # This specific location might not always have temp_audio_path defined if error is early
        # Consider moving cleanup to a broader finally block if needed, though the save block has one.
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
