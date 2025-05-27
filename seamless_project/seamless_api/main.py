import os
import torch
import torchaudio  # For saving audio
import tempfile  # For creating temporary files
import shutil  # For saving uploaded file
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask  # For cleanup
from pydantic import BaseModel
from seamless_communication.inference import Translator
import numpy as np

# Global variable to hold the model translator
model_translator: Translator = None

# --- Pydantic Models ---


class TTSRequest(BaseModel):
    text: str
    source_language: str  # Added source language parameter
    target_language: str  # e.g., 'eng', 'spa', 'fra'


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

        print(
            f"Attempting to load model on {device_str} with dtype {dtype}...")

        # Model name and vocoder name (these are the actual identifiers)
        model_identifier = "seamlessM4T_v2_large"
        vocoder_identifier = "vocoder_v2"

        # Load the translator model
        translator_instance = Translator(
            model_identifier,    # First positional argument: model name/identifier
            vocoder_identifier,  # Second positional argument: vocoder name/identifier
            device=device,       # Keyword argument for device
            dtype=dtype          # Keyword argument for dtype
        )

        model_translator = translator_instance
        print(
            f"Successfully loaded Seamless M4T v2 Large model ('{model_identifier}') on {device_str}.")

    except Exception as e:
        print(f"Error loading model: {e}")
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

if __name__ == "__main__":
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
        raise HTTPException(
            status_code=503, detail="Model not loaded or failed to load.")

    temp_audio_path = None
    try:
        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        print(
            f"Received STT request: Saved uploaded audio to temporary file: {temp_audio_path}")

        transcribed_text, detected_language_code, _, _ = model_translator.predict(
            input=temp_audio_path,
            task_str="s2t_transcript",
            tgt_lang="eng"
        )

        if transcribed_text is None or detected_language_code is None:
            raise HTTPException(
                status_code=500, detail="Model failed to transcribe audio or detect language.")

        print(
            f"Transcribed text: '{transcribed_text}', Detected language: '{detected_language_code}'")

        return STTResponse(text=transcribed_text, language=detected_language_code)

    except HTTPException as e_http:
        raise e_http
    except Exception as e:
        print(f"Error during STT processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"STT processing failed: {str(e)}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"Cleaned up temporary audio file: {temp_audio_path}")
            except Exception as e_cleanup:
                print(
                    f"Error cleaning up temporary file {temp_audio_path}: {e_cleanup}")


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translates text from a source language to a target language.
    """
    global model_translator

    if model_translator is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded or failed to load.")

    try:
        print(
            f"Received Translation request: Text='{request.text}', Source='{request.source_language}', Target='{request.target_language}'")

        translated_text_raw, _ = model_translator.predict(
            input=request.text,
            task_str="t2tt",
            src_lang=request.source_language,
            tgt_lang=request.target_language
        )

        if translated_text_raw is None:
            raise HTTPException(
                status_code=500, detail="Model failed to translate text.")

        # --- NEW FIX APPLIED HERE ---
        # Extract the string from the list and ensure it's a standard Python string
        if isinstance(translated_text_raw, list) and len(translated_text_raw) > 0:
            final_translated_text = str(translated_text_raw[0])
        else:
            # Fallback for unexpected cases, but the error implies it will be a list
            final_translated_text = str(translated_text_raw)

        print(f"Translated text: '{final_translated_text}'")

        return TranslationResponse(translated_text=final_translated_text)

    except HTTPException as e_http:
        raise e_http
    except Exception as e:
        print(f"Error during text translation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Text translation failed: {str(e)}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Generates speech from text in the specified target language.
    """
    global model_translator

    if model_translator is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded or failed to load.")

    temp_audio_path = None
    try:
        print(f"[TTS] Starting TTS generation with parameters:")
        print(f"[TTS] Text: '{request.text}'")
        print(f"[TTS] Source Language: '{request.source_language}'")
        print(f"[TTS] Target Language: '{request.target_language}'")
        print(f"[TTS] Model translator type: {type(model_translator)}")
        print(
            f"[TTS] Available task strings: {model_translator.supported_tasks if hasattr(model_translator, 'supported_tasks') else 'Not available'}")

        try:
            result = model_translator.predict(
                input=request.text,
                task_str="t2st",
                src_lang=request.source_language,
                tgt_lang=request.target_language,
            )
            print(f"[TTS] Model prediction result type: {type(result)}")
            print(f"[TTS] Model prediction result: {result}")

            # Handle different return types
            if isinstance(result, tuple):
                if len(result) == 3:
                    output_text, audio_waveform, audio_sample_rate = result
                elif len(result) == 2:
                    output_text, audio_waveform = result
                    audio_sample_rate = 16000  # Default sample rate for Seamless M4T
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Unexpected number of return values from model: {len(result)}"
                    )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected return type from model: {type(result)}"
                )

            print(f"[TTS] Model prediction completed:")
            print(f"[TTS] Output text type: {type(output_text)}")
            print(f"[TTS] Audio waveform type: {type(audio_waveform)}")
            print(f"[TTS] Audio sample rate: {audio_sample_rate}")

            # Handle BatchedSpeechOutput
            if hasattr(audio_waveform, 'audio'):
                print(f"[TTS] Converting BatchedSpeechOutput to tensor")
                audio_waveform = audio_waveform.audio
                if isinstance(audio_waveform, list):
                    # Take first item if it's a list
                    audio_waveform = audio_waveform[0]

            if audio_waveform is None:
                raise HTTPException(
                    status_code=500,
                    detail="Model failed to generate audio waveform"
                )

            print(f"[TTS] Processing audio data...")
            try:
                # Convert to numpy array first if it's not already a tensor
                if not isinstance(audio_waveform, torch.Tensor):
                    print(f"[TTS] Converting to numpy array first")
                    if hasattr(audio_waveform, 'numpy'):
                        audio_numpy = audio_waveform.numpy()
                    elif hasattr(audio_waveform, 'detach'):
                        # If it's a tensor-like object with detach method
                        audio_numpy = audio_waveform.detach().cpu().numpy()
                    else:
                        # Try to convert directly to numpy
                        audio_numpy = np.array(audio_waveform)

                    # Ensure correct dtype
                    print(
                        f"[TTS] Original numpy array dtype: {audio_numpy.dtype}")
                    audio_numpy = audio_numpy.astype(np.float32)
                    print(
                        f"[TTS] Converted numpy array dtype: {audio_numpy.dtype}")

                    print(f"[TTS] Converting numpy array to tensor")
                    audio_waveform = torch.from_numpy(audio_numpy)

                # Ensure we're on the right device
                if torch.cuda.is_available():
                    audio_waveform = audio_waveform.cuda()

                # Add channel dimension if needed
                if audio_waveform.ndim == 1:
                    print(f"[TTS] Adding channel dimension to audio data")
                    audio_waveform = audio_waveform.unsqueeze(0)

                print(f"[TTS] Audio data shape: {audio_waveform.shape}")
                print(f"[TTS] Audio data type: {audio_waveform.dtype}")
                print(f"[TTS] Audio data device: {audio_waveform.device}")

                # Only move to CPU when saving to file
                print(f"[TTS] Moving audio data to CPU for saving...")
                audio_data_cpu = audio_waveform.cpu()

            except Exception as e:
                print(f"[TTS] Error processing audio data: {e}")
                print(f"[TTS] Audio waveform type: {type(audio_waveform)}")
                if hasattr(audio_waveform, 'shape'):
                    print(
                        f"[TTS] Audio waveform shape: {audio_waveform.shape}")
                if hasattr(audio_waveform, 'device'):
                    print(
                        f"[TTS] Audio waveform device: {audio_waveform.device}")
                if hasattr(audio_waveform, 'dtype'):
                    print(
                        f"[TTS] Audio waveform dtype: {audio_waveform.dtype}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process audio data: {str(e)}"
                )

            fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            print(f"[TTS] Created temporary file: {temp_audio_path}")

            try:
                print(f"[TTS] Saving audio to file...")
                torchaudio.save(
                    temp_audio_path,
                    audio_data_cpu,
                    audio_sample_rate
                )
                print(f"[TTS] Successfully saved audio to: {temp_audio_path}")

                response = FileResponse(
                    temp_audio_path,
                    media_type="audio/wav",
                    filename="tts_output.wav",
                    background=BackgroundTask(os.remove, temp_audio_path)
                )
                print(f"[TTS] Successfully created FileResponse")
                return response

            except Exception as e_save:
                print(f"[TTS] Error saving audio to file:")
                print(f"[TTS] Error type: {type(e_save)}")
                print(f"[TTS] Error message: {str(e_save)}")
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                raise HTTPException(
                    status_code=500, detail=f"Failed to save generated audio: {str(e_save)}")

        except Exception as predict_error:
            print(f"[TTS] Error during model prediction:")
            print(f"[TTS] Error type: {type(predict_error)}")
            print(f"[TTS] Error message: {str(predict_error)}")
            print(
                f"[TTS] Error details: {predict_error.__dict__ if hasattr(predict_error, '__dict__') else 'No additional details'}")
            raise HTTPException(
                status_code=500,
                detail=f"TTS generation failed: {str(predict_error)}"
            )

    except HTTPException as e_http:
        print(f"[TTS] HTTP Exception raised:")
        print(f"[TTS] Status code: {e_http.status_code}")
        print(f"[TTS] Detail: {e_http.detail}")
        raise e_http
    except Exception as e:
        print(f"[TTS] Unexpected error during TTS generation:")
        print(f"[TTS] Error type: {type(e)}")
        print(f"[TTS] Error message: {str(e)}")
        print(
            f"[TTS] Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
        # Clean up temp file if it exists
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(
                    f"[TTS] Cleaned up temporary audio file after error: {temp_audio_path}")
            except Exception as e_cleanup:
                print(
                    f"[TTS] Error cleaning up temporary file {temp_audio_path}: {e_cleanup}")
        raise HTTPException(
            status_code=500, detail=f"TTS generation failed: {str(e)}")
