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
                print(
                    f"[TTS] BatchedSpeechOutput type: {type(audio_waveform)}")
                print(
                    f"[TTS] BatchedSpeechOutput attributes: {dir(audio_waveform)}")

                # Get the audio data from BatchedSpeechOutput
                audio_waveform = audio_waveform.audio
                # print(f"[TTS] Audio attribute type: {type(audio_waveform)}") # Covered by initial logging

            # --- Start of new logic ---
            # 1. Add Initial Logging
            print(f"[TTS] Initial audio_waveform type: {type(audio_waveform)}")
            if hasattr(audio_waveform, 'dtype'): print(f"[TTS] Initial audio_waveform dtype: {audio_waveform.dtype}")
            if hasattr(audio_waveform, 'shape'): print(f"[TTS] Initial audio_waveform shape: {audio_waveform.shape}")
            if isinstance(audio_waveform, list) and len(audio_waveform) > 0: print(f"[TTS] Initial audio_waveform is list, first element type: {type(audio_waveform[0])}")

            # 2. Handle List of Tensors/Arrays for audio_waveform
            if isinstance(audio_waveform, list):
                if len(audio_waveform) > 0 and all(isinstance(item, torch.Tensor) for item in audio_waveform):
                    print(f"[TTS] audio_waveform is a list of tensors. Concatenating...")
                    audio_waveform = torch.cat([item.cpu().float() for item in audio_waveform], dim=0) # Ensure items are float and on CPU before cat
                    print(f"[TTS] Concatenated audio_waveform: shape={audio_waveform.shape}, dtype={audio_waveform.dtype}")
                elif len(audio_waveform) > 0 and all(isinstance(item, np.ndarray) for item in audio_waveform):
                    print(f"[TTS] audio_waveform is a list of numpy arrays. Concatenating...")
                    audio_waveform = np.concatenate([item.astype(np.float32) for item in audio_waveform], axis=0)
                    audio_waveform = torch.from_numpy(audio_waveform) # Convert to tensor
                    print(f"[TTS] Concatenated and tensor-converted audio_waveform: shape={audio_waveform.shape}, dtype={audio_waveform.dtype}")
                else:
                    print(f"[TTS] audio_waveform is a list, but elements are not all tensors or ndarrays, or list is empty. Type of first element: {type(audio_waveform[0]) if len(audio_waveform) > 0 else 'Empty list'}")
                    # Attempt to convert to tensor if it's a list of numbers (e.g. from S2ST)
                    if len(audio_waveform) > 0 and isinstance(audio_waveform[0], (int, float)):
                        print(f"[TTS] Attempting to convert list of numbers to tensor.")
                        try:
                            audio_waveform = torch.tensor(audio_waveform, dtype=torch.float32)
                            print(f"[TTS] Converted list of numbers to tensor: shape={audio_waveform.shape}, dtype={audio_waveform.dtype}")
                        except Exception as e_conv:
                            print(f"[TTS] Failed to convert list of numbers to tensor: {e_conv}")
                            raise HTTPException(status_code=500, detail="Audio data is a list of numbers but could not be converted to a tensor.")


            # 3. Ensure audio_waveform becomes a Tensor (if it was numpy.ndarray)
            if isinstance(audio_waveform, np.ndarray):
                print(f"[TTS] Converting numpy.ndarray audio_waveform to tensor. Original dtype: {audio_waveform.dtype}")
                audio_waveform = torch.from_numpy(audio_waveform.astype(np.float32)) # Ensure float32
                print(f"[TTS] Converted to tensor: shape={audio_waveform.shape}, dtype={audio_waveform.dtype}")

            # 4. Existing block for non-tensor types (e.g. custom objects with .numpy())
            # This should now primarily handle cases not covered by list processing or direct ndarray/tensor returns
            if not isinstance(audio_waveform, torch.Tensor):
                print(f"[TTS] audio_waveform is not a tensor. Type: {type(audio_waveform)}. Attempting conversion.")
                if hasattr(audio_waveform, 'numpy'): # For types that have a .numpy() method
                    print(f"[TTS] Using numpy() method")
                    audio_numpy = audio_waveform.numpy()
                elif hasattr(audio_waveform, 'detach') and hasattr(audio_waveform.detach(), 'cpu') and hasattr(audio_waveform.detach().cpu(), 'numpy'): # For types that need detach().cpu().numpy()
                    print(f"[TTS] Using detach().cpu().numpy()")
                    audio_numpy = audio_waveform.detach().cpu().numpy()
                # Add more specific conversions if other types are encountered
                else:
                    # If it's still not a tensor here, and not np.ndarray (handled above), and not list (handled above)
                    # it might be an unexpected type.
                    print(f"[TTS] audio_waveform is of unexpected type {type(audio_waveform)}. Trying direct np.array conversion.")
                    try:
                        audio_numpy = np.array(audio_waveform, dtype=object) # Try converting, then check dtype
                        if audio_numpy.dtype == object:
                            print(f"[TTS] Warning: audio_numpy is of dtype 'object'. This might lead to issues.")
                            # Attempt to flatten if it's a list of lists/arrays of numbers
                            if isinstance(audio_numpy[0], (list, np.ndarray)): 
                                print(f"[TTS] Attempting to flatten and convert object array.")
                                audio_numpy = np.concatenate([np.array(item, dtype=np.float32) for item in audio_numpy.tolist()]).astype(np.float32)
                            else:
                                raise HTTPException(status_code=500, detail=f"Audio data is of problematic object type after conversion: {audio_numpy.dtype}")
                    except Exception as e_direct_np:
                        print(f"[TTS] Direct np.array conversion failed: {e_direct_np}")
                        raise HTTPException(status_code=500, detail=f"Audio data is of unsupported type {type(audio_waveform)} and could not be converted.")

                print(f"[TTS] Numpy array before astype: dtype={audio_numpy.dtype}, shape={audio_numpy.shape if hasattr(audio_numpy, 'shape') else 'N/A'}")
                audio_numpy = audio_numpy.astype(np.float32) # Ensure float32
                print(f"[TTS] Converted numpy array dtype: {audio_numpy.dtype}, shape={audio_numpy.shape}")
                audio_waveform = torch.from_numpy(audio_numpy)
                print(f"[TTS] Converted to tensor: shape={audio_waveform.shape}, dtype={audio_waveform.dtype}")


            if audio_waveform is None: # Should be redundant if errors are raised above, but good check.
                raise HTTPException(
                    status_code=500,
                    detail="Model failed to generate audio waveform or waveform became None after processing"
                )

            print(f"[TTS] Processing audio data...")
            try:
                # Ensure we have a tensor before checking dimensions
                if not isinstance(audio_waveform, torch.Tensor):
                    # This should ideally not be reached if the above logic is comprehensive
                    print(f"[TTS] CRITICAL: audio_waveform is NOT a tensor before dimension check. Type: {type(audio_waveform)}. This indicates a flaw in processing logic.")
                    # Attempt a final conversion, though this signifies an issue
                    try:
                        audio_waveform = torch.tensor(np.array(audio_waveform), dtype=torch.float32)
                        print(f"[TTS] Last resort conversion to tensor successful. Shape: {audio_waveform.shape}, dtype: {audio_waveform.dtype}")
                    except Exception as e_final_conv:
                         print(f"[TTS] Last resort conversion to tensor FAILED: {e_final_conv}")
                         raise HTTPException(status_code=500, detail=f"Audio data could not be converted to a tensor before saving. Type was {type(audio_waveform)}.")


                # Add channel dimension if needed
                if audio_waveform.ndim == 1:
                    print(f"[TTS] Adding channel dimension to audio data")
                    audio_waveform = audio_waveform.unsqueeze(0)
                elif audio_waveform.ndim > 2: # E.g. if it was a batch and got concatenated to (batch_size, channels, length)
                    print(f"[TTS] Audio waveform has more than 2 dimensions (shape: {audio_waveform.shape}). Assuming first dim is batch, taking first element.")
                    audio_waveform = audio_waveform[0]
                    if audio_waveform.ndim == 1: # Check again after taking first element
                        audio_waveform = audio_waveform.unsqueeze(0)


                print(f"[TTS] Final audio data shape for saving: {audio_waveform.shape}")
                print(f"[TTS] Final audio data type for saving: {audio_waveform.dtype}")
                print(
                    f"[TTS] Final audio data device for saving: {audio_waveform.device}")

                # 5. Final Data Type Check and Conversion before torchaudio.save
                print(f"[TTS] Before .cpu(): audio_waveform type={type(audio_waveform)}, dtype={audio_waveform.dtype if hasattr(audio_waveform, 'dtype') else 'N/A'}, device={audio_waveform.device if hasattr(audio_waveform, 'device') else 'N/A'}, shape={audio_waveform.shape if hasattr(audio_waveform, 'shape') else 'N/A'}")
                if not isinstance(audio_waveform, torch.Tensor):
                    print(f"[TTS] ERROR: audio_waveform is not a tensor just before .cpu() call. Type: {type(audio_waveform)}")
                    # This indicates a severe logic error if reached.
                    raise HTTPException(status_code=500, detail="Audio data was not a tensor before final CPU conversion.")
                audio_data_cpu = audio_waveform.float().cpu() # Ensure float32

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
                # 6. Add Logging before torchaudio.save
                print(f"[TTS] Attempting torchaudio.save with audio_data_cpu: type={type(audio_data_cpu)}, dtype={audio_data_cpu.dtype}, shape={audio_data_cpu.shape}, device={audio_data_cpu.device}")
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


@app.get("/health")
async def health_check():
    """
    Health check endpoint to monitor API status.
    """
    try:
        if model_translator is None:
            return {"status": "error", "message": "Model not loaded"}

        # Try a simple prediction to verify model is working
        test_text = "Hello"
        result = model_translator.predict(
            input=test_text,
            task_str="t2tt",
            src_lang="eng",
            tgt_lang="eng"
        )

        return {
            "status": "healthy",
            "model_loaded": True,
            "gpu_available": torch.cuda.is_available(),
            "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "model_loaded": model_translator is not None,
            "gpu_available": torch.cuda.is_available()
        }

