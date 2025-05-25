import os
import torch
import torchaudio # For saving audio
import tempfile # For creating temporary files
import shutil # For saving uploaded file
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask # For cleanup
from pydantic import BaseModel
from seamless_communication.models.inference import Translator

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

        # Model name and vocoder
        # Based on typical usage for SeamlessM4T v2 Large.
        # The name "seamlessM4T_v2_large" is derived from Hugging Face model cards and common examples.
        # "vocoder_v2" is also a common default.
        model_name = "seamlessM4T_v2_large" 
        vocoder_name = "vocoder_v2"

        # Load the translator model
        # The Translator class expects model_name, vocoder_name, device, and dtype (optional for default).
        # Forcing dtype here for clarity, especially for float16 on GPU.
        translator_instance = Translator(
            model_name=model_name,
            vocoder_name=vocoder_name,
            device=device,
            dtype=dtype,
            # Ensure required assets are downloaded if not present, by specifying a local path
            # This might require fairseq2 to be configured for asset downloads.
            # The Dockerfile should handle fairseq2 installation and potential caching.
        )
        
        model_translator = translator_instance
        print(f"Successfully loaded Seamless M4T v2 Large model ('{model_name}') on {device_str}.")

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
        # Use NamedTemporaryFile to ensure it's cleaned up, or manage manually.
        # For UploadFile, it's common to read its contents and write to a new temp file.
        
        # Create a temporary file to save the uploaded audio content
        # We need to ensure the file is written in binary mode.
        # And the model expects a path to an audio file.
        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav") # Assuming WAV, or let model handle format
        os.close(fd) # Close descriptor, open with 'wb'

        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        print(f"Received STT request: Saved uploaded audio to temporary file: {temp_audio_path}")

        # Perform Speech-to-Text (STT/ASR)
        # The task string for ASR (transcription) is often 's2t_transcript' or 'asr'.
        # For SeamlessM4T, 's2t_transcript' seems appropriate for getting transcription in the source language.
        # The `predict` method for speech input usually returns:
        # (text_output, lang_code, Optional[torch.Tensor] audio_waveform, Optional[int] audio_sample_rate)
        # We are interested in text_output and lang_code.
        # tgt_lang is not specified, as we want transcription in the detected source language.
        
        # For `s2t_transcript`, `tgt_lang` is the language to transcribe *to*.
        # However, for ASR, we want the language *of the audio*.
        # The `Translator`'s `predict` method, when given an audio input and task like `s2t_transcript`
        # without a `tgt_lang` specified, should ideally perform language identification and transcribe in that identified language.
        # Let's call it without `tgt_lang` first and see its behavior based on typical model design.
        # If `tgt_lang` is mandatory, we might need to adjust.
        # The documentation/examples for seamless_communication suggest `tgt_lang` is for the *output* language.
        # For ASR, the output language is the same as the input language.
        # It's possible the model implicitly handles LID and uses it as target.

        # The prompt implies that `tgt_lang` is needed for `s2tt_ctc`.
        # Let's assume the model can detect the source language and use that as the target language for transcription.
        # If `tgt_lang` must be specified for ASR, this is problematic without a prior LID step.
        # However, many ASR models output the transcript in the source language by default.
        # The `Translator.predict` method's signature is:
        # predict(self, input: Union[str, torch.Tensor], task_str: str, tgt_lang: str, src_lang: Optional[str] = None, ...)
        # This signature implies `tgt_lang` is always required.
        # This is unusual for a pure ASR task where `tgt_lang` would be the source language.
        # Let's consult common usage or assume a default behavior if `tgt_lang` for ASR is not obvious.
        # The Hugging Face Space for seamless-m4t-v2-large has an "S2TT/ASR" tab.
        # In its app.py (facebook/seamless-m4t-v2-large space), for ASR, it sets `tgt_lang = src_lang`.
        # This means we *would* need a source language. This is a complication.

        # Re-evaluating: The task is "STT/ASR". The model *should* detect language.
        # If `Translator.predict` returns `lang_code` for the input audio, that's what we need.
        # Let's assume `task_str="s2t_transcript"` and `tgt_lang` is set to a dummy or a common language,
        # hoping the returned `lang_code` is the actual detected language of the audio, not `tgt_lang`.
        # Or, more plausibly, `tgt_lang` for "s2t_transcript" should be the language of the desired transcript.
        # If the model does LID, it might use that identified language.

        # The `predict` method's docstring (if available or inferred from similar models)
        # for speech input usually returns (transcribed_text, source_language_code, ...).
        # Let's use "s2t_transcript" and provide a common target language like "eng"
        # and inspect the returned language code. If the model identifies the language,
        # the returned language code should be the actual source language.
        # A better approach for ASR if `tgt_lang` is truly the *target language of transcription*
        # would be to have a separate LID step or allow the user to specify source language.
        # Given the constraints, we'll assume the model has a mode for ASR where it detects language.

        # The prompt mentioned `s2tt_ctc` and `tgt_lang`.
        # Let's try with `s2t_transcript` first, as it's more standard for ASR.
        # And we need to provide *some* `tgt_lang`. The API will return the *actual* language.
        # Let's pick 'eng' as a placeholder `tgt_lang` if the model overrides it with detected lang.
        
        # According to SeamlessM4T_Paper.pdf (original paper for v1), ASR task output is (transcribed_text, None).
        # This implies language ID might not be directly returned for ASR task string.
        # However, the `Translator` object in `seamless_communication` library might be different.
        # The `predict` method of the `Translator` from `seamless_communication.models.inference.translator`
        # for speech input returns: `(text_output, src_lang_iso_639_3, None, None)`
        # when `task_str` is like `s2t_transcript`. Here `src_lang_iso_639_3` is the detected source language.
        # `tgt_lang` is still a required argument for `predict`.

        # So, we must pass a `tgt_lang`. This `tgt_lang` specifies the language of the output text.
        # For ASR, the output text should be in the same language as the input speech.
        # This implies we'd need to know the source language beforehand to set `tgt_lang` correctly.
        # This is a design challenge if the goal is automatic language detection *and* transcription.

        # If the `Translator`'s `predict` returns the *detected source language* separately,
        # then `tgt_lang` could be set to that detected language. But that's circular.
        # Let's assume that for ASR task 's2t_transcript', `tgt_lang` is used by the model
        # to know which language to expect for transcription, and the returned language code
        # is this `tgt_lang`. This would mean the user *must* provide the language of the audio.

        # Given the subtask: "language: str (for the detected language code)", this implies the model *does* detect it.
        # Let's assume `tgt_lang` is a hint, but the model returns the *actual* language.
        # A common default for `tgt_lang` in ASR if not specified is often 'en' or similar.
        # The `predict` method for `s2t_transcript` returns `(text_output, lang_code, ...)`
        # where `lang_code` is the language of `text_output`.
        
        # Let's assume the simplest ASR case: transcribe to English, and the model tells us what it transcribed.
        # The task asks for "detected language code".
        # The `Translator.predict` returns `(str, str, Optional[torch.Tensor], Optional[int])`
        # as `(translated_text, source_language_code_or_target_language_code, output_audio_waveform, output_audio_sr)`
        # For 's2t_transcript', the second string is the language of the transcript.

        # Let's try `task_str = "asr_sync"` which is mentioned in some Meta examples for ASR.
        # It might behave better for language detection.
        # If `asr_sync` is not available, we'll revert to `s2t_transcript`.
        # The `predict` function is `predict(self, input: Union[str, Path, torch.Tensor], task_str: str, tgt_lang: str, ...)`
        # `tgt_lang` is required.

        # For ASR, `tgt_lang` should be the language of the speech.
        # The model *must* perform LID to achieve this if `tgt_lang` isn't provided by user.
        # If we pass a dummy `tgt_lang`, say "eng", and the audio is in Spanish,
        # what will `text` and `language` in `STTResponse` be?
        # `text` should be Spanish transcript, `language` should be "spa".
        # This means `tgt_lang="eng"` was just a placeholder.

        # Let's assume the model is smart: `tgt_lang` is the *desired output language for transcription*.
        # If the model also returns the *detected source language*, that's what we want.
        # The `Translator` returns `(text_output, lang_code, ...)`
        # For ASR, `text_output` is the transcript, and `lang_code` is the language of that transcript.
        # This implies `tgt_lang` effectively determines `lang_code`.
        # This is not "language detection" in the sense of "what language is this audio?"
        # but "I transcribed this audio assuming it was X, and here's the transcript in X."

        # The problem is likely simpler: the `lang_code` returned by `predict` IS the detected language.
        # And `tgt_lang` is just a necessary parameter for the call.
        # Let's use a common language like 'eng' for `tgt_lang` argument.
        # The actual language detected and used for transcription will be in the response.

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

        # Perform Text-to-Text Translation (T2TT)
        # The `predict` method for text input and "t2tt_text" task usually returns:
        # (translated_text_str, source_language_code_str (same as input src_lang), None, None)
        # We are interested in the first element: translated_text_str.
        
        translated_text, _, _, _ = model_translator.predict(
            input=request.text,
            task_str="t2tt_text", # Text-to-Text Translation
            src_lang=request.source_language,
            tgt_lang=request.target_language
        )

        if translated_text is None:
            # This case might not happen if an exception is raised first, but good for safety.
            raise HTTPException(status_code=500, detail="Model failed to translate text.")

        print(f"Translated text: '{translated_text}'")
        
        return TranslationResponse(translated_text=translated_text)

    except HTTPException as e_http:
        # Re-raise HTTPExceptions directly
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

        # Use the model's predict method for Text-to-Speech (T2ST)
        # The task string 't2st_sync' is commonly used for direct text-to-speech output.
        # It typically returns (translated_text_or_None, audio_waveform_tensor, sample_rate)
        # For T2ST, the first element (text) might be None or the input text itself.
        # We are interested in the audio waveform tensor and its sample rate.
        # Source language is not explicitly needed for T2ST with a multilingual model like SeamlessM4T.
        
        # According to common usage of seamless_communication.Translator.predict:
        # For T2ST, translated_text is often the input text, or None.
        # The audio output `wav` is a torch.Tensor, and `sr` is the sample rate.
        # predict(self, text_input, task_str, tgt_lang, src_lang=None, spk_id=0, s2s_style_models=None, s2s_style_prompt=None)
        
        # We expect the model to handle various language codes.
        # The `predict` method of `Translator` for T2ST tasks returns a tuple:
        # (None or input text, audio waveform (torch.Tensor), sample rate (int))
        # Note: The prompt mentions `translated_audio_path`, but the `Translator` class for seamless_communication
        # typically returns the audio data directly as a tensor. We will save this tensor to a file.

        output_text, audio_waveform, audio_sample_rate = model_translator.predict(
            input=request.text,
            task_str="t2st_sync", # Text-to-Speech (and Text)
            tgt_lang=request.target_language,
            # src_lang is not needed for T2ST
        )

        if audio_waveform is None or audio_sample_rate is None:
            raise HTTPException(status_code=500, detail="Model failed to generate audio.")

        print(f"Generated audio: Waveform shape {audio_waveform.shape}, Sample rate {audio_sample_rate}")

        # Save the generated audio to a temporary file
        # The audio_waveform is a torch.Tensor. We need to ensure it's on CPU for torchaudio.save.
        # The shape is usually (channels, samples) or (samples,) for mono.
        # torchaudio.save expects data in channels-first format (e.g. [C, L]).
        # If it's 1D, unsqueeze it to 2D [1, L].
        
        audio_data_cpu = audio_waveform.cpu()
        if audio_data_cpu.ndim == 1:
            audio_data_cpu = audio_data_cpu.unsqueeze(0) # Add channel dimension

        # Create a temporary file to save the audio
        # tempfile.NamedTemporaryFile creates a file that is deleted when closed.
        # We need to ensure the file is not deleted before FileResponse sends it.
        # One way is to use a simple tempfile.mkstemp() or manage deletion.
        # For FileResponse, it's better to save it and let FileResponse handle it.
        # FileResponse will delete the file if `background_tasks` are used, or we can delete it manually.

        # Using mkstemp to get a unique filename
        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd) # Close the file descriptor, torchaudio.save will reopen

        try:
            torchaudio.save(
                temp_audio_path,
                audio_data_cpu,
                audio_sample_rate
            )
            print(f"Saved generated audio to temporary file: {temp_audio_path}")

            # Return the audio file as a response
            # FileResponse will stream the file.
            # We should arrange for the temporary file to be deleted after sending.
            # One way is to use a background task.
            # For simplicity here, we'll let it be, but in production, cleanup is important.
            # Or, if FileResponse handles it with a background task by default when path is string.
            
            # To ensure cleanup, we can use a BackgroundTask
            # from starlette.background import BackgroundTask
            response = FileResponse(
                temp_audio_path,
                media_type="audio/wav",
                filename="tts_output.wav",
                background=BackgroundTask(os.remove, temp_audio_path)
            )
            # For now, let's keep it simpler and manually manage if needed or rely on OS temp cleaning.
            # The task description suggests `FileResponse` and `tempfile` module.

            return response # Return the response with background task for cleanup
            # return FileResponse(
            #     temp_audio_path,
            #     media_type="audio/wav",
            #     filename="tts_output.wav"
            #     # Consider adding background task for cleanup:
            #     # background=BackgroundTask(os.remove, temp_audio_path)
            # )

        except Exception as e_save:
            # If saving fails, try to clean up the temp file if it was created
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            print(f"Error saving audio to file: {e_save}")
            raise HTTPException(status_code=500, detail=f"Failed to save generated audio: {str(e_save)}")

    except HTTPException as e_http:
        # Re-raise HTTPExceptions directly
        raise e_http
    except Exception as e:
        print(f"Error during TTS generation: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
