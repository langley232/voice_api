import streamlit as st
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import queue
import threading

# API URL (assuming Docker Compose networking)
API_URL = "http://seamless_api:8000"

# Page Configuration
st.set_page_config(layout="wide")
st.title("Voice Translation and Speech Services")

# --- Translation Section ---
st.header("Translate Speech / Text")

# Language Selection
# Using a predefined list of common languages for simplicity.
# These should ideally match the languages supported by the model.
SUPPORTED_LANGUAGES = {
    "English": "eng", "Spanish": "spa", "French": "fra",
    "German": "deu", "Mandarin Chinese": "cmn", "Arabic": "ara",
    "Hindi": "hin", "Russian": "rus", "Japanese": "jpn", "Korean": "kor"
    # Add more as needed
}
# We'll display full names but use language codes for the API
lang_display_names = list(SUPPORTED_LANGUAGES.keys())
lang_codes = list(SUPPORTED_LANGUAGES.values())

col1, col2 = st.columns(2)
with col1:
    source_language_display = st.selectbox(
        "Source Language",
        options=lang_display_names,
        key='translate_source_lang_display',
        index=0  # Default to English
    )
    source_language = SUPPORTED_LANGUAGES[source_language_display]

with col2:
    target_language_display = st.selectbox(
        "Target Language",
        options=lang_display_names,
        key='translate_target_lang_display',
        index=1  # Default to Spanish
    )
    target_language = SUPPORTED_LANGUAGES[target_language_display]

# Audio Input
st.subheader("Record Audio to Translate")

# Initialize session state for audio data if not exists
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'asr_audio_data' not in st.session_state:
    st.session_state.asr_audio_data = None

# Audio recording callbacks


def translation_audio_frame_callback(frame):
    try:
        if frame is not None:
            # Convert frame to numpy array
            audio = frame.to_ndarray()
            # Store in session state
            st.session_state.audio_data = audio
            print(
                f"[Translation] Received audio frame of shape: {audio.shape}")
        return frame
    except Exception as e:
        print(f"[Translation] Error in audio frame callback: {e}")
        return frame


def asr_audio_frame_callback(frame):
    try:
        if frame is not None:
            # Convert frame to numpy array
            audio = frame.to_ndarray()
            # Store in session state
            st.session_state.asr_audio_data = audio
            print(f"[ASR] Received audio frame of shape: {audio.shape}")
        return frame
    except Exception as e:
        print(f"[ASR] Error in audio frame callback: {e}")
        return frame


# WebRTC streamer for audio recording
webrtc_ctx = webrtc_streamer(
    key="translate_audio",
    mode="recording",  # Using string value instead of enum
    audio_receiver_size=1024,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}]},
    audio_frame_callback=translation_audio_frame_callback,
)

# Process recorded audio when available
if st.session_state.audio_data is not None:
    st.audio(st.session_state.audio_data, format="audio/wav")

    # Convert numpy array to bytes for API request
    audio_bytes = st.session_state.audio_data.tobytes()

    if st.button("Transcribe Recorded Audio", key="transcribe_button_audio"):
        files = {'audio_file': ('recorded_audio.wav',
                                audio_bytes, 'audio/wav')}
        try:
            st.info("Transcribing audio...")
            stt_response = requests.post(
                f"{API_URL}/stt", files=files, timeout=60)
            if stt_response.status_code == 200:
                stt_data = stt_response.json()
                transcribed_text_from_audio = stt_data.get('text')
                detected_lang_code = stt_data.get('language')

                # Try to match detected lang code with our display names
                detected_lang_display = detected_lang_code
                for name, code in SUPPORTED_LANGUAGES.items():
                    if code == detected_lang_code:
                        detected_lang_display = name
                        break

                st.success(
                    f"Transcription successful! Detected Language: {detected_lang_display}")
                # Pre-fill text area
                st.session_state.text_to_translate_input = transcribed_text_from_audio
                st.write(f"Transcribed text: {transcribed_text_from_audio}")

                # Update source language if different from current selection and detected
                if source_language != detected_lang_code and detected_lang_code in lang_codes:
                    st.session_state.translate_source_lang_display = detected_lang_display
                    source_language = detected_lang_code
                    st.info(
                        f"Source language automatically updated to: {detected_lang_display}")

            else:
                st.error(
                    f"STT Error: {stt_response.status_code} - {stt_response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"STT Request failed: {e}")
        except Exception as e_gen:
            st.error(f"An unexpected error occurred during STT: {e_gen}")

# Text Input (Alternative)
st.subheader("Or Type Text to Translate")
if 'text_to_translate_input' not in st.session_state:
    st.session_state.text_to_translate_input = ""

text_input = st.text_area(
    "Text to translate:",
    value=st.session_state.text_to_translate_input,
    key='text_to_translate_input_area',  # Use a different key for the widget itself
    height=150
)


# Translate Button
if st.button("Translate", key='translate_button'):
    # Determine the text to translate
    final_text_to_translate = None

    # Priority to newly transcribed text if available and text area hasn't been manually edited since
    if transcribed_text_from_audio and st.session_state.text_to_translate_input == transcribed_text_from_audio:
        final_text_to_translate = transcribed_text_from_audio
        st.info("Using transcribed text from recent recording for translation.")
    # Text from text_area (could be from STT or manually typed)
    elif text_input:
        final_text_to_translate = text_input
        st.info("Using text from the text area for translation.")
    # If audio_bytes exists but wasn't explicitly transcribed via its button,
    # we could add logic here to transcribe it now, but current flow requires explicit STT button click.
    # This simplifies the logic for now. User must click "Transcribe Recorded Audio" first.

    if not final_text_to_translate:
        st.error("Please record audio and transcribe it, or type text to translate.")
    else:
        st.markdown("---")
        st.subheader("Translation Result")
        st.write(f"**Original Text ({source_language_display}):**")
        st.markdown(f"> {final_text_to_translate}")

        payload = {
            "text": final_text_to_translate,
            "source_language": source_language,  # Use the code
            "target_language": target_language  # Use the code
        }
        try:
            st.info("Translating text...")
            translate_response = requests.post(
                f"{API_URL}/translate", json=payload, timeout=60)  # Increased timeout
            if translate_response.status_code == 200:
                translated_text = translate_response.json().get('translated_text')
                st.success(f"**Translated Text ({target_language_display}):**")
                st.markdown(f"> {translated_text}")
            else:
                st.error(
                    f"Translation Error: {translate_response.status_code} - {translate_response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Translation Request failed: {e}")
        except Exception as e_gen:
            st.error(
                f"An unexpected error occurred during translation: {e_gen}")

st.markdown("---")
st.info("""
**Instructions:**
1.  Select the **Source Language** and **Target Language**.
2.  **To translate speech:** Click the microphone icon to record. Click again to stop. 
    Then click "Transcribe Recorded Audio". The transcribed text will appear.
3.  **To translate text:** Type or paste text into the text area.
4.  Click the **Translate** button.
""")

# Placeholder for other functionalities (TTS Page) later
# e.g., if st.sidebar.button("Go to TTS Page"):
#    st.session_state.current_page = "TTS" (example of page navigation)


st.markdown("---")  # Visual separator

# --- Text-to-Speech (TTS) Section ---
st.header("Text-to-Speech (TTS)")

tts_text_input = st.text_area(
    "Enter text for TTS:", key='tts_text', height=100)

# Add source language selection
tts_source_language_display = st.selectbox(
    "Source Language of Text",
    options=lang_display_names,
    key='tts_source_lang_display',
    index=0  # Default to English
)
tts_source_language = SUPPORTED_LANGUAGES[tts_source_language_display]

# Use the same language selection as in Translation for consistency
tts_target_language_display = st.selectbox(
    "Target Language for Speech",
    options=lang_display_names,
    key='tts_target_lang_display',
    index=0  # Default to English
)
tts_target_language = SUPPORTED_LANGUAGES[tts_target_language_display]

tts_output_option = st.radio(
    "Output Option",
    ["Play audio", "Download WAV"],
    key='tts_output_type',
    horizontal=True
)

if st.button("Generate Speech", key='tts_generate_button'):
    if not tts_text_input.strip():
        st.error("Please enter some text to generate speech.")
    else:
        payload = {
            "text": tts_text_input,
            "source_language": tts_source_language,  # Added source language
            "target_language": tts_target_language
        }
        try:
            st.info("Generating speech, please wait...")
            # The /tts endpoint returns the audio file directly in the response body
            # Increased timeout for potentially long generation
            tts_response = requests.post(
                f"{API_URL}/tts", json=payload, timeout=120)

            if tts_response.status_code == 200:
                audio_content_bytes = tts_response.content  # Raw bytes of the WAV file

                if tts_output_option == "Play audio":
                    st.audio(audio_content_bytes,
                             format="audio/wav", start_time=0)
                    st.success("Speech generated successfully!")
                elif tts_output_option == "Download WAV":
                    st.download_button(
                        label="Download WAV",
                        data=audio_content_bytes,
                        file_name="tts_output.wav",
                        mime="audio/wav"
                    )
                    st.success(
                        "Speech generated! Click the button above to download.")
            else:
                try:
                    error_detail = tts_response.json().get("detail", tts_response.text)
                except ValueError:  # If response is not JSON
                    error_detail = tts_response.text
                st.error(
                    f"TTS Generation Error: {tts_response.status_code} - {error_detail}")

        except requests.exceptions.Timeout:
            st.error(
                "TTS Request timed out. The model might be taking too long to respond.")
        except requests.exceptions.RequestException as e:
            st.error(f"TTS Request failed: {e}")
        except Exception as e_gen:
            st.error(
                f"An unexpected error occurred during TTS generation: {e_gen}")

st.markdown("---")  # Visual separator

# --- Speech-to-Text (STT/ASR) Section ---
st.header("Speech-to-Text (STT/ASR Transcription)")

st.subheader("Record Audio for Transcription")
asr_webrtc_ctx = webrtc_streamer(
    key="asr_audio_recorder",
    mode="recording",  # Using string value instead of enum
    audio_receiver_size=1024,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}]},
    audio_frame_callback=asr_audio_frame_callback,
)

st.subheader("Or Upload an Audio File for Transcription")
asr_uploaded_file = st.file_uploader(
    "Upload audio file",
    type=['wav', 'mp3', 'ogg'],  # Common audio types
    key='asr_file_uploader'
)

# Determine which audio source to use (recorder takes precedence if both are somehow populated)
final_asr_audio_bytes = None
audio_source_message = None

if st.session_state.asr_audio_data is not None:
    final_asr_audio_bytes = st.session_state.asr_audio_data.tobytes()
    st.info("Using audio from live recording.")
    st.audio(st.session_state.asr_audio_data, format="audio/wav")
    audio_source_message = "from live recording"
elif asr_uploaded_file:
    final_asr_audio_bytes = asr_uploaded_file.getvalue()
    st.info(f"Using audio from uploaded file: {asr_uploaded_file.name}")
    st.audio(final_asr_audio_bytes)  # Let Streamlit infer format for preview
    audio_source_message = f"from uploaded file '{asr_uploaded_file.name}'"


if st.button("Transcribe Speech", key='asr_transcribe_button'):
    if not final_asr_audio_bytes:
        st.error("Please record audio or upload an audio file to transcribe.")
    else:
        st.info(f"Transcribing speech {audio_source_message}, please wait...")
        files = {'audio_file': (
            'asr_audio.wav', final_asr_audio_bytes, 'audio/wav')}

        try:
            stt_response = requests.post(
                f"{API_URL}/stt", files=files, timeout=90)  # Increased timeout

            if stt_response.status_code == 200:
                response_data = stt_response.json()
                transcribed_text = response_data.get('text')
                detected_language_code = response_data.get('language')

                # Map detected language code to full name if possible
                detected_language_display = detected_language_code
                for name, code in SUPPORTED_LANGUAGES.items():  # Reuse from Translation section
                    if code == detected_language_code:
                        detected_language_display = name
                        break

                st.success("Transcription successful!")
                st.write(
                    f"**Detected Language:** {detected_language_display} ({detected_language_code})")
                st.text_area("Transcribed Text:", value=transcribed_text,
                             height=200, key='asr_transcribed_output', disabled=False)
            else:
                try:
                    error_detail = stt_response.json().get("detail", stt_response.text)
                except ValueError:
                    error_detail = stt_response.text
                st.error(
                    f"STT/ASR Error: {stt_response.status_code} - {error_detail}")

        except requests.exceptions.Timeout:
            st.error(
                "STT/ASR Request timed out. The model might be taking too long or the audio is too long.")
        except requests.exceptions.RequestException as e:
            st.error(f"STT/ASR Request failed: {e}")
        except Exception as e_gen:
            st.error(f"An unexpected error occurred during STT/ASR: {e_gen}")
