# SeamlessM4T V2 Services: API and Streamlit UI

## Overview

This project provides a suite of services for accessing the capabilities of Meta's Seamless M4T v2 Large model. It includes:

*   **`seamless_api`**: A backend FastAPI service that exposes endpoints for Text-to-Speech (TTS), Speech-to-Text (STT/ASR), and Text-to-Text Translation.
*   **`seamless_st`**: A frontend Streamlit application that provides a user-friendly interface to interact with the `seamless_api` for all its functionalities.

The core functionalities offered are:
*   **Text-to-Speech (TTS)**: Generate speech from text in various languages.
*   **Speech-to-Text (STT/ASR)**: Transcribe speech from audio into text, with language detection.
*   **Text-to-Text Translation**: Translate text between supported languages.

## Prerequisites

To run this project, you will need the following installed on your system:

*   **Docker**: For containerizing and running the services.
*   **Docker Compose**: For orchestrating the multi-container application (usually included with Docker Desktop).
*   **NVIDIA GPU**: An NVIDIA GPU with the latest drivers installed is required for the `seamless_api` service to perform model inference efficiently.
*   **NVIDIA Container Toolkit**: To enable Docker to use NVIDIA GPUs.

## Project Structure

The project is organized as follows:

```
seamless_project/
├── seamless_api/       # FastAPI backend service, model loading, and API logic
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── seamless_st/        # Streamlit frontend service and UI logic
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── whl/                # Contains the pre-built wheel for seamless_communication
│   └── seamless_communication-1.0.0-py3-none-any.whl
└── docker-compose.yml  # Docker Compose file for orchestrating the services
└── README.md           # This file
```

## Setup & Configuration

*   **Seamless Communication Wheel**: The required `seamless_communication-1.0.0-py3-none-any.whl` package is included in the `whl/` directory and is automatically installed when building the `seamless_api` Docker image.
*   **Model Caching**: Model assets (like the Seamless M4T v2 Large model weights) downloaded by Hugging Face Transformers are cached in a Docker named volume called `hf_cache`. This ensures that models are not re-downloaded every time the service starts.
*   **Environment Variables**:
    *   `HF_HOME`: This is set to `/home/user/.cache/huggingface` within the `seamless_api` service (via `docker-compose.yml`) to direct Hugging Face to use the named volume for caching.
*   **GPU Allocation**: The `docker-compose.yml` file is configured to allocate one NVIDIA GPU to the `seamless_api` service. If you need to adjust this (e.g., use a specific GPU or change the count), you can modify the `deploy` section within the `seamless_api` service definition in `docker-compose.yml`.

## How to Run

1.  **Clone the repository** (if you haven't already).
2.  **Navigate to the project root directory**: `cd path/to/seamless_project`
3.  **Build and run the services using Docker Compose**:
    *   To run in detached mode (in the background):
        ```bash
        docker-compose up --build -d
        ```
    *   To run in the foreground and see logs directly (useful for the first run or debugging):
        ```bash
        docker-compose up --build
        ```

4.  **Access the services**:
    *   **Streamlit UI**: Open your web browser and go to `http://localhost:8501`
    *   **FastAPI (API Endpoints)**: The API is accessible at `http://localhost:8000`. You can use tools like Postman, curl, or custom scripts to interact with it directly. Main endpoints include:
        *   `GET /`: Root endpoint to check API status.
        *   `POST /tts`: Text-to-Speech. Expects JSON: `{"text": "...", "target_language": "..."}`. Returns WAV audio.
        *   `POST /stt`: Speech-to-Text. Expects a multipart form with an `audio_file`. Returns JSON: `{"text": "...", "language": "..."}`.
        *   `POST /translate`: Text-to-Text Translation. Expects JSON: `{"text": "...", "source_language": "...", "target_language": "..."}`. Returns JSON: `{"translated_text": "..."}`.

## How to Stop

To stop the running services and remove the containers:

1.  Navigate to the project root directory (`seamless_project`).
2.  Run the command:
    ```bash
    docker-compose down
    ```
    If you also want to remove the named volume `hf_cache` (which will delete cached models), you can use `docker-compose down -v`.

## Services Description

### `seamless_api` (FastAPI Service)

*   **Description**: This service provides a RESTful API for accessing the Seamless M4T v2 Large model. It handles model loading (on GPU if available) and exposes endpoints for various speech and text processing tasks.
*   **Main API Endpoints**:
    *   `GET /`: Checks API status and model loading status.
    *   `POST /tts`: Converts text to speech for a given target language.
    *   `POST /stt`: Transcribes speech from an audio file to text and detects the audio's language.
    *   `POST /translate`: Translates text from a source language to a target language.

### `seamless_st` (Streamlit Service)

*   **Description**: This service provides an interactive web interface built with Streamlit to use the functionalities of the `seamless_api`.
*   **Features**:
    *   **Translation Page**: Allows users to translate text or recorded/uploaded audio from a source language to a target language. Includes audio recording and file upload.
    *   **Text-to-Speech (TTS) Page**: Enables users to input text, select a target language, and generate speech. Users can play the audio directly or download it as a WAV file.
    *   **Speech-to-Text (STT/ASR) Page**: Allows users to record live audio or upload an audio file for transcription. Displays the transcribed text and the detected language.

## Notes

*   **First Run & Model Download**: The first time you run `docker-compose up --build`, the `seamless_api` service will download the Seamless M4T v2 Large model weights from Hugging Face. This model is several gigabytes in size, so the initial startup may take a significant amount of time depending on your internet connection. Subsequent runs will be much faster as the model will be loaded from the `hf_cache` Docker volume.
*   **STT/ASR Language Detection**: The STT/ASR endpoint (`/stt`) is designed to detect the language of the input audio and provide the transcript in that language. The underlying `predict` method of the model library is called with `tgt_lang="eng"` as a required parameter, but the returned language code is expected to be the actual detected source language. This behavior should be generally effective but has been noted as a point for confirmation during thorough testing across various languages.
*   **Resource Usage**: Running large AI models like Seamless M4T v2 is resource-intensive, particularly on the GPU. Ensure your system meets the necessary hardware requirements.
