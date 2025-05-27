FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt directly from the build context (which is seamless_st/)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- DEBUGGING STEPS START ---
# 1. List all installed Python packages to check the version of streamlit_webrtc
RUN echo "--- Listing installed Python packages in seamless_st ---" && \
    pip list | grep streamlit-webrtc || echo "streamlit-webrtc not found in pip list!"

# 2. Find the installation path of streamlit_webrtc and list its contents
RUN echo "--- Inspecting streamlit_webrtc installation directory ---" && \
    SITE_PACKAGES_PATH=$(python -c "import site; print(site.getsitepackages()[0])") && \
    ls -l "${SITE_PACKAGES_PATH}/streamlit_webrtc" || \
    echo "streamlit_webrtc directory not found in site-packages!"

# 3. Attempt to import webrtc_streamer directly during the build
# This will cause the build to fail if the module cannot be imported, providing early feedback.
RUN echo "--- Attempting to import webrtc_streamer from streamlit_webrtc ---" && \
    python -c "from streamlit_webrtc import webrtc_streamer; print('webrtc_streamer import successful!')"
# --- DEBUGGING STEPS END ---

# Copy all files from the build context (seamless_st/) into the container's /app
COPY . .

# Command to run your Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


