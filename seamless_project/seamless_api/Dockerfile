# 1. Base Image and Python Setup
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    # python build dependencies \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    # ffmpeg for audio processing (used by torchaudio, fairseq2, etc.) \
    ffmpeg \
    # fairseq2 dependency \
    libsndfile-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. User and Workdir
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:${PATH}
WORKDIR ${HOME}/app

# Python installation using pyenv
RUN curl https://pyenv.run | bash
ENV PATH=${HOME}/.pyenv/shims:${HOME}/.pyenv/bin:${PATH}
ARG PYTHON_VERSION=3.10.13
RUN pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION} && \
    pyenv rehash && \
    pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir "huggingface-hub==0.19.3" "hf-transfer==0.1.4"

# 3. Dependencies Installation

# Copy the WHL directory from build context (seamless_project/whl)
COPY whl ${HOME}/app/whl/

# Copy the API-specific requirements.txt from build context (seamless_project/seamless_api/requirements.txt)
COPY seamless_api/requirements.txt ${HOME}/app/requirements.txt

# Install Python dependencies from the API's requirements.txt
RUN pip install --no-cache-dir -r ${HOME}/app/requirements.txt

# Install fairseq2
RUN pip install --no-cache-dir fairseq2 --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/pt2.1.0/cu121

# Install the seamless_communication wheel
RUN pip install --no-cache-dir ${HOME}/app/whl/seamless_communication-1.0.0-py3-none-any.whl

# 4. Application Code
# Copy the seamless_api application code from build context (seamless_project/seamless_api/)
# This should be one of the last COPY operations to leverage Docker cache for dependencies
COPY seamless_api/ ${HOME}/app/

# 5. Environment Variables
ENV PYTHONPATH=${HOME}/app \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # Kept Gradio variables as per instruction, can be reviewed later
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    TQDM_POSITION=-1 \
    TQDM_MININTERVAL=1 \
    SYSTEM=spaces # This might be specific to HF Spaces, kept for now

# 6. CMD
# Assuming main.py will contain the FastAPI app instance named 'app'
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
