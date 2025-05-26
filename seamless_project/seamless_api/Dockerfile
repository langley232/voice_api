FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
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
    ffmpeg \
    libsndfile-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:${PATH}
WORKDIR ${HOME}/app

RUN curl https://pyenv.run | bash
ENV PATH=${HOME}/.pyenv/shims:${HOME}/.pyenv/bin:${PATH}
ARG PYTHON_VERSION=3.10.13
RUN pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION} && \
    pyenv rehash && \
    # Ensure pip, setuptools, and wheel are fully updated before installing other packages
    pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir "huggingface-hub==0.19.3" "hf-transfer==0.1.4"


COPY whl ${HOME}/app/whl/
COPY seamless_api/requirements.txt ${HOME}/app/requirements.txt

# Install fairseq2 and then the local wheel file
RUN pip install --no-cache-dir -r ${HOME}/app/requirements.txt && \
    pip install --no-cache-dir fairseq2 --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/pt2.1.0/cu121 && \
    pip install --no-cache-dir ${HOME}/app/whl/seamless_communication-1.0.0-py3-none-any.whl

# --- DEBUGGING STEPS START ---
# 1. Verify the wheel file is present in the expected location before installation
RUN echo "--- Verifying seamless_communication wheel file presence ---" && \
    ls -l ${HOME}/app/whl/seamless_communication-1.0.0-py3-none-any.whl || echo "Wheel file NOT found!"

# 2. List all installed Python packages to check if seamless_communication appears
RUN echo "--- Listing installed Python packages ---" && \
    pip list

# 4. Inspect the site-packages directory for seamless_communication (NEW STEP)
# This will show if the package directory exists and what its contents are.
RUN echo "--- Inspecting site-packages for seamless_communication ---" && \
    ls -l ${HOME}/.pyenv/versions/${PYTHON_VERSION}/lib/python${PYTHON_VERSION%.*}/site-packages/seamless_communication* || \
    echo "seamless_communication directory not found in site-packages!"

# 3. Attempt to import the problematic module directly during the build
# This will cause the build to fail if the module cannot be imported, providing early feedback.
RUN echo "--- Attempting to import seamless_communication.models.inference ---" && \
    python -c "import seamless_communication.models.inference; print('Import successful!')"
# --- DEBUGGING STEPS END ---

COPY seamless_api/ ${HOME}/app/

ENV PYTHONPATH=${HOME}/app \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    TQDM_POSITION=-1 \
    TQDM_MININTERVAL=1 \
    SYSTEM=spaces

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

