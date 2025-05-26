FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt directly from the build context (which is seamless_st/)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the build context (seamless_st/) into the container's /app
# This will copy app.py, Dockerfile, requirements.txt, etc.
# If you only want app.py, you'd specify 'COPY app.py .'
COPY . .

# Command to run your Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
