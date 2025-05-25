# 1. Base Image
FROM python:3.10-slim

# 2. Working Directory
WORKDIR /app

# 3. Environment variables (optional, but good practice)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 4. Copy Requirements
# The build context will be seamless_project.
# So, we need to copy from seamless_st/requirements.txt relative to that context.
COPY seamless_st/requirements.txt .

# 5. Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
# This will copy everything from seamless_project/seamless_st/ into /app in the image.
COPY seamless_st/ .

# 7. Expose Port
EXPOSE 8501

# 8. Command to run the application
# Assuming the main Streamlit application file will be app.py
CMD ["streamlit", "run", "app.py"]
