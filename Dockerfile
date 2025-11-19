FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed by imaging and some wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libjpeg62-turbo \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy optional requirements file (if you maintain one)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies. If a requirements.txt exists it will be used,
# otherwise a sensible default set is installed.
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir streamlit tensorflow pillow pymupdf numpy h5py; \
    fi

# Copy application and model
COPY streamlit_app_v2.py /app/streamlit_app_v2.py
COPY best_model_fold_4.h5 /app/best_model_fold_4.h5

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app_v2.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
