FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY grok-admaster/server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend (Internal Brain)
COPY grok-admaster/server/ ./server/

# Copy agent skills (skill instructions for the autonomous operator)
COPY .agent/ ./server/.agent/

# Set working directory to server
WORKDIR /app/server

# Expose the API port
EXPOSE 8000

# Default: Run the API server
# Override with CMD to run the autonomous operator instead
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
