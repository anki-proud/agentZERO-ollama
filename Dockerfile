# agentZERO-ollama Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data models

# Set environment variables
ENV PYTHONPATH=/app
ENV OLLAMA_HOST=0.0.0.0:11434
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose ports
EXPOSE 11434 8501

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting agentZERO-ollama..."\n\
\n\
# Start Ollama in background\n\
echo "Starting Ollama server..."\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
# Wait for Ollama to be ready\n\
echo "Waiting for Ollama to be ready..."\n\
while ! curl -s http://localhost:11434/api/tags > /dev/null; do\n\
    sleep 1\n\
done\n\
echo "Ollama is ready!"\n\
\n\
# Pull required models if they dont exist\n\
echo "Checking for required models..."\n\
\n\
if ! ollama list | grep -q "smollm:135m"; then\n\
    echo "Pulling smollm:135m..."\n\
    ollama pull smollm:135m\n\
fi\n\
\n\
if ! ollama list | grep -q "smollm:360m"; then\n\
    echo "Pulling smollm:360m..."\n\
    ollama pull smollm:360m\n\
fi\n\
\n\
if ! ollama list | grep -q "smollm:1.7b"; then\n\
    echo "Pulling smollm:1.7b..."\n\
    ollama pull smollm:1.7b\n\
fi\n\
\n\
echo "All models ready!"\n\
\n\
# Start Streamlit app\n\
echo "Starting Streamlit app..."\n\
streamlit run ui/app.py --server.address=0.0.0.0 --server.port=8501 &\n\
STREAMLIT_PID=$!\n\
\n\
echo "agentZERO-ollama is running!"\n\
echo "Ollama API: http://localhost:11434"\n\
echo "Web Interface: http://localhost:8501"\n\
\n\
# Wait for either process to exit\n\
wait -n $OLLAMA_PID $STREAMLIT_PID\n\
\n\
# Kill remaining processes\n\
kill $OLLAMA_PID $STREAMLIT_PID 2>/dev/null || true\n\
' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:11434/api/tags && curl -f http://localhost:8501/_stcore/health

# Default command
CMD ["/app/start.sh"]

