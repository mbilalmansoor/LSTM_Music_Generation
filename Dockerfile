FROM python:3.10-slim

# Install system dependencies for audio synthesis
RUN apt-get update && apt-get install -y \
    abcmidi \
    timidity \
    fluid-soundfont-gm \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" for security compliance on Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install python dependencies
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application files (including your app.py and weights file!)
COPY --chown=user . $HOME/app

# Expose Gradio's default port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]