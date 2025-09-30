#!/usr/bin/env python3
"""
Modal deployment wrapper for Voice Changer FastAPI app.
Wraps the existing voice_changer.py FastAPI application for serverless deployment.
"""

import modal

# Define the Modal app
app = modal.App("voice-changer-modal")

# Create image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install system dependencies
    .apt_install("ffmpeg")
    # Install Python dependencies matching pyproject.toml
    .pip_install(
        [
            "fastapi>=0.111",
            "uvicorn>=0.29",
            "aiortc>=1.7",
            "av>=11.0",
            "pylibsrtp>=0.10",
            "librosa>=0.10",
            "soundfile>=0.12",
            "numpy>=1.26",
            "scipy>=1.11",
            "python-multipart>=0.0.6",
            "aiofiles>=23.2",
            "pydub>=0.25",
        ]
    )
    # Add local source files
    .add_local_python_source(
        modal.Mount.from_local_file("voice_changer.py", remote_path="/root/voice_changer.py")
    )
)


@app.function(
    image=image,
    # Configure for concurrent real-time processing
    concurrency_limit=10,
    # Allow longer timeouts for audio processing
    timeout=300,
    # Enable container reuse for better performance
    container_idle_timeout=240,
)
@modal.asgi_app()
def fastapi_app():
    """Create the ASGI application served by Modal.

    Returns:
        FastAPI: The Voice Changer application defined in ``voice_changer.py``.
    """
    # Import the existing FastAPI app from voice_changer.py
    from voice_changer import app as voice_changer_app

    return voice_changer_app


@app.local_entrypoint()
def main():
    """Provide a helpful message when running the Modal app locally."""
    print("🚀 Voice Changer Modal App")
    print("Use 'modal serve voice_changer_modal.py' to run locally")
    print("Use 'modal deploy voice_changer_modal.py' to deploy to production")


if __name__ == "__main__":
    main()
