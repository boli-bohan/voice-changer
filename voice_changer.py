#!/usr/bin/env python3
"""
Voice Changer PCM Worker Service - Raw PCM Audio Processing
Simplified FastAPI WebSocket server for real-time pitch shifting on raw PCM data.
"""

import json
import logging
import librosa
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Changer PCM Worker", version="2.0.0")


class PCMAudioProcessor:
    """Real-time streaming audio processor for raw PCM data."""

    def __init__(self, pitch_shift_semitones: float = 4.0, sample_rate: int = 44100):
        self.pitch_shift_semitones = pitch_shift_semitones
        self.sample_rate = sample_rate

        # Buffer for accumulating PCM samples
        self.pcm_buffer: list[float] = []
        self.min_chunk_size = int(sample_rate * 0.1)  # 100ms worth of samples

    def add_pcm_samples(self, pcm_data: bytes) -> np.ndarray | None:
        """
        Add raw PCM samples and return processed audio if enough data is available.

        Args:
            pcm_data: Raw PCM data as bytes (32-bit float little endian)

        Returns:
            Processed audio samples as numpy array, or None if not enough data
        """
        float_samples = np.frombuffer(pcm_data, dtype=np.float32)
        self.pcm_buffer.extend(float_samples.tolist())

        # Process if we have enough samples
        if len(self.pcm_buffer) >= self.min_chunk_size:
            # Extract chunk for processing
            chunk = np.array(self.pcm_buffer[: self.min_chunk_size], dtype=np.float32)
            self.pcm_buffer = self.pcm_buffer[self.min_chunk_size :]

            # Apply pitch shifting
            try:
                processed_chunk = librosa.effects.pitch_shift(
                    chunk, sr=self.sample_rate, n_steps=self.pitch_shift_semitones
                )
                return processed_chunk
            except Exception as e:
                logger.error(f"Pitch shifting error: {e}")
                return chunk  # Return original on error

        return None

    def flush_remaining(self) -> np.ndarray | None:
        """Process any remaining samples in the buffer."""
        if len(self.pcm_buffer) > 0:
            chunk = np.array(self.pcm_buffer, dtype=np.float32)
            self.pcm_buffer.clear()

            try:
                processed_chunk = librosa.effects.pitch_shift(
                    chunk, sr=self.sample_rate, n_steps=self.pitch_shift_semitones
                )
                return processed_chunk
            except Exception as e:
                logger.error(f"Final pitch shifting error: {e}")
                return chunk

        return None

    def numpy_to_pcm_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy audio data back to PCM float32 bytes."""
        audio_data = np.clip(audio_data, -1.0, 1.0).astype(np.float32)
        return audio_data.tobytes()


@app.get("/")
async def get_status():
    """Health check endpoint."""
    return {"status": "Voice Changer PCM Worker is running", "version": "2.0.0"}


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "service": "voice_changer_pcm_worker",
        "version": "2.0.0",
        "capabilities": ["pitch_shifting", "real_time_pcm_processing", "raw_audio"],
    }


@app.websocket("/process")
async def websocket_pcm_processor(websocket: WebSocket):
    """WebSocket endpoint for real-time PCM voice processing."""
    await websocket.accept()

    # Initialize processor for this connection
    processor = PCMAudioProcessor(pitch_shift_semitones=4.0)

    logger.info("🔊 Voice changer PCM worker: New processing connection established")

    try:
        while True:
            # Receive data from Go API server
            data = await websocket.receive()

            if "text" in data:
                # Handle control messages
                try:
                    message = json.loads(data["text"])
                    message_type = message.get("type")

                    if message_type == "flush":
                        logger.info("🔄 Received flush command, processing remaining audio")

                        # Process any remaining audio
                        final_chunk = processor.flush_remaining()
                        if final_chunk is not None:
                            pcm_bytes = processor.numpy_to_pcm_bytes(final_chunk)
                            await websocket.send_bytes(pcm_bytes)

                        # Send done message
                        await websocket.send_text(
                            json.dumps({"type": "done", "message": "PCM processing completed"})
                        )
                        logger.info("✅ Sent done message, PCM processing completed")
                        # Break out of the loop after sending done message
                        break

                    elif message_type == "config":
                        # Update processing configuration
                        pitch_shift = message.get("pitch_shift", 4.0)
                        processor.pitch_shift_semitones = pitch_shift
                        logger.info(f"🎛️ Updated pitch shift to {pitch_shift} semitones")

                        await websocket.send_text(
                            json.dumps({"type": "config_updated", "pitch_shift": pitch_shift})
                        )

                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")

            elif "bytes" in data:
                # Handle raw PCM data
                pcm_data = data["bytes"]
                logger.info("🎵 Received PCM chunk: %d bytes", len(pcm_data))

                # Process the PCM data
                processed_chunk = processor.add_pcm_samples(pcm_data)

                if processed_chunk is not None:
                    # Convert back to PCM bytes and send
                    pcm_bytes = processor.numpy_to_pcm_bytes(processed_chunk)
                    await websocket.send_bytes(pcm_bytes)
                    logger.info("📤 Sent processed PCM chunk: %d bytes", len(pcm_bytes))

    except WebSocketDisconnect:
        logger.info("🔌 Voice changer PCM worker: Connection disconnected")
    except Exception as e:
        logger.error(f"❌ Voice changer PCM worker error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # For development/testing
    logger.info("🚀 Starting Voice Changer PCM Worker Service")
    uvicorn.run("voice_changer_pcm:app", host="127.0.0.1", port=8001, log_level="info", reload=True)
