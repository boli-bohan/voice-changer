#!/usr/bin/env python3
"""
Voice Changer PCM Worker Service - Raw PCM Audio Processing
Simplified FastAPI WebSocket server for real-time pitch shifting on raw PCM data.
"""

import asyncio
import json
import logging
import struct

from fractions import Fraction
from typing import Any

import librosa
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from av import AudioFrame

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Changer PCM Worker", version="2.0.0")


class PCMStreamTrack(AudioStreamTrack):
    """Audio track that pushes PCM frames to a WebRTC peer connection."""

    kind = "audio"

    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._pts: int = 0

    async def recv(self) -> AudioFrame:
        pcm_bytes = await self._queue.get()
        sample_count = len(pcm_bytes) // 2
        frame = AudioFrame(format="s16", layout="mono", samples=sample_count)
        frame.planes[0].update(pcm_bytes)
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._pts
        self._pts += sample_count
        return frame

    async def queue_pcm(self, pcm_bytes: bytes) -> None:
        await self._queue.put(pcm_bytes)

    def reset(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._pts = 0


class PCMAudioProcessor:
    """Real-time streaming audio processor for raw PCM data."""

    def __init__(self, pitch_shift_semitones: float = 4.0, sample_rate: int = 48000):
        self.pitch_shift_semitones = pitch_shift_semitones
        self.sample_rate = sample_rate

        # Buffer for accumulating PCM samples
        self.pcm_buffer: list[float] = []
        self.min_chunk_size = int(sample_rate * 0.1)  # 100ms worth of samples

    def add_pcm_samples(self, pcm_data: bytes) -> np.ndarray | None:
        """
        Add raw PCM samples and return processed audio if enough data is available.

        Args:
            pcm_data: Raw PCM data as bytes (16-bit signed little endian)

        Returns:
            Processed audio samples as numpy array, or None if not enough data
        """
        # Convert bytes to int16 samples
        sample_count = len(pcm_data) // 2
        samples = struct.unpack(f"<{sample_count}h", pcm_data)

        # Convert to float32 and normalize
        float_samples = [s / 32768.0 for s in samples]
        self.pcm_buffer.extend(float_samples)

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
        """Convert numpy audio data back to PCM bytes."""
        # Ensure we're in the right range
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # Convert to int16
        int16_data = (audio_data * 32767).astype(np.int16)

        # Pack to bytes
        return struct.pack(f"<{len(int16_data)}h", *int16_data)


class WorkerSession:
    """Session state for a single worker WebSocket connection."""

    def __init__(
        self, websocket: WebSocket, pitch_shift: float = 4.0, sample_rate: int = 48000
    ) -> None:
        self.websocket = websocket
        self.processor = PCMAudioProcessor(
            pitch_shift_semitones=pitch_shift, sample_rate=sample_rate
        )
        self.pitch_shift = pitch_shift
        self.sample_rate = sample_rate
        self.pc: RTCPeerConnection | None = None
        self.track: PCMStreamTrack | None = None
        self.streaming_started = False

    async def ensure_peer_connection(self) -> None:
        if self.pc is not None:
            return

        self.pc = RTCPeerConnection()
        self.track = PCMStreamTrack(self.sample_rate)
        self.pc.addTrack(self.track)

        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate: Any) -> None:
            if candidate is None:
                return
            payload = {
                "type": "webrtc_ice_candidate",
                "candidate": {
                    "candidate": candidate.candidate,
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                },
            }
            await self.websocket.send_text(json.dumps(payload))

        @self.pc.on("connectionstatechange")
        async def on_state_change() -> None:
            logger.info("WebRTC connection state changed: %s", self.pc.connectionState)  # type: ignore[arg-type]

    async def handle_offer(self, sdp: str, sdp_type: str | None) -> None:
        if self.pc is not None:
            await self.close()

        await self.ensure_peer_connection()

        if sdp_type is None or sdp_type == "":
            sdp_type = "offer"

        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        assert self.pc is not None
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        response = {
            "type": "webrtc_answer",
            "sdp": self.pc.localDescription.sdp if self.pc.localDescription else "",
            "sdp_type": self.pc.localDescription.type if self.pc.localDescription else "answer",
        }
        await self.websocket.send_text(json.dumps(response))

    async def add_ice_candidate(self, payload: dict[str, Any]) -> None:
        if self.pc is None:
            return

        candidate = payload.get("candidate")
        if not candidate:
            return

        rtc_candidate = candidate
        await self.pc.addIceCandidate(rtc_candidate)

    async def queue_processed_audio(self, pcm_bytes: bytes) -> None:
        if not self.streaming_started:
            await self.websocket.send_text(
                json.dumps({"type": "streaming_started", "message": "Starting audio playback"})
            )
            self.streaming_started = True

        if self.track is not None:
            await self.track.queue_pcm(pcm_bytes)
        else:
            await self.websocket.send_bytes(pcm_bytes)

    async def flush(self) -> None:
        if self.streaming_started:
            await self.websocket.send_text(
                json.dumps({"type": "streaming_completed", "message": "Audio streaming completed"})
            )
            self.streaming_started = False

    async def close(self) -> None:
        if self.pc is not None:
            await self.pc.close()
        self.pc = None
        if self.track is not None:
            self.track.reset()
            self.track.stop()
        self.track = None
        self.streaming_started = False


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

    session = WorkerSession(websocket)

    logger.info("🔊 Voice changer PCM worker: New processing connection established")

    try:
        while True:
            data = await websocket.receive()

            if "text" in data:
                try:
                    message = json.loads(data["text"])
                    message_type = message.get("type")

                    if message_type == "flush":
                        logger.info("🔄 Received flush command, processing remaining audio")

                        final_chunk = session.processor.flush_remaining()
                        if final_chunk is not None:
                            pcm_bytes = session.processor.numpy_to_pcm_bytes(final_chunk)
                            await session.queue_processed_audio(pcm_bytes)

                        await session.flush()
                        await websocket.send_text(
                            json.dumps({"type": "done", "message": "PCM processing completed"})
                        )
                        logger.info("✅ Sent done message, PCM processing completed")
                        break

                    elif message_type == "config":
                        pitch_shift = message.get("pitch_shift", 4.0)
                        session.processor.pitch_shift_semitones = pitch_shift
                        session.pitch_shift = pitch_shift
                        logger.info("🎛️ Updated pitch shift to %s semitones", pitch_shift)

                        await websocket.send_text(
                            json.dumps({"type": "config_updated", "pitch_shift": pitch_shift})
                        )

                    elif message_type == "webrtc_offer":
                        sdp = message.get("sdp", "")
                        sdp_type = message.get("sdp_type")
                        if not sdp:
                            logger.warning("Received empty WebRTC offer")
                            continue
                        await session.handle_offer(sdp, sdp_type)

                    elif message_type == "webrtc_ice_candidate":
                        await session.add_ice_candidate(message)

                    else:
                        logger.warning("⚠️ Unknown control message type received: %s", message_type)

                except json.JSONDecodeError:
                    logger.error("❌ Failed to parse control message as JSON")

            elif "bytes" in data:
                pcm_data = data["bytes"]
                logger.info("🎵 Received PCM chunk: %d bytes", len(pcm_data))

                processed_chunk = session.processor.add_pcm_samples(pcm_data)

                if processed_chunk is not None:
                    processed_bytes = session.processor.numpy_to_pcm_bytes(processed_chunk)
                    await session.queue_processed_audio(processed_bytes)
                    logger.info("📤 Queued processed PCM chunk: %d bytes", len(processed_bytes))

    except WebSocketDisconnect:
        logger.info("🔌 Voice changer PCM worker: Connection disconnected")
    except Exception as e:
        logger.error("❌ Voice changer PCM worker error: %s", e)
    finally:
        await session.flush()
        await session.close()


if __name__ == "__main__":
    import uvicorn

    # For development/testing
    logger.info("🚀 Starting Voice Changer PCM Worker Service")
    uvicorn.run("voice_changer_pcm:app", host="127.0.0.1", port=8001, log_level="info", reload=True)
