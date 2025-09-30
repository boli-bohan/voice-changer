#!/usr/bin/env python3
"""Simple WebRTC worker that echoes audio frames without modification."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Worker configuration for heartbeat and load balancing
WORKER_ID = os.environ.get("HOSTNAME", str(uuid.uuid4()))
WORKER_URL = f"http://{os.environ.get('POD_IP', '127.0.0.1')}:8001"
API_URL = os.environ.get("API_URL", "http://voice-changer-api:8000")
HEARTBEAT_INTERVAL = 5  # seconds
MAX_CONNECTIONS = 4


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize worker and start background tasks on application startup."""
    asyncio.create_task(heartbeat_loop())
    logger.info("✅ Echo worker initialized: id=%s url=%s", WORKER_ID, WORKER_URL)
    yield
    await asyncio.gather(*(_cleanup_peer(pc) for pc in list(active_peers)), return_exceptions=True)


app = FastAPI(title="Echo WebRTC Worker", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SDPModel(BaseModel):
    """Pydantic model representing a WebRTC SDP payload."""

    sdp: str
    type: str


class EchoTrack(MediaStreamTrack):
    """Media track that simply forwards incoming audio frames."""

    kind = "audio"

    def __init__(self, source: MediaStreamTrack):
        super().__init__()
        self._source = source
        self._debug_dir = Path(f"/tmp/audio/{WORKER_ID}")
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._input_wav_path = self._debug_dir / "input.wav"
        self._output_wav_path = self._debug_dir / "output.wav"
        self._input_wav = wave.open(str(self._input_wav_path), "wb")
        self._input_wav.setnchannels(1)
        self._input_wav.setsampwidth(2)
        self._input_wav.setframerate(48_000)
        self._output_wav = wave.open(str(self._output_wav_path), "wb")
        self._output_wav.setnchannels(1)
        self._output_wav.setsampwidth(2)
        self._output_wav.setframerate(48_000)
        logger.info("🎙️ Echo worker debug audio streaming enabled: %s", self._debug_dir)

    async def recv(self) -> AudioFrame:  # type: ignore[override]
        frame = await self._source.recv()

        samples = frame.to_ndarray()
        if samples.ndim == 1:
            mono = samples.astype(np.float32, copy=False)
        else:
            mono = samples[0].astype(np.float32, copy=False)

        if np.issubdtype(samples.dtype, np.integer):
            info = np.iinfo(samples.dtype)
            denom = float(max(abs(info.min), info.max)) or 1.0
            mono = mono / denom
        elif np.issubdtype(samples.dtype, np.floating):
            pass
        else:  # pragma: no cover - defensive fallback
            peak = float(np.max(np.abs(mono))) or 1.0
            mono = mono / peak

        mono = np.clip(mono, -1.0, 1.0)
        pcm16 = (mono * 32767.0).astype(np.int16)
        if pcm16.size:
            bytes_ = pcm16.tobytes()
            self._input_wav.writeframes(bytes_)
            self._output_wav.writeframes(bytes_)

        return frame

    def close_debug_files(self) -> None:
        try:
            self._input_wav.close()
            self._output_wav.close()
            logger.info(
                "💾 Echo debug audio saved: input=%s output=%s",
                self._input_wav_path,
                self._output_wav_path,
            )
        except Exception as exc:
            logger.warning("⚠️ Error closing echo debug WAV files: %s", exc)


active_peers: set[RTCPeerConnection] = set()


async def _wait_for_ice_completion(pc: RTCPeerConnection) -> None:
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.05)


async def _cleanup_peer(pc: RTCPeerConnection) -> None:
    if pc in active_peers:
        active_peers.remove(pc)
    await pc.close()
    logger.info("🔌 Closed peer connection (%s)", pc)


async def heartbeat_loop() -> None:
    logger.info(
        "🫀 Starting heartbeat loop to %s (worker_id=%s, worker_url=%s)",
        API_URL,
        WORKER_ID,
        WORKER_URL,
    )

    async with httpx.AsyncClient() as client:
        while True:
            try:
                await client.post(
                    f"{API_URL}/heartbeat",
                    json={
                        "worker_id": WORKER_ID,
                        "worker_url": WORKER_URL,
                        "connection_count": len(active_peers),
                        "max_connections": MAX_CONNECTIONS,
                    },
                    timeout=5.0,
                )
                logger.debug(
                    "💓 Heartbeat sent: %d/%d connections",
                    len(active_peers),
                    MAX_CONNECTIONS,
                )
            except Exception as exc:
                logger.warning("⚠️ Heartbeat failed: %s", exc)

            await asyncio.sleep(HEARTBEAT_INTERVAL)


@app.get("/")
async def get_status() -> dict[str, str]:
    return {"status": "running", "service": "voice_changer_echo_worker", "version": "1.0.0"}


@app.get("/health")
async def health_check() -> dict[str, object]:
    return {
        "status": "healthy",
        "active_peers": len(active_peers),
        "service": "voice_changer_echo_worker",
        "version": "1.0.0",
        "capabilities": ["echo", "webrtc", "streaming_audio"],
    }


@app.get("/capacity")
async def get_capacity() -> dict[str, object]:
    return {
        "worker_id": WORKER_ID,
        "worker_url": WORKER_URL,
        "connection_count": len(active_peers),
        "max_connections": MAX_CONNECTIONS,
        "available": len(active_peers) < MAX_CONNECTIONS,
    }


@app.post("/offer")
async def handle_offer(payload: SDPModel) -> dict[str, str]:
    pc = RTCPeerConnection()
    active_peers.add(pc)
    logger.info("📡 Received new WebRTC offer; active peers: %d", len(active_peers))

    @pc.on("connectionstatechange")
    async def on_connection_state_change() -> None:  # pragma: no cover - depends on network
        logger.info("Peer connection state: %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await _cleanup_peer(pc)

    track_ready = asyncio.Event()

    @pc.on("track")
    async def on_track(track: MediaStreamTrack) -> None:
        logger.info("🎙️ Incoming track: %s", track.kind)
        if track.kind == "audio":
            echoed = EchoTrack(track)
            pc.addTrack(echoed)
            track_ready.set()

            @track.on("ended")
            async def on_track_ended() -> None:
                logger.info("🏁 Audio track ended")
                echoed.close_debug_files()
                await _cleanup_peer(pc)

    offer = RTCSessionDescription(sdp=payload.sdp, type=payload.type)

    try:
        await pc.setRemoteDescription(offer)
        try:
            await asyncio.wait_for(track_ready.wait(), timeout=5.0)
        except TimeoutError:
            logger.warning("⚠️ No audio track received before answer generation")
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_for_ice_completion(pc)
    except Exception as exc:  # pragma: no cover - network failure handling
        await _cleanup_peer(pc)
        raise HTTPException(status_code=500, detail=f"Failed to process SDP offer: {exc}")

    logger.info("✅ Generated SDP answer")
    assert pc.localDescription is not None
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting Echo WebRTC Worker Service")
    uvicorn.run("echo:app", host="127.0.0.1", port=8001, log_level="info")
