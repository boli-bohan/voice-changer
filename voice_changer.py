#!/usr/bin/env python3
"""
Voice Changer Worker Service powered by WebRTC streaming.

This service accepts WebRTC peer connections, applies a pitch shift to
incoming audio in real time, and streams the transformed audio back to the
client. The HTTP API exposes an SDP offer endpoint that can be used by a
signalling service (the Go API) to negotiate connections.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Set

import librosa
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Changer WebRTC Worker", version="3.0.0")

# Allow local development hosts to negotiate directly with the worker.
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


class PitchShiftTrack(MediaStreamTrack):
    """Wraps an audio track and applies a pitch shift to each frame."""

    kind = "audio"

    def __init__(self, source: MediaStreamTrack, pitch_shift_semitones: float = 4.0):
        super().__init__()
        self._source = source
        self._pitch_shift = pitch_shift_semitones
        self._channels: int | None = None

    async def recv(self) -> AudioFrame:
        frame = await self._source.recv()
        samples = frame.to_ndarray()

        if samples.ndim == 1:
            mono = samples
            channels = 1
        else:
            # Average channels down to mono for processing.
            mono = samples.mean(axis=0)
            channels = samples.shape[0]

        if self._channels is None:
            self._channels = channels

        try:
            shifted = librosa.effects.pitch_shift(
                mono, sr=frame.sample_rate, n_steps=self._pitch_shift
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Pitch shifting error: %s", exc)
            shifted = mono

        shifted = np.clip(shifted, -1.0, 1.0)

        if self._channels and self._channels > 1:
            processed = np.tile(shifted, (self._channels, 1))
        else:
            processed = shifted[np.newaxis, :]

        pcm16 = np.clip(processed * 32767.0, -32768, 32767).astype(np.int16)

        out_frame = AudioFrame.from_ndarray(pcm16, format="s16")
        out_frame.sample_rate = frame.sample_rate
        out_frame.pts = frame.pts
        out_frame.time_base = frame.time_base
        return out_frame


active_peers: Set[RTCPeerConnection] = set()


async def _wait_for_ice_completion(pc: RTCPeerConnection) -> None:
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.05)


async def _cleanup_peer(pc: RTCPeerConnection) -> None:
    if pc in active_peers:
        active_peers.remove(pc)
    await pc.close()
    logger.info("🔌 Closed peer connection (%s)", pc)


@app.get("/")
async def get_status() -> dict[str, str]:
    return {"status": "running", "service": "voice_changer_webrtc_worker", "version": "3.0.0"}


@app.get("/health")
async def health_check() -> dict[str, object]:
    return {
        "status": "healthy",
        "active_peers": len(active_peers),
        "service": "voice_changer_webrtc_worker",
        "version": "3.0.0",
        "capabilities": ["pitch_shifting", "webrtc", "streaming_audio"],
    }


@app.post("/offer")
async def handle_offer(payload: SDPModel) -> dict[str, str]:
    """Accept an SDP offer and return the corresponding answer."""

    pc = RTCPeerConnection()
    active_peers.add(pc)
    logger.info("📡 Received new WebRTC offer; active peers: %d", len(active_peers))

    @pc.on("connectionstatechange")
    async def on_connection_state_change() -> None:  # pragma: no cover - relies on network state
        logger.info("Peer connection state: %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await _cleanup_peer(pc)

    track_ready = asyncio.Event()

    @pc.on("track")
    async def on_track(track: MediaStreamTrack) -> None:
        logger.info("🎙️ Incoming track: %s", track.kind)
        if track.kind == "audio":
            transformed = PitchShiftTrack(track)
            pc.addTrack(transformed)
            track_ready.set()

            @track.on("ended")
            async def on_track_ended() -> None:
                logger.info("🏁 Audio track ended")
                await _cleanup_peer(pc)

    offer = RTCSessionDescription(sdp=payload.sdp, type=payload.type)

    try:
        await pc.setRemoteDescription(offer)
        try:
            await asyncio.wait_for(track_ready.wait(), timeout=5.0)
        except asyncio.TimeoutError:
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


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await asyncio.gather(*(_cleanup_peer(pc) for pc in list(active_peers)), return_exceptions=True)


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting Voice Changer WebRTC Worker Service")
    uvicorn.run("voice_changer:app", host="127.0.0.1", port=8001, log_level="info")
