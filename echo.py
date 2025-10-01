#!/usr/bin/env python3
"""Simple WebRTC worker that echoes audio frames without modification."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Reduce verbose heartbeat request logs from httpx while preserving warnings.
for noisy_logger in ("httpx", "httpcore"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Worker configuration for heartbeat and load balancing
WORKER_ID = os.environ.get("HOSTNAME", str(uuid.uuid4()))
WORKER_URL = f"http://{os.environ.get('POD_IP', '127.0.0.1')}:8001"
API_URL = os.environ.get("API_URL", "http://voice-changer-api:8000")
HEARTBEAT_INTERVAL = 5  # seconds
MAX_CONNECTIONS = 4


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize worker and start background tasks on application startup.

    Args:
        app: FastAPI application instance managed by the lifespan context.
    """
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

    sdp: str = Field(description="Session description protocol body provided by the browser.")
    type: str = Field(description="SDP type such as 'offer' or 'answer'.")


class EchoTrack(MediaStreamTrack):
    """Media track that simply forwards incoming audio frames."""

    kind = "audio"

    def __init__(self, source: MediaStreamTrack):
        """Initialize an echo track that proxies frames from the source.

        Args:
            source: Upstream media track whose audio should be forwarded.
        """
        super().__init__()
        self._source = source

    async def recv(self) -> AudioFrame:  # type: ignore[override]
        """Return the next frame received from the source without modification.

        Returns:
            Audio frame sourced directly from the upstream track.
        """

        return await self._source.recv()


active_peers: set[RTCPeerConnection] = set()


async def _wait_for_ice_completion(pc: RTCPeerConnection) -> None:
    """Wait until ICE gathering completes for a peer connection.

    Args:
        pc: Connection to monitor for ICE completion.
    """
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.05)


async def _cleanup_peer(pc: RTCPeerConnection) -> None:
    """Remove the peer from the active set and close the connection.

    Args:
        pc: Connection that should be closed and deregistered.
    """
    if pc in active_peers:
        active_peers.remove(pc)
    await pc.close()
    logger.info("🔌 Closed peer connection (%s)", pc)


async def heartbeat_loop() -> None:
    """Send periodic heartbeats so the API can track worker availability."""
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
    """Return readiness metadata for health probes.

    Returns:
        Service identifier and status metadata.
    """

    return {"status": "running", "service": "voice_changer_echo_worker", "version": "1.0.0"}


@app.get("/health")
async def health_check() -> dict[str, object]:
    """Report the current health state and connection metrics.

    Returns:
        Details about service status and active peers.
    """
    return {
        "status": "healthy",
        "active_peers": len(active_peers),
        "service": "voice_changer_echo_worker",
        "version": "1.0.0",
        "capabilities": ["echo", "webrtc", "streaming_audio"],
    }


@app.get("/capacity")
async def get_capacity() -> dict[str, object]:
    """Return current worker capacity and availability information.

    Returns:
        Worker identification, load, and availability status.
    """
    return {
        "worker_id": WORKER_ID,
        "worker_url": WORKER_URL,
        "connection_count": len(active_peers),
        "max_connections": MAX_CONNECTIONS,
        "available": len(active_peers) < MAX_CONNECTIONS,
    }


@app.post("/offer")
async def handle_offer(payload: SDPModel) -> dict[str, str]:
    """Accept an SDP offer and return the corresponding answer.

    Args:
        payload: The SDP model supplied by the signalling API.

    Returns:
        The generated SDP answer describing the worker session.
    """
    pc = RTCPeerConnection()
    active_peers.add(pc)
    logger.info("📡 Received new WebRTC offer; active peers: %d", len(active_peers))

    @pc.on("connectionstatechange")
    async def on_connection_state_change() -> None:  # pragma: no cover - depends on network
        """Monitor peer state changes and trigger cleanup when disconnected."""
        logger.info("Peer connection state: %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await _cleanup_peer(pc)

    track_ready = asyncio.Event()

    @pc.on("track")
    async def on_track(track: MediaStreamTrack) -> None:
        """Handle incoming audio tracks by echoing frames back to the caller.

        Args:
            track: Remote track received from the signalling peer.
        """
        logger.info("🎙️ Incoming track: %s", track.kind)
        if track.kind == "audio":
            echoed = EchoTrack(track)
            pc.addTrack(echoed)
            track_ready.set()

            @track.on("ended")
            async def on_track_ended() -> None:
                """Log track completion and dispose of the peer connection."""
                logger.info("🏁 Audio track ended")
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
