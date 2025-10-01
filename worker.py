"""Shared utilities for building WebRTC worker FastAPI apps."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable

import httpx
from aiortc import RTCPeerConnection
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


RouteRegistrar = Callable[[FastAPI, "WorkerContext"], None]


@dataclass(slots=True)
class WorkerSettings:
    """Configuration describing a worker's identity and runtime behaviour."""

    title: str
    version: str
    service_name: str
    capabilities: list[str]
    heartbeat_interval: float = 5.0
    max_connections: int = 4
    worker_port: int = int(os.environ.get("WORKER_PORT", "8001"))
    api_url: str = os.environ.get("API_URL", "http://voice-changer-api:8000")
    worker_host: str = os.environ.get("POD_IP", "127.0.0.1")
    worker_id: str = field(default_factory=lambda: os.environ.get("HOSTNAME", str(uuid.uuid4())))

    def __post_init__(self) -> None:
        self.worker_url = f"http://{self.worker_host}:{self.worker_port}"


class WorkerContext:
    """Holds shared state and helpers for individual workers."""

    def __init__(self, settings: WorkerSettings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.active_peers: set[RTCPeerConnection] = set()

    async def wait_for_ice_completion(self, pc: RTCPeerConnection) -> None:
        """Poll until ICE gathering completes for the given peer connection."""
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)

    async def cleanup_peer(self, pc: RTCPeerConnection) -> None:
        """Remove a peer from the active set and close its connection."""
        if pc in self.active_peers:
            self.active_peers.remove(pc)
        await pc.close()
        self.logger.info("🔌 Closed peer connection (%s)", pc)

    async def cleanup_all_peers(self) -> None:
        """Close all currently active peer connections."""
        if not self.active_peers:
            return
        await asyncio.gather(
            *(self.cleanup_peer(pc) for pc in list(self.active_peers)),
            return_exceptions=True,
        )

    async def heartbeat_loop(self) -> None:
        """Send periodic heartbeats so the API can track worker availability."""
        self.logger.info(
            "🫀 Starting heartbeat loop to %s (worker_id=%s, worker_url=%s)",
            self.settings.api_url,
            self.settings.worker_id,
            self.settings.worker_url,
        )

        async with httpx.AsyncClient() as client:
            while True:
                try:
                    await client.post(
                        f"{self.settings.api_url}/heartbeat",
                        json={
                            "worker_id": self.settings.worker_id,
                            "worker_url": self.settings.worker_url,
                            "connection_count": len(self.active_peers),
                            "max_connections": self.settings.max_connections,
                        },
                        timeout=5.0,
                    )
                    self.logger.debug(
                        "💓 Heartbeat sent: %d/%d connections",
                        len(self.active_peers),
                        self.settings.max_connections,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.warning("⚠️ Heartbeat failed: %s", exc)

                await asyncio.sleep(self.settings.heartbeat_interval)

    def status_payload(self) -> dict[str, str]:
        return {
            "status": "running",
            "service": self.settings.service_name,
            "version": self.settings.version,
        }

    def health_payload(self) -> dict[str, object]:
        return {
            "status": "healthy",
            "active_peers": len(self.active_peers),
            "service": self.settings.service_name,
            "version": self.settings.version,
            "capabilities": self.settings.capabilities,
        }

    def capacity_payload(self) -> dict[str, object]:
        return {
            "worker_id": self.settings.worker_id,
            "worker_url": self.settings.worker_url,
            "connection_count": len(self.active_peers),
            "max_connections": self.settings.max_connections,
            "available": len(self.active_peers) < self.settings.max_connections,
        }


def make_worker_app(
    *,
    settings: WorkerSettings,
    logger: logging.Logger,
    register_routes: RouteRegistrar,
    middleware_origins: Iterable[str] | None = None,
) -> FastAPI:
    """Create a FastAPI app wired with shared worker behaviour."""

    context = WorkerContext(settings=settings, logger=logger)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        heartbeat_task = asyncio.create_task(context.heartbeat_loop())
        logger.info("✅ Worker initialized: id=%s url=%s", settings.worker_id, settings.worker_url)
        try:
            yield
        finally:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
            await context.cleanup_all_peers()

    app = FastAPI(title=settings.title, version=settings.version, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(middleware_origins) if middleware_origins is not None else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def get_status() -> dict[str, str]:
        return context.status_payload()

    @app.get("/health")
    async def health_check() -> dict[str, object]:
        return context.health_payload()

    @app.get("/capacity")
    async def get_capacity() -> dict[str, object]:
        return context.capacity_payload()

    register_routes(app, context)

    return app


__all__ = [
    "WorkerSettings",
    "WorkerContext",
    "make_worker_app",
]
