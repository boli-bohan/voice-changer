"""WebRTC end-to-end test client for the voice changer stack."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib import request

if "pytest" in sys.modules:  # pragma: no cover - avoids pytest collecting this CLI utility
    import pytest

    pytest.skip("test_client.py is an integration utility, not a pytest module", allow_module_level=True)

import wave

import numpy as np
import typer
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from aiortc.mediastreams import MediaStreamError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _post_json(url: str, payload: dict[str, object], timeout: float = 15.0) -> dict[str, object]:
    """Send a JSON payload via HTTP POST and return the decoded response.

    Args:
        url (str): Destination URL for the request.
        payload (dict[str, object]): JSON-serialisable body to send.
        timeout (float): Timeout in seconds for the HTTP request.

    Returns:
        dict[str, object]: Parsed JSON response.

    Raises:
        RuntimeError: If the server responds with an error status code.
    """
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}: {body}")
        return json.loads(body)


def _ensure_parent(path: Path) -> None:
    """Create the parent directories for ``path`` if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class TestConfig:
    """Runtime configuration for the WebRTC test harness."""

    api_base: str = "http://localhost:8000"
    offer_path: str = "/webrtc/offer"
    wait_after_audio: float = 1.0


class VoiceChangerWebRTCTestClient:
    """WebRTC client used for exercising the Voice Changer worker."""

    def __init__(self, config: TestConfig):
        """Store configuration and initialise placeholders for WebRTC state."""
        self.config = config
        self.pc: RTCPeerConnection | None = None
        self.player: MediaPlayer | None = None
        self.record_task: asyncio.Task[int] | None = None

    async def _wait_for_ice(self) -> None:
        """Wait until ICE gathering for the active peer connection completes."""
        assert self.pc is not None
        while self.pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)

    async def _negotiate(self) -> None:
        """Negotiate the WebRTC session by exchanging SDP with the signalling API."""
        assert self.pc is not None
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        await self._wait_for_ice()

        assert self.pc.localDescription is not None
        url = self.config.api_base.rstrip("/") + self.config.offer_path
        response = await asyncio.to_thread(
            _post_json,
            url,
            {"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type},
        )

        answer = RTCSessionDescription(sdp=response["sdp"], type=response["type"])
        await self.pc.setRemoteDescription(answer)

    async def _record_remote_audio(self, track: MediaStreamTrack, output_path: Path) -> int:
        """Collect audio frames from the remote track and persist them to disk.

        Returns:
            int: The number of PCM samples written to disk. ``0`` indicates that no audio
            was captured, either because the track never produced data or an error
            occurred while recording.
        """

        chunks: list[np.ndarray] = []
        sample_rate: int | None = None
        channels: int | None = None

        try:
            while True:
                frame = await track.recv()
                sample_rate = frame.sample_rate or sample_rate
                data = frame.to_ndarray()
                if data.ndim == 1:
                    data = data[np.newaxis, :]
                channels = data.shape[0]
                chunks.append(np.copy(data))
        except MediaStreamError:
            logger.debug("Remote track ended; finalising recording")
        except Exception as exc:
            logger.error("Failed to record remote audio: %s", exc)
            return 0

        if not chunks:
            logger.error("❌ No audio samples received; nothing written to %s", output_path)
            return 0

        samples = np.concatenate(chunks, axis=1)
        pcm = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype(np.int16)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(channels or 1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate or 48000)
            wf.writeframes(pcm16.T.tobytes())

        sample_count = int(pcm16.shape[1])
        logger.info("💾 Wrote %s samples to %s", sample_count, output_path)
        return sample_count

    async def process_audio_file(self, input_file: str, output_file: str, expected_file: str | None = None) -> bool:
        """Stream an input file through the worker and optionally verify the output.

        Args:
            input_file (str): Path to the source audio file to send.
            output_file (str): Path for recording the transformed audio.
            expected_file (str | None): Optional reference file for validation.

        Returns:
            bool: ``True`` when processing (and verification, if requested) succeeds.
        """
        logger.info("🎧 Starting WebRTC test session")
        input_path = Path(input_file)
        output_path = Path(output_file)
        _ensure_parent(output_path)
        logger.debug("Recording output to %s", output_path.resolve())

        self.pc = RTCPeerConnection()
        self.player = MediaPlayer(str(input_path))

        if not self.player.audio:
            raise RuntimeError("Input file does not contain an audio track")

        @self.pc.on("track")
        async def on_track(track):
            if track.kind == "audio":
                logger.info("🔁 Remote audio track received")
                if self.record_task is None:
                    self.record_task = asyncio.create_task(
                        self._record_remote_audio(track, output_path)
                    )

        self.pc.addTrack(self.player.audio)

        await self._negotiate()

        while self.player.audio.readyState != "ended":
            await asyncio.sleep(0.1)

        await asyncio.sleep(self.config.wait_after_audio)

        if self.player and self.player.audio:
            # MediaPlayer in aiortc doesn't expose an async stop(), so stop the audio track directly.
            self.player.audio.stop()
        if self.pc:
            await self.pc.close()
        recorded_samples = 0
        if self.record_task:
            recorded_samples = await self.record_task
            self.record_task = None

        if recorded_samples > 0:
            logger.info("✅ Audio captured to %s", output_path)
        else:
            logger.error("❌ Audio capture failed for %s", output_path)

        if expected_file:
            return self.verify_output(output_path, Path(expected_file))

        exists = output_path.exists() and recorded_samples > 0
        if exists:
            logger.info("📦 Output file available (%s bytes)", os.path.getsize(output_path))
        else:
            logger.error("❌ Output file was not created")
        return exists

    @staticmethod
    def verify_output(actual_file: Path, expected_file: Path) -> bool:
        """Compare the generated audio file to an expected reference file.

        Args:
            actual_file (Path): Path to the recorded output file.
            expected_file (Path): Path to the expected reference audio file.

        Returns:
            bool: ``True`` if the file sizes are within tolerance, otherwise ``False``.
        """
        if not expected_file.exists():
            logger.warning("⚠️ Expected file not found: %s", expected_file)
            return False

        if not actual_file.exists():
            logger.error("❌ Output file missing: %s", actual_file)
            return False

        actual_size = os.path.getsize(actual_file)
        expected_size = os.path.getsize(expected_file)
        diff_pct = abs(actual_size - expected_size) / max(expected_size, 1) * 100

        logger.info(
            "🔍 Output comparison: actual=%s bytes expected=%s bytes diff=%.1f%%",
            actual_size,
            expected_size,
            diff_pct,
        )

        success = diff_pct <= 15
        if success:
            logger.info("🎉 Verification passed")
        else:
            logger.error("💥 Verification failed")
        return success


async def run_test(input_file: str, output_file: str, expected_file: str | None, api_base: str) -> bool:
    """Execute the end-to-end test workflow with the provided parameters.

    Args:
        input_file (str): Audio file to send through the worker.
        output_file (str): File path where the transformed audio should be saved.
        expected_file (str | None): Optional file path for verification.
        api_base (str): Base URL of the signalling API.

    Returns:
        bool: ``True`` if processing and optional verification succeed.
    """
    client = VoiceChangerWebRTCTestClient(TestConfig(api_base=api_base))
    return await client.process_audio_file(input_file, output_file, expected_file)


app = typer.Typer(help="Voice Changer WebRTC Test Client")


@app.command()
def main_sync(
    input_file: Path = typer.Argument(Path("data/test_input.webm"), help="Input audio file"),
    output_file: Path = typer.Argument(Path("output.wav"), help="Output audio file"),
    expected_file: Path | None = typer.Option(None, "--expected", "-e", help="Expected output for verification"),
    api_base: str = typer.Option("http://localhost:8000", "--api-base", "-a", help="API base URL"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """CLI entry point for running an end-to-end worker verification test."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    typer.echo("\n🧪 Starting WebRTC end-to-end test")
    typer.echo(f"   📥 Input: {input_file}")
    typer.echo(f"   💾 Output: {output_file}")
    typer.echo(f"   🌐 API: {api_base}")
    if expected_file:
        typer.echo(f"   ✅ Expected: {expected_file}")

    success = asyncio.run(run_test(str(input_file), str(output_file), str(expected_file) if expected_file else None, api_base))

    if success:
        typer.echo("\n✅ Test completed successfully")
        typer.echo(f"📁 Output saved to {output_file}")
    else:
        typer.echo("\n❌ Test failed", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
