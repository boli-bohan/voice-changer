#!/usr/bin/env python3
"""Minimal WebRTC regression harness that streams audio with MediaPlayer."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from urllib import request

import numpy as np
import typer
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from aiortc.contrib.media import MediaPlayer
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

try:
    from av import AudioFrame
except ImportError as exc:  # pragma: no cover - surface clearer failure
    raise RuntimeError("PyAV is required for the WebRTC client") from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _post_json(url: str, payload: dict[str, object], timeout: float = 15.0) -> dict[str, object]:
    """Send a JSON payload to the worker API and parse the response.

    Args:
        url: Endpoint to which the JSON payload should be posted.
        payload: Body to serialise and send in the HTTP request.
        timeout: Maximum seconds to wait for the API response.

    Returns:
        Parsed JSON body returned by the worker API.
    """
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}: {body}")
        return json.loads(body)


def _ensure_parent(path: Path) -> None:
    """Ensure the parent directory exists for the provided path.

    Args:
        path: Target file path whose parent directory should be created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _validate_file_size(actual_file: Path, expected_file: Path, tolerance_pct: float = 15.0) -> bool:
    """Compare output audio file size against an expected reference.

    Args:
        actual_file: Newly generated audio file to validate.
        expected_file: Reference file representing the desired size envelope.
        tolerance_pct: Percentage difference tolerated between file sizes.

    Returns:
        ``True`` if the size difference is within the tolerance threshold.
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
        "🔍 File size comparison: actual=%s bytes expected=%s bytes diff=%.1f%%",
        actual_size,
        expected_size,
        diff_pct,
    )

    if diff_pct > tolerance_pct:
        logger.error("💥 File size difference exceeds %.1f%% threshold", tolerance_pct)
        return False

    return True


@dataclass(slots=True)
class ClientConfig:
    """Configuration governing how the client negotiates WebRTC sessions."""

    api_base: str = "http://localhost:8000"
    offer_path: str = "/webrtc/offer"
    wait_after_input: float = 1.0


async def negotiate(pc: RTCPeerConnection, config: ClientConfig) -> None:
    """Perform SDP offer/answer negotiation with the worker API.

    Args:
        pc: Peer connection that will generate and apply SDP descriptions.
        config: Connection endpoints and timing configuration.
    """
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.05)

    assert pc.localDescription is not None
    url = config.api_base.rstrip("/") + config.offer_path
    response = await asyncio.to_thread(
        _post_json,
        url,
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
    )

    answer = RTCSessionDescription(sdp=response["sdp"], type=response["type"])
    await pc.setRemoteDescription(answer)


def _ensure_interleaved(frame: AudioFrame, channels: int) -> np.ndarray:
    """Convert frame data into an interleaved numpy array.

    Args:
        frame: Audio frame produced by the remote track.
        channels: Expected channel count for the audio frame.

    Returns:
        Interleaved PCM samples ready for writing to disk.
    """
    array = frame.to_ndarray()
    if array.ndim == 1:
        if channels <= 0:
            raise ValueError("Audio frame without channels cannot be flattened")
        array = array.reshape(-1, channels)
    elif array.ndim == 2:
        if array.shape[0] == channels:
            array = array.transpose(1, 0)
        elif array.shape[1] != channels:
            total = array.size
            if channels <= 0 or total % channels != 0:
                raise ValueError(f"Unexpected audio frame shape {array.shape} for {channels} channels")
            array = array.reshape(-1, channels)
    else:
        raise ValueError(f"Unsupported audio frame dimensions: {array.shape}")
    return np.ascontiguousarray(array)


def _drain_async(awaitable: Awaitable[object]) -> None:
    """Ensure an awaitable is executed even when called from sync context.

    Args:
        awaitable: Coroutine or awaitable returned by a cleanup routine.
    """
    async def _consume() -> None:
        try:
            await awaitable
        except Exception:  # pragma: no cover - best effort cleanup
            logger.exception("Exception while awaiting async cleanup")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_consume())
    else:
        loop.create_task(_consume())


class DurationLoggingTrack(MediaStreamTrack):
    """Proxy track that logs cumulative duration of outgoing audio."""

    kind = "audio"

    def __init__(self, source: MediaStreamTrack, label: str):
        """Wrap a media track and log cumulative audio duration.

        Args:
            source: Underlying media track to proxy.
            label: Human-readable identifier for log messages.
        """
        super().__init__()
        self._source = source
        self._label = label
        self._total_samples = 0
        self._last_sample_rate = 0
        self._frame_count = 0
        self._log_interval_sec = 0.5
        self._log_interval_samples = 0
        self._next_log_sample = 0

    async def recv(self) -> AudioFrame:  # type: ignore[override]
        """Receive and return the next frame while updating duration metrics.

        Returns:
            Audio frame fetched from the underlying source.
        """
        frame = await self._source.recv()
        assert isinstance(frame, AudioFrame)

        samples = getattr(frame, "samples", None)
        if samples is None:
            array = frame.to_ndarray()
            samples = array.shape[-1] if array.size else 0

        self._frame_count += 1
        self._total_samples += samples
        if frame.sample_rate:
            self._last_sample_rate = frame.sample_rate

        if self._last_sample_rate:
            self._log_interval_samples = max(
                1, int(self._last_sample_rate * self._log_interval_sec)
            )
            if self._next_log_sample == 0:
                self._next_log_sample = self._log_interval_samples

            while self._total_samples >= self._next_log_sample:
                duration = self._total_samples / float(self._last_sample_rate)
                logger.info(
                    "🎚️ Client %s cumulative duration: %.3f s (frames=%d, rate=%d)",
                    self._label,
                    duration,
                    self._total_samples,
                    self._last_sample_rate,
                )
                self._next_log_sample += self._log_interval_samples

        return frame

    def stop(self) -> None:  # pragma: no cover - passthrough cleanup
        """Stop the proxy track while draining asynchronous cleanup hooks."""
        maybe_awaitable = super().stop()
        if inspect.isawaitable(maybe_awaitable):
            _drain_async(maybe_awaitable)

        source_stop = getattr(self._source, "stop", None)
        if callable(source_stop):
            maybe_awaitable = source_stop()
            if inspect.isawaitable(maybe_awaitable):  # type: ignore[misc]
                _drain_async(maybe_awaitable)


async def record_audio(track: MediaStreamTrack, output_path: Path) -> int:
    """Capture audio from a remote track and write it to disk.

    Args:
        track: Remote media track streaming audio from the worker.
        output_path: Destination where recorded audio should be written.

    Returns:
        Number of audio frames recorded to the output file.
    """
    total_frames = 0
    writer: wave.Wave_write | None = None
    sample_rate: int | None = None
    sample_width: int | None = None
    channels: int | None = None
    log_interval_frames = 0
    next_log_frame = 0

    _ensure_parent(output_path)

    try:
        while True:
            frame = await track.recv()
            assert isinstance(frame, AudioFrame)

            frame_channels = getattr(frame.layout, "channels", None)
            if frame_channels is None:
                frame_channels = getattr(frame, "channels", 1)
            if isinstance(frame_channels, list | tuple):
                frame_channels = len(frame_channels)
            frame_width = frame.format.bytes
            frame_rate = frame.sample_rate

            if writer is None:
                writer = wave.open(str(output_path), "wb")
                writer.setnchannels(frame_channels)
                writer.setsampwidth(frame_width)
                writer.setframerate(frame_rate)
                sample_rate = frame_rate
                sample_width = frame_width
                channels = frame_channels
                logger.info(
                    "📝 Output format: %s Hz, %s bytes/sample, %s channels", sample_rate, sample_width, channels
                )
                log_interval_frames = max(1, int(sample_rate * 0.5))
                next_log_frame = log_interval_frames
            else:
                if frame_rate != sample_rate or frame_width != sample_width or frame_channels != channels:
                    logger.warning(
                        "Frame format change detected (rate=%s width=%s channels=%s); adjusting writer",
                        frame_rate,
                        frame_width,
                        frame_channels,
                    )
                    writer.setframerate(frame_rate)
                    writer.setsampwidth(frame_width)
                    writer.setnchannels(frame_channels)
                    sample_rate = frame_rate
                    sample_width = frame_width
                    channels = frame_channels
                    log_interval_frames = max(1, int(sample_rate * 0.5)) if sample_rate else 0
                    next_log_frame = total_frames + log_interval_frames if log_interval_frames else 0

            assert channels is not None
            interleaved = _ensure_interleaved(frame, channels)

            if interleaved.size == 0:
                continue

            writer.writeframes(interleaved.tobytes())
            total_frames += interleaved.shape[0]

            if log_interval_frames and sample_rate:
                while total_frames >= next_log_frame:
                    duration = total_frames / float(sample_rate)
                    logger.info(
                        "🎚️ Client recv cumulative duration: %.3f s (frames=%d, rate=%d)",
                        duration,
                        total_frames,
                        sample_rate,
                    )
                    next_log_frame += log_interval_frames

            if sample_rate:
                duration = total_frames / float(sample_rate)
                logger.info(
                    "🎚️ Client recv cumulative duration: %.3f s (frames=%d, rate=%d)",
                    duration,
                    total_frames,
                    sample_rate,
                )
    except MediaStreamError:
        logger.info("Remote track ended; stopping recorder")
    finally:
        if writer is not None:
            writer.close()

    return total_frames


async def run_session(
    input_path: Path,
    output_path: Path,
    config: ClientConfig,
    expected_output: Path | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> bool:
    """Stream audio to the worker and optionally validate the recorded output.

    Args:
        input_path: Audio file to stream via WebRTC.
        output_path: Destination for the recorded remote audio.
        config: Runtime configuration controlling negotiation and waiting.
        expected_output: Optional reference file for size comparison.
        progress_cb: Optional callback used to report progress messages.

    Returns:
        ``True`` when audio is received and optional validation succeeds.
    """
    progress = progress_cb or (lambda message: logger.info(message))

    player = MediaPlayer(str(input_path))
    if player.audio is None:
        raise RuntimeError(f"No audio stream available in {input_path}")

    pc = RTCPeerConnection()
    control_channel: RTCDataChannel = pc.createDataChannel("control")
    control_ready = asyncio.Event()
    flush_done_event = asyncio.Event()
    flush_acknowledged = False

    @control_channel.on("open")
    def on_control_open() -> None:
        progress("🛰️ Control channel established")
        control_ready.set()

    @control_channel.on("close")
    def on_control_close() -> None:
        logger.info("Control data channel closed")

    @control_channel.on("message")
    def on_control_message(message: object) -> None:
        nonlocal flush_acknowledged

        if isinstance(message, bytes):
            try:
                message = message.decode("utf-8")
            except Exception:
                logger.warning("Received non-UTF8 payload on control channel; ignoring")
                return

        if isinstance(message, str) and message.strip().lower() == "flush_done":
            flush_acknowledged = True
            progress("🚰 Worker reported flush completion")
            flush_done_event.set()
        else:
            logger.debug("Ignoring unexpected control message: %r", message)

    record_task: asyncio.Task[int] | None = None
    recorder_result: int = 0
    send_track = DurationLoggingTrack(player.audio, label="send")

    @pc.on("track")
    async def on_track(track):
        """Handle remote tracks by starting the recorder when audio appears.

        Args:
            track: Remote track received from the worker.
        """
        nonlocal record_task
        if track.kind != "audio":
            logger.debug("Ignoring non-audio track of kind %s", track.kind)
            return
        progress("🔁 Remote audio track established")
        if record_task is None:
            record_task = asyncio.create_task(record_audio(track, output_path))

    pc.addTrack(send_track)

    try:
        await negotiate(pc, config)
        progress("📡 SDP negotiation complete")

        source_track = player.audio
        while source_track.readyState != "ended":
            await asyncio.sleep(0.05)

        await asyncio.sleep(config.wait_after_input)

        try:
            await asyncio.wait_for(control_ready.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("❌ Control data channel failed to open within timeout")
            return False

        if control_channel.readyState != "open":
            logger.error("❌ Control data channel not open after negotiation")
            return False

        try:
            control_channel.send("flush")
            progress("🚰 Requested worker flush")
        except Exception as exc:
            logger.error("❌ Failed to send flush request: %s", exc)
            return False

        try:
            await asyncio.wait_for(flush_done_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error("❌ Timed out waiting for worker flush acknowledgment")
            return False

        if not flush_acknowledged:
            logger.error("❌ Worker did not acknowledge flush request")
            return False

        if record_task is not None:
            try:
                recorder_result = await asyncio.wait_for(record_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.error("❌ Recorder task did not finish after flush")
                return False
        else:
            logger.error("❌ Remote audio track was never received")
            return False

        progress(f"💾 Captured {recorder_result} frames to {output_path}")
        success = recorder_result > 0 and output_path.exists()

        if success and expected_output is not None:
            if _validate_file_size(output_path, expected_output):
                progress("📐 Output file size within tolerance")
            else:
                progress("💥 Output file size validation failed")
                success = False

        progress("🏁 Session complete")
        return success
    finally:
        if control_channel.readyState == "open":
            try:
                control_channel.close()
            except Exception:
                logger.exception("Failed to close control channel cleanly")
        await pc.close()
        stop = getattr(player, "stop", None)
        if callable(stop):
            maybe_awaitable = stop()
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        elif getattr(player.audio, "stop", None):
            maybe_awaitable = player.audio.stop()
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        send_track.stop()


app = typer.Typer(help="Simple WebRTC client for exercising the worker")


@app.command()
def main(  # pragma: no cover - CLI entry point
    input_file: Path = typer.Argument(Path("data/test_input.wav"), help="Audio to stream via WebRTC"),
    output_file: Path = typer.Argument(Path("output.wav"), help="Where to save the transformed audio"),
    api_base: str = typer.Option("http://localhost:8000", "--api-base", "-a", help="Worker API base URL"),
    offer_path: str = typer.Option("/webrtc/offer", "--offer-path", help="Signalling endpoint path"),
    wait_after_input: float = typer.Option(
        1.0,
        "--wait-after-input",
        help="Seconds to wait for remote audio after local track finishes",
    ),
    expected_file: Path | None = typer.Option(
        Path("data/test_output.wav"),
        "--expected",
        "-e",
        help="Reference output for size validation",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Entry point for the WebRTC client command-line interface.

    Args:
        input_file: Audio to stream via WebRTC to the worker.
        output_file: Destination path for the received audio.
        api_base: Base URL of the worker API.
        offer_path: Endpoint path used for SDP signalling.
        wait_after_input: Seconds to wait after the local stream ends.
        expected_file: Optional reference file used for validation.
        verbose: Whether to enable verbose logging output.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    typer.echo("\n🎛️  Starting WebRTC test")
    typer.echo(f"   📥 Input: {input_file}")
    typer.echo(f"   💾 Output: {output_file}")
    typer.echo(f"   🌐 API: {api_base.rstrip('/')}{offer_path}")
    if expected_file:
        typer.echo(f"   ✅ Expected: {expected_file}")

    config = ClientConfig(api_base=api_base, offer_path=offer_path, wait_after_input=wait_after_input)

    def progress(message: str) -> None:
        """Display progress messages for the CLI workflow.

        Args:
            message: Text to echo to the terminal.
        """
        typer.echo(f"   {message}")

    success = asyncio.run(
        run_session(
            input_file,
            output_file,
            config,
            expected_output=expected_file,
            progress_cb=progress,
        )
    )

    if success:
        typer.echo("\n✅ Test completed successfully")
        typer.echo(f"📁 Output saved to {output_file}")
    else:
        typer.echo("\n❌ Test failed", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover - script execution guard
    app()
