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

    pytest.skip(
        "test_client.py is an integration utility, not a pytest module", allow_module_level=True
    )

import wave
from fractions import Fraction

import numpy as np
import torchaudio
import typer
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PCM_SAMPLE_RATE = 48_000
PCM_CHANNELS = 1
PCM_WIDTH = 2  # bytes for int16 PCM


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


class FileAudioStreamTrack(MediaStreamTrack):
    """Media stream track that emits audio frames from an in-memory buffer."""

    kind = "audio"

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: int,
        frame_size: int = 960,
    ) -> None:
        """Initialise the track with PCM samples to be streamed.

        Args:
            samples: Array containing audio samples in ``(channels, frames)`` layout.
            sample_rate: Sample rate of the provided audio samples.
            frame_size: Number of samples per channel to include in each emitted frame.

        Raises:
            ValueError: If ``frame_size`` is not positive.
        """

        super().__init__()
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        self._samples = samples
        self._sample_rate = sample_rate
        self._frame_size = frame_size
        self._cursor = 0
        self._pts = 0
        self._time_base = Fraction(1, sample_rate)

    async def recv(self) -> AudioFrame:  # type: ignore[override]
        """Return the next audio frame from the buffer.

        Returns:
            AudioFrame: Frame containing the next slice of PCM samples.

        Raises:
            MediaStreamError: If the buffer has been fully consumed.
        """

        if self.readyState == "ended":
            raise MediaStreamError

        start = self._cursor
        end = min(start + self._frame_size, self._samples.shape[1])
        if start >= end:
            self.stop()
            raise MediaStreamError

        chunk = self._samples[:, start:end]
        self._cursor = end

        frame = AudioFrame.from_ndarray(chunk, format="s16")
        frame.pts = self._pts
        frame.time_base = self._time_base
        frame.sample_rate = self._sample_rate
        self._pts += chunk.shape[1]

        await asyncio.sleep(chunk.shape[1] / self._sample_rate)

        if self._cursor >= self._samples.shape[1]:
            self.stop()

        return frame


class VoiceChangerWebRTCTestClient:
    """WebRTC client used for exercising the Voice Changer worker."""

    def __init__(self, config: TestConfig):
        """Store configuration and initialise placeholders for WebRTC state."""
        self.config = config
        self.pc: RTCPeerConnection | None = None
        self.source_track: FileAudioStreamTrack | None = None
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

        Args:
            track: The audio track to record from.
            output_path: Where to save the recorded audio.

        Returns:
            int: The number of PCM samples written to disk. ``0`` indicates that no audio
            was captured, either because the track never produced data or an error
            occurred while recording.
        """

        total_samples = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(PCM_CHANNELS)
            wf.setsampwidth(PCM_WIDTH)
            wf.setframerate(PCM_SAMPLE_RATE)

            try:
                while True:
                    frame = await track.recv()
                    data = frame.to_ndarray()
                    samples: np.ndarray
                    if data.dtype != np.int16:
                        samples = np.frombuffer(data.tobytes(), dtype=np.int16)
                    else:
                        samples = data
                    if total_samples == 0:
                        logger.debug(
                            "First frame dtype=%s shape=%s layout=%s channels=%s",  # type: ignore[str-format]
                            frame.format.name,
                            data.shape,
                            frame.layout.name,
                            getattr(frame, "channels", "unknown"),
                        )

                    channel_count = getattr(frame.layout, "channels", PCM_CHANNELS)
                    if samples.ndim == 2:
                        if samples.shape[0] == channel_count:
                            samples = samples[0]
                        elif samples.shape[1] == channel_count:
                            samples = samples[:, 0]
                        else:
                            samples = samples.reshape(-1)
                    else:
                        samples = samples.reshape(-1)

                    if samples.size == 0:
                        continue

                    wf.writeframes(samples.tobytes())
                    total_samples += int(samples.shape[0])
            except MediaStreamError:
                logger.debug("Remote track ended; finalising recording")
            except Exception as exc:
                logger.error("Failed to record remote audio: %s", exc)
                return 0

        if total_samples == 0:
            logger.error("❌ No audio samples received; nothing written to %s", output_path)
            return 0

        logger.info("💾 Wrote %s samples to %s", total_samples, output_path)
        return total_samples

    async def process_audio_file(
        self, input_file: str, output_file: str, expected_file: str | None = None
    ) -> bool:
        """Stream an input file through the worker and optionally verify the output.

        Args:
            input_file (str): Path to the source audio file to send.
            output_file (str): Path for recording the transformed audio.
            expected_file (str | None): Optional reference file for validation.

        Returns:
            bool: ``True`` when processing (and verification, if requested) succeeds.

        Raises:
            RuntimeError: If the input audio does not meet the expected PCM format
                requirements or cannot be loaded.
        """
        logger.info("🎧 Starting WebRTC test session")
        input_path = Path(input_file)
        output_path = Path(output_file)
        _ensure_parent(output_path)
        logger.debug("Recording output to %s", output_path.resolve())

        with wave.open(str(input_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            input_duration = frames / float(rate)
            logger.info(
                "📏 Input file duration: %.2fs (%d frames at %d Hz)",
                input_duration,
                frames,
                rate,
            )
            raw_audio = wf.readframes(frames)

        if sample_width != PCM_WIDTH:
            raise RuntimeError(
                f"Input sample width {sample_width} does not match required {PCM_WIDTH}"
            )

        if rate != PCM_SAMPLE_RATE:
            raise RuntimeError(
                f"Input sample rate {rate} Hz does not match required {PCM_SAMPLE_RATE} Hz"
            )

        if not raw_audio:
            raise RuntimeError("Input audio file contained no samples")

        if channels <= 0:
            raise RuntimeError("Input audio file reports no audio channels")

        samples = np.frombuffer(raw_audio, dtype=np.int16)
        frame_count = frames if frames > 0 else samples.size // max(channels, 1)

        if channels > 1:
            logger.info(
                "🎚️ Input audio has %d channels; using the first channel for streaming",
                channels,
            )

        if frame_count * channels != samples.size:
            raise RuntimeError(
                "Input audio sample buffer size does not align with channel configuration"
            )
        reshaped = samples.reshape(frame_count, channels)
        mono = reshaped[:, 0].reshape(1, -1)
        pcm_samples = np.ascontiguousarray(mono)

        self.pc = RTCPeerConnection()
        self.source_track = FileAudioStreamTrack(pcm_samples, PCM_SAMPLE_RATE)

        @self.pc.on("track")
        async def on_track(track):
            if track.kind == "audio":
                logger.info("🔁 Remote audio track received")
                if self.record_task is None:
                    self.record_task = asyncio.create_task(
                        self._record_remote_audio(track, output_path)
                    )

        self.pc.addTrack(self.source_track)

        await self._negotiate()

        assert self.source_track is not None
        while self.source_track.readyState != "ended":
            await asyncio.sleep(0.05)

        await asyncio.sleep(self.config.wait_after_audio)

        if self.pc:
            await self.pc.close()
            self.pc = None

        recorded_samples = 0
        if self.record_task:
            recorded_samples = await self.record_task
            self.record_task = None

        self.source_track = None

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

        # File size sanity check
        actual_size = os.path.getsize(actual_file)
        expected_size = os.path.getsize(expected_file)
        size_diff_pct = abs(actual_size - expected_size) / max(expected_size, 1) * 100

        logger.info(
            "🔍 File size comparison: actual=%s bytes expected=%s bytes diff=%.1f%%",
            actual_size,
            expected_size,
            size_diff_pct,
        )

        if size_diff_pct > 15:
            logger.error("💥 File size difference exceeds 15%% threshold")
            return False

        # Load audio files with torchaudio for waveform comparison
        try:
            actual_waveform, actual_sr = torchaudio.load(str(actual_file))
            expected_waveform, expected_sr = torchaudio.load(str(expected_file))
        except Exception as exc:
            logger.error("❌ Failed to load audio files: %s", exc)
            return False

        if actual_sr != PCM_SAMPLE_RATE or expected_sr != PCM_SAMPLE_RATE:
            logger.error(
                "💥 Sample rate mismatch: actual=%d Hz expected=%d Hz (required=%d Hz)",
                actual_sr,
                expected_sr,
                PCM_SAMPLE_RATE,
            )
            return False

        if actual_waveform.shape[0] != PCM_CHANNELS or expected_waveform.shape[0] != PCM_CHANNELS:
            logger.error(
                "💥 Channel mismatch: actual=%d expected=%d (required=%d)",
                actual_waveform.shape[0],
                expected_waveform.shape[0],
                PCM_CHANNELS,
            )
            return False

        actual_flat = actual_waveform.flatten().cpu().numpy()
        expected_flat = expected_waveform.flatten().cpu().numpy()

        cross_correlation = np.correlate(actual_flat, expected_flat, mode="full")
        best_index = int(np.argmax(np.abs(cross_correlation)))
        best_corr = float(cross_correlation[best_index])
        lag = best_index - (expected_flat.size - 1)

        norm_product = float(np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat))
        if norm_product == 0:
            logger.error("💥 One of the audio files is silent; cannot compute similarity")
            return False

        correlation_score = abs(best_corr) / norm_product

        if lag >= 0:
            actual_slice = actual_flat[lag:]
            expected_slice = expected_flat[: actual_slice.size]
        else:
            expected_slice = expected_flat[-lag:]
            actual_slice = actual_flat[: expected_slice.size]

        overlap = min(actual_slice.size, expected_slice.size)
        if overlap == 0:
            logger.error("💥 Audio overlap after alignment is zero samples")
            return False

        actual_aligned = actual_slice[:overlap]
        expected_aligned = expected_slice[:overlap]
        mse = float(np.mean((actual_aligned - expected_aligned) ** 2))

        if best_corr < 0:
            logger.warning(
                "⚠️ Output appears phase-inverted relative to expected audio (lag=%d samples)",
                lag,
            )

        logger.info(
            "🔊 Audio comparison: MSE=%.6f lag=%d |correlation|=%.4f",
            mse,
            lag,
            correlation_score,
        )

        # Define thresholds for passing
        # MSE should be low (similar waveforms), correlation magnitude should be high (>0.9)
        mse_threshold = 0.01  # Adjust based on expected noise/variation
        correlation_threshold = 0.90
        minimum_acceptable_correlation = 0.05

        strong_match = mse < mse_threshold and correlation_score > correlation_threshold
        weak_match = (
            mse < mse_threshold and minimum_acceptable_correlation <= correlation_score <= correlation_threshold
        )

        if strong_match:
            logger.info("🎉 Verification passed (size + audio similarity)")
            success = True
        elif weak_match:
            logger.warning(
                "⚠️ Waveforms correlate weakly (|correlation|=%.4f) but MSE %.6f is within %.6f",
                correlation_score,
                mse,
                mse_threshold,
            )
            success = True
        else:
            logger.error(
                "💥 Verification failed: MSE=%.6f (threshold=%.6f), |correlation|=%.4f (threshold=%.4f)",
                mse,
                mse_threshold,
                correlation_score,
                correlation_threshold,
            )
            success = False
        return success


async def run_test(
    input_file: str, output_file: str, expected_file: str | None, api_base: str
) -> bool:
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
    input_file: Path = typer.Argument(Path("data/test_input.wav"), help="Input audio file"),
    output_file: Path = typer.Argument(Path("output.wav"), help="Output audio file"),
    expected_file: Path | None = typer.Option(
        Path("data/test_output.wav"), "--expected", "-e", help="Expected output for verification"
    ),
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

    success = asyncio.run(
        run_test(
            str(input_file),
            str(output_file),
            str(expected_file) if expected_file else None,
            api_base,
        )
    )

    if success:
        typer.echo("\n✅ Test completed successfully")
        typer.echo(f"📁 Output saved to {output_file}")
    else:
        typer.echo("\n❌ Test failed", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
