#!/usr/bin/env python3
"""Voice changer worker that pitch-shifts PCM audio frames over WebRTC."""

from __future__ import annotations

import asyncio
import io
import logging
import librosa
import numpy as np
import soundfile as sf
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from worker import WorkerContext, WorkerSettings, make_worker_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

for noisy_logger in ("httpx", "httpcore"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


class SDPModel(BaseModel):
    """Pydantic model representing a WebRTC SDP payload."""

    sdp: str = Field(description="Session description protocol body provided by the browser.")
    type: str = Field(description="SDP type such as 'offer' or 'answer'.")


TARGET_SAMPLE_RATE = 48_000
DEFAULT_PITCH_SHIFT = 4.0


class PitchShiftTrack(MediaStreamTrack):
    """Forward PCM frames while applying a pitch shift when requested."""

    kind = "audio"

    def __init__(self, source: MediaStreamTrack, pitch_shift_semitones: float = DEFAULT_PITCH_SHIFT):
        """Initialize the track wrapper that performs pitch shifting on audio frames.

        Args:
            source: Upstream media track supplying PCM audio frames.
            pitch_shift_semitones: Number of semitones to shift the audio signal.
        """
        super().__init__()
        self._source = source
        self._pitch_shift = pitch_shift_semitones
        self._frame_count = 0
        self._input_samples_total = 0
        self._output_samples_total = 0
        self._last_input_rate = TARGET_SAMPLE_RATE
        self._last_output_rate = TARGET_SAMPLE_RATE

    async def recv(self) -> AudioFrame:
        """Receive, transform, and return the next audio frame from the source.

        Returns:
            Audio frame with the configured pitch shift applied.
        """
        frame: AudioFrame = await self._source.recv()
        self._frame_count += 1

        sample_rate = frame.sample_rate or TARGET_SAMPLE_RATE
        format_name = frame.format.name
        layout_name = frame.layout.name

        pcm = frame.to_ndarray()
        audio_data = np.asarray(pcm, dtype=np.int16)
        if audio_data.size == 0 or abs(self._pitch_shift) < 1e-6:
            return frame

        if audio_data.ndim == 1:
            data_for_io = audio_data
            frame_samples = audio_data.shape[0]
        else:
            data_for_io = audio_data.transpose(1, 0)
            frame_samples = audio_data.shape[1]

        self._last_input_rate = sample_rate
        self._input_samples_total += frame_samples
        self._log_progress("recv", self._input_samples_total, self._last_input_rate)

        if self._frame_count % 100 == 1:
            logger.info(
                "📊 Processing frame #%d: sample_rate=%s format=%s layout=%s pitch_shift=%.2f",
                self._frame_count,
                sample_rate,
                format_name,
                layout_name,
                self._pitch_shift,
            )

        with io.BytesIO() as buffer:
            sf.write(
                buffer,
                np.ascontiguousarray(data_for_io),
                sample_rate,
                subtype="PCM_16",
                format="WAV",
            )
            buffer.seek(0)
            float_samples, loaded_sr = librosa.load(
                buffer,
                sr=sample_rate,
                mono=False,
            )

        if float_samples.ndim == 1:
            float_samples = float_samples[np.newaxis, :]

        try:
            shifted_channels: list[np.ndarray] = []
            for channel in range(float_samples.shape[0]):
                channel_shifted = librosa.effects.pitch_shift(
                    float_samples[channel],
                    sr=loaded_sr,
                    n_steps=self._pitch_shift,
                )
                channel_shifted = librosa.util.fix_length(channel_shifted, size=frame_samples)
                shifted_channels.append(channel_shifted.astype(np.float32))
            shifted = np.stack(shifted_channels, axis=0)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Pitch shifting error: %s", exc)
            return frame

        shifted = np.clip(shifted, -1.0, 1.0)
        shifted_pcm = (shifted * 32767.0).astype(np.int16)

        if shifted_pcm.ndim == 1:
            array = shifted_pcm[np.newaxis, :]
        else:
            array = shifted_pcm

        out_frame = AudioFrame.from_ndarray(np.ascontiguousarray(array), format=format_name, layout=layout_name)
        out_frame.sample_rate = loaded_sr
        out_frame.pts = frame.pts
        out_frame.time_base = frame.time_base

        self._last_output_rate = loaded_sr
        self._output_samples_total += frame_samples
        self._log_progress("send", self._output_samples_total, self._last_output_rate)

        return out_frame

    def _log_progress(self, direction: str, total_samples: int, sample_rate: int) -> None:
        """Emit structured logs describing audio throughput.

        Args:
            direction: Indicates whether logging covers received or sent samples.
            total_samples: Aggregate sample count processed to date.
            sample_rate: Sample rate used when converting to seconds of audio.
        """
        if not sample_rate or total_samples <= 0:
            return
        duration = total_samples / float(sample_rate)
        logger.info(
            "🎚️ Worker %s cumulative duration: %.3f s (samples=%d, rate=%d)",
            direction,
            duration,
            total_samples,
            sample_rate,
        )


SETTINGS = WorkerSettings(
    title="Voice Changer WebRTC Worker",
    version="3.0.0",
    service_name="voice_changer_webrtc_worker",
    capabilities=["pitch_shifting", "webrtc", "streaming_audio"],
)


def register_routes(app: FastAPI, ctx: WorkerContext) -> None:
    @app.post("/offer")
    async def handle_offer(payload: SDPModel) -> dict[str, str]:
        pc = RTCPeerConnection()
        ctx.active_peers.add(pc)
        logger.info("📡 Received new WebRTC offer; active peers: %d", len(ctx.active_peers))

        cleanup_started = False

        async def cleanup() -> None:
            nonlocal cleanup_started
            if cleanup_started:
                return
            cleanup_started = True
            await ctx.cleanup_peer(pc)

        @pc.on("connectionstatechange")
        async def on_connection_state_change() -> None:  # pragma: no cover - network dependent
            logger.info("Peer connection state: %s", pc.connectionState)
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await cleanup()

        track_ready = asyncio.Event()

        @pc.on("track")
        async def on_track(track: MediaStreamTrack) -> None:
            logger.info("🎙️ Incoming track: %s", track.kind)
            if track.kind != "audio":
                logger.debug("Ignoring non-audio track of kind %s", track.kind)
                return

            transformed = PitchShiftTrack(track)
            pc.addTrack(transformed)
            track_ready.set()

            @track.on("ended")
            async def on_track_ended() -> None:
                logger.info("🏁 Audio track ended")
                await cleanup()

        offer = RTCSessionDescription(sdp=payload.sdp, type=payload.type)

        try:
            await pc.setRemoteDescription(offer)
            try:
                await asyncio.wait_for(track_ready.wait(), timeout=5.0)
            except TimeoutError:
                logger.warning("⚠️ No audio track received before answer generation")
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await ctx.wait_for_ice_completion(pc)
        except Exception as exc:  # pragma: no cover - network failure handling
            await cleanup()
            raise HTTPException(status_code=500, detail=f"Failed to process SDP offer: {exc}")

        logger.info("✅ Generated SDP answer")
        assert pc.localDescription is not None
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


app = make_worker_app(settings=SETTINGS, logger=logger, register_routes=register_routes)


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting Voice Changer WebRTC Worker Service")
    uvicorn.run("voice_changer:app", host="127.0.0.1", port=8001, log_level="info")
