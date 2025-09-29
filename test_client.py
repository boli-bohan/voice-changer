#!/usr/bin/env python3
"""
WebSocket test client for voice changer server.

This enhanced test client mimics the behavior of the frontend UI and provides verification:
1. Connecting to the WebSocket server
2. Sending start_recording message
3. Streaming audio file data in chunks (like MediaRecorder would)
4. Sending stop_recording message
5. Receiving and saving the processed audio response
6. Verifying output against expected results

Usage:
    python test_client.py <input_file> <output_file> [expected_file]
    python test_client.py data/test_input.webm actual_output.wav data/test_output.wav
    python test_client.py test.m4a output.wav
"""

import asyncio
import io
import json
import logging
import os
import wave
from dataclasses import dataclass
from pathlib import Path

import typer
import websockets
from pydub import AudioSegment
from pydub.utils import which

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for ffmpeg
if not which("ffmpeg"):
    logger.warning("ffmpeg not found - some audio formats may not work")
    logger.info("Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Ubuntu)")


@dataclass
class AudioFormat:
    sample_rate: int = 44100
    channels: int = 1
    bits_per_sample: int = 16
    encoding: str = "pcm_s16le"


class VoiceChangerTestClient:
    def __init__(self, server_url: str = "ws://localhost:8000/ws"):
        self.server_url = server_url
        self.websocket = None
        self.received_audio = []
        self.processing_complete = False
        self.test_passed = False
        self.audio_format = AudioFormat()

    async def connect(self):
        """Connect to the WebSocket server."""
        logger.info(f"Connecting to {self.server_url}")
        # Set proper Origin header to pass the server's CheckOrigin check
        extra_headers = {"Origin": "http://localhost:5173"}
        self.websocket = await websockets.connect(self.server_url, additional_headers=extra_headers)
        logger.info("✅ Connected to server")

    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            logger.info("🔌 Disconnected from server")

    def load_webm_file(self, input_file: str) -> io.BytesIO:
        """Load WebM file directly or convert audio file to WebM format."""
        logger.info(f"📁 Loading audio file: {input_file}")

        if input_file.endswith('.webm'):
            # Load WebM file directly
            with open(input_file, 'rb') as f:
                webm_data = f.read()
            webm_buffer = io.BytesIO(webm_data)
            logger.info(f"📂 Loaded WebM file directly: {len(webm_data)} bytes")
        else:
            # Convert to WebM format to match browser MediaRecorder API
            audio = AudioSegment.from_file(input_file)

            # Convert to format similar to MediaRecorder WebM output
            # Use 48kHz sample rate (common for WebM), stereo, 16-bit depth
            audio = audio.set_frame_rate(48000).set_channels(2).set_sample_width(2)

            # Export to bytes buffer as WebM (using opus codec)
            webm_buffer = io.BytesIO()
            audio.export(webm_buffer, format="webm", codec="libopus")
            webm_buffer.seek(0)

            logger.info(f"🔄 Converted to WebM: {len(webm_buffer.getvalue())} bytes, "
                        f"{len(audio)}ms duration, {audio.frame_rate}Hz")

        return webm_buffer

    def chunk_audio_data(self, webm_buffer: io.BytesIO, chunk_size: int = 4096):
        """Split audio data into chunks like MediaRecorder would do."""
        webm_buffer.seek(0)
        chunks = []

        while True:
            chunk = webm_buffer.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)

        logger.info(f"📊 Split audio into {len(chunks)} chunks of ~{chunk_size} bytes each")
        return chunks

    async def send_start_recording(self):
        """Send start_recording message to server."""
        message = {"type": "start_recording"}
        await self.websocket.send(json.dumps(message))
        logger.info("🎙️ Sent start_recording message")

    async def send_stop_recording(self):
        """Send stop_recording message to server."""
        message = {"type": "stop_recording"}
        await self.websocket.send(json.dumps(message))
        logger.info("⏹️ Sent stop_recording message")

    async def stream_audio_chunks(self, chunks: list, delay: float = 0.1):
        """Stream audio chunks to server with delay to simulate real-time recording."""
        logger.info(f"📤 Streaming {len(chunks)} audio chunks (delay: {delay}s between chunks)")

        for i, chunk in enumerate(chunks, 1):
            await self.websocket.send(chunk)
            logger.debug(f"📦 Sent chunk {i}/{len(chunks)}: {len(chunk)} bytes")

            # Add small delay to simulate real-time streaming
            if i < len(chunks):  # Don't delay after the last chunk
                await asyncio.sleep(delay)

        logger.info("✅ Finished streaming audio chunks")

    async def receive_messages(self):
        """Listen for messages and audio data from server."""
        logger.info("👂 Starting to listen for server responses")

        try:
            async for message in self.websocket:
                if isinstance(message, str):
                    # JSON message
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "unknown")
                        msg_content = data.get("message", "")

                        logger.info(f"📨 Received {msg_type}: {msg_content}")

                        if msg_type == "audio_format":
                            self.audio_format = AudioFormat(
                                sample_rate=data.get("sample_rate", self.audio_format.sample_rate),
                                channels=data.get("channels", self.audio_format.channels),
                                bits_per_sample=data.get("bits_per_sample", self.audio_format.bits_per_sample),
                                encoding=data.get("encoding", self.audio_format.encoding),
                            )
                            logger.info(
                                "ℹ️ Received audio format: %s Hz, %s channels, %s-bit %s",
                                self.audio_format.sample_rate,
                                self.audio_format.channels,
                                self.audio_format.bits_per_sample,
                                self.audio_format.encoding,
                            )
                        elif msg_type in {"audio_complete", "streaming_completed", "done"}:
                            logger.info("🎵 Audio processing completed!")
                            self.processing_complete = True
                            break
                        elif msg_type == "error":
                            logger.error(f"❌ Server error: {msg_content}")
                            break

                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode JSON message: {message}")

                elif isinstance(message, bytes):
                    # Binary audio data
                    self.received_audio.append(message)
                    logger.debug(f"🎵 Received audio chunk: {len(message)} bytes")

        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Connection closed by server")
        except Exception as e:
            logger.error(f"❌ Error receiving messages: {str(e)}")

    def save_received_audio(self, output_file: str) -> bool:
        """Save received audio chunks to file."""
        if not self.received_audio:
            logger.warning("⚠️ No audio data received to save")
            return False

        # Combine all received audio chunks
        combined_audio = b"".join(self.received_audio)
        total_size = len(combined_audio)

        logger.info(f"💾 Saving {total_size} bytes to {output_file}")

        try:
            output_path = Path(output_file)
            if output_path.suffix.lower() == ".wav":
                with wave.open(output_file, "wb") as wav_file:
                    wav_file.setnchannels(self.audio_format.channels)
                    wav_file.setsampwidth(self.audio_format.bits_per_sample // 8)
                    wav_file.setframerate(self.audio_format.sample_rate)
                    wav_file.writeframes(combined_audio)
                logger.info(
                    "✅ Saved WAV audio: %s (%s Hz, %s channels)",
                    output_file,
                    self.audio_format.sample_rate,
                    self.audio_format.channels,
                )
            else:
                with open(output_file, "wb") as f:
                    f.write(combined_audio)
                logger.info("✅ Saved raw PCM audio: %s", output_file)

            return True

        except Exception as e:
            logger.error(f"❌ Failed to save audio: {str(e)}")
            return False

    def verify_output(self, actual_file: str, expected_file: str | None = None) -> tuple[bool, str]:
        """Verify the generated output against expected results by comparing file sizes."""
        if not expected_file:
            logger.info("🔍 No expected output file provided - skipping verification")
            return True, "No verification performed"

        if not Path(expected_file).exists():
            logger.warning(f"⚠️ Expected file not found: {expected_file}")
            return False, f"Expected file not found: {expected_file}"

        if not Path(actual_file).exists():
            logger.error(f"❌ Generated file not found: {actual_file}")
            return False, f"Generated file not found: {actual_file}"

        try:
            # Get file sizes
            actual_size = os.path.getsize(actual_file)
            expected_size = os.path.getsize(expected_file)

            # Check file size with tolerance (allow 10% difference to account for streaming variability)
            size_diff_pct = abs(actual_size - expected_size) / expected_size * 100
            size_ok = size_diff_pct <= 10

            logger.info("🔍 File size comparison:")
            logger.info(f"  Actual: {actual_size:,} bytes")
            logger.info(f"  Expected: {expected_size:,} bytes")
            logger.info(f"  Difference: {size_diff_pct:.1f}%")

            if size_ok:
                logger.info("🎉 Output verification PASSED! File sizes match within tolerance.")
                return True, f"File size verification passed: {size_diff_pct:.1f}% difference"
            else:
                logger.error(f"💥 Output verification FAILED! File size difference too large: {size_diff_pct:.1f}%")
                return False, f"File size verification failed: {size_diff_pct:.1f}% difference (>1% tolerance)"

        except Exception as e:
            logger.error(f"❌ Error during verification: {str(e)}")
            return False, f"Verification error: {str(e)}"

    async def process_audio_file(self, input_file: str, output_file: str = "output.wav", expected_file: str | None = None):
        """Main method to process an audio file through the voice changer."""
        try:
            # Step 1: Connect to server
            await self.connect()

            # Step 2: Load and prepare audio
            webm_buffer = self.load_webm_file(input_file)
            chunks = self.chunk_audio_data(webm_buffer)

            # Step 3: Start listening for responses
            listen_task = asyncio.create_task(self.receive_messages())

            # Step 4: Send recording workflow
            await self.send_start_recording()

            # Wait a moment for server to be ready
            await asyncio.sleep(0.1)

            # Step 5: Stream audio data
            await self.stream_audio_chunks(chunks, delay=0.05)  # Faster streaming

            # Step 6: Signal end of recording
            await self.send_stop_recording()

            # Step 7: Wait for processing to complete
            logger.info("⏳ Waiting for audio processing to complete...")
            await listen_task

            # Step 8: Save result
            save_success = self.save_received_audio(output_file)

            if not save_success:
                logger.error("❌ Failed to save processed audio")
                return False

            logger.info(f"🎉 Voice changing completed! Output saved to: {output_file}")

            # Step 9: Verify output if expected file provided
            verify_success, verify_msg = self.verify_output(output_file, expected_file)
            logger.info(f"📊 Verification result: {verify_msg}")

            self.test_passed = verify_success
            return save_success and (verify_success if expected_file else True)

        except Exception as e:
            logger.error(f"❌ Error processing audio file: {str(e)}")
            return False

        finally:
            await self.disconnect()


app = typer.Typer(help="Voice Changer End-to-End Test Client")


@app.command()
def main_sync(
    input_file: Path | None = typer.Argument(
        None,
        help="Input audio file path",
    ),
    output_file: Path | None = typer.Argument(
        None,
        help="Output audio file path",
    ),
    expected_file: Path | None = typer.Option(
        None,
        "--expected",
        "-e",
        help="Expected output file for verification",
    ),
    server_url: str = typer.Option(
        "ws://localhost:8000/ws",
        "--server",
        "-s",
        help="WebSocket server URL",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Run end-to-end test of the voice changer system.

    By default, uses test files from the data/ directory if no arguments provided.
    """
    # Set up logging level
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Use defaults if no arguments provided
    if input_file is None:
        input_file = Path("data/test_input.webm")
        typer.echo("🔧 Using default input file: data/test_input.webm")

    if output_file is None:
        output_file = Path("output.wav")
        typer.echo("🔧 Using default output file: output.wav")

    if expected_file is None and input_file == Path("data/test_input.webm"):
        expected_file = Path("data/test_output.wav")
        typer.echo("🔧 Using default expected file: data/test_output.wav")

    # Validate input file exists
    if not input_file.exists():
        typer.echo(f"❌ Input file not found: {input_file}", err=True)
        raise typer.Exit(1)

    # Validate expected file if provided
    if expected_file and not expected_file.exists():
        typer.echo(
            f"⚠️ Expected file not found: {expected_file} - continuing without verification",
            err=True
        )
        expected_file = None

    typer.echo("\n🧪 Starting end-to-end test:")
    typer.echo(f"   📁 Input: {input_file}")
    typer.echo(f"   📄 Output: {output_file}")
    if expected_file:
        typer.echo(f"   ✅ Expected: {expected_file}")
    typer.echo(f"   🌐 Server: {server_url}\n")

    # Run the async main function
    success = asyncio.run(run_test(
        str(input_file), str(output_file), str(expected_file) if expected_file else None, server_url
    ))

    # Get the client result for test status
    # This is a simplified approach - in a real scenario we'd return more details
    client_test_passed = success  # Simplified for now

    if success:
        if expected_file and client_test_passed:
            logger.info("🎊 END-TO-END TEST PASSED! All checks successful.")
            typer.echo("\n✅ Test Result: PASSED", color=True)
            typer.echo(f"📁 Output file: {output_file}")
            typer.echo(f"🎵 Play result: afplay {output_file}")  # macOS
        else:
            logger.info("🎊 Audio processing completed successfully!")
            typer.echo("\n✅ Processing Result: SUCCESS", color=True)
            typer.echo(f"📁 Output file: {output_file}")
            typer.echo(f"🎵 Play result: afplay {output_file}")  # macOS
    else:
        if expected_file:
            logger.error("💥 END-TO-END TEST FAILED!")
            typer.echo("\n❌ Test Result: FAILED", color=True, err=True)
        else:
            logger.error("💥 Audio processing failed!")
            typer.echo("\n❌ Processing Result: FAILED", color=True, err=True)
        raise typer.Exit(1)


async def run_test(
    input_file: str, output_file: str, expected_file: str | None, server_url: str
) -> bool:
    """Async function to run the actual test."""
    # Create client and process audio
    client = VoiceChangerTestClient(server_url=server_url)
    success = await client.process_audio_file(input_file, output_file, expected_file)
    return success and (client.test_passed if expected_file else True)


if __name__ == "__main__":
    app()
