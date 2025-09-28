#!/usr/bin/env python3
"""
WebSocket client for testing voice changer server.

This client mimics the behavior of the frontend UI by:
1. Connecting to the WebSocket server
2. Sending start_recording message
3. Streaming audio file data in chunks (like MediaRecorder would)
4. Sending stop_recording message
5. Receiving and saving the processed audio response

Usage:
    python client.py [audio_file] [output_file]
    python client.py test.m4a output.wav
"""

import asyncio
import io
import json
import logging
import sys
from pathlib import Path

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


class VoiceChangerClient:
    def __init__(self, server_url: str = "ws://localhost:8000/ws"):
        self.server_url = server_url
        self.websocket = None
        self.received_audio = []
        self.processing_complete = False

    async def connect(self):
        """Connect to the WebSocket server."""
        logger.info(f"Connecting to {self.server_url}")
        self.websocket = await websockets.connect(self.server_url)
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

                        if msg_type == "audio_complete":
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

    def save_received_audio(self, output_file: str):
        """Save received audio chunks to file."""
        if not self.received_audio:
            logger.warning("⚠️ No audio data received to save")
            return False

        # Combine all received audio chunks
        combined_audio = b"".join(self.received_audio)
        total_size = len(combined_audio)

        logger.info(f"💾 Saving {total_size} bytes to {output_file}")

        try:
            with open(output_file, "wb") as f:
                f.write(combined_audio)

            logger.info(f"✅ Successfully saved processed audio to {output_file}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save audio: {str(e)}")
            return False

    async def process_audio_file(self, input_file: str, output_file: str = "output.wav"):
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
            success = self.save_received_audio(output_file)

            if success:
                logger.info(f"🎉 Voice changing completed! Output saved to: {output_file}")
            else:
                logger.error("❌ Failed to save processed audio")

            return success

        except Exception as e:
            logger.error(f"❌ Error processing audio file: {str(e)}")
            return False

        finally:
            await self.disconnect()


async def main():
    """Main function."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python client.py <input_file> [output_file]")
        print("Example: python client.py data/test_input.webm output.wav")
        print("Example: python client.py test.m4a output.wav")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output.wav"

    # Validate input file
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    # Create client and process audio
    client = VoiceChangerClient()
    success = await client.process_audio_file(input_file, output_file)

    if success:
        logger.info("🎊 All done! You can play the output file to hear the result.")
        print(f"\nTo play the result: afplay {output_file}")  # macOS
        print(f"Or with any audio player: vlc {output_file}")
    else:
        logger.error("💥 Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
