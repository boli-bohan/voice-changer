# Voice Changer

A real-time voice changing application with push-to-talk functionality. Built with FastAPI backend and React TypeScript frontend.

## Overview

This project implements a push-to-talk voice changer where users can:
1. Press and hold a button to start recording
2. Release the button to stop recording and automatically process the audio
3. Hear the pitch-shifted version of their voice played back

## Architecture

![Architecture](assets/arch.png)

- **API Server**: FastAPI with WebSocket connections to frontend and worker
- **Voice Changer Worker**: Dedicated FastAPI service for real-time audio processing
- **Frontend**: React TypeScript with push-to-talk WebSocket streaming
- **Audio Processing**: Streaming pitch shifting using librosa with WebM→WAV conversion
- **Real-time Architecture**: Immediate playback while processing continues in background

## Project Structure

```
voice-changer/
├── Justfile                    # Task automation
├── pyproject.toml             # Python dependencies and config
├── server.py                  # FastAPI WebSocket server
├── shift_pitch.py             # Audio processing script
├── plan.md                    # Detailed implementation plan
├── frontend/                  # React TypeScript frontend
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   └── PushToTalkButton.tsx
│   │   └── hooks/
│   │       ├── useWebSocket.ts
│   │       └── useAudioRecorder.ts
│   └── public/
└── temp_audio/               # Temporary audio files (auto-created)
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) (Python package installer)
- [Just](https://github.com/casey/just) (Command runner)

### Install Prerequisites

**macOS:**
```bash
brew install uv just
```

**Ubuntu/Debian:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
```

## Quick Start

1. **Clone and navigate to the project:**
   ```bash
   cd voice-changer
   ```

2. **Install dependencies:**
   ```bash
   just install
   ```

3. **Start all services (Streaming Mode):**
   ```bash
   just up
   ```
   This starts:
   - API Server (port 8000)
   - Voice Changer Worker (port 8001)
   - Frontend UI (port 5173)

4. **Access the application:**
   - Frontend UI: http://localhost:5173
   - API Server: http://127.0.0.1:8000
   - Voice Changer Worker: http://127.0.0.1:8001

5. **Use the voice changer:**
   - Press and hold "Push to Talk" button
   - Speak into your microphone
   - Release button to hear real-time processed audio!

6. **Stop all services:**
   ```bash
   just down
   ```

## Development Commands

- **Individual services:**
  ```bash
  just api     # API server only
  just worker  # Voice changer worker only
  ```

- **Service management:**
  ```bash
  just status  # Check running services
  just down    # Stop all services
  ```

## Usage

### Web Interface

1. Open http://localhost:5173 in your browser
2. Allow microphone access when prompted
3. Press and hold the large circular button to start recording
4. Speak into your microphone while holding the button
5. Release the button to stop recording
6. Wait for processing (a few seconds)
7. Listen to your pitch-shifted voice played back automatically

### Voice Processing

The application applies a +4 semitone pitch shift by default, making voices sound higher. You can modify the pitch shift amount in the `shift_pitch.py` script.

## Available Commands

```bash
just --list                 # Show all available commands
just install               # Install backend and frontend dependencies
just up                    # Start both backend and frontend servers
just down                  # Stop all services
just lint                  # Run linting on backend (frontend disabled temporarily)
just format                # Format code with ruff and prettier
just clean                 # Clean temporary files and build artifacts
just test                  # Run tests (when implemented)
```

## Development

### Backend Development

The FastAPI server (`server.py`) handles:
- WebSocket connections for real-time audio streaming
- Temporary file management for audio processing
- Integration with the pitch-shifting script
- CORS configuration for frontend communication

Key endpoints:
- `GET /` - Health check and API info
- `GET /health` - Detailed health status
- `WebSocket /ws` - Audio streaming endpoint

### Frontend Development

The React frontend provides:
- Push-to-talk button interface
- Audio recording using MediaRecorder API
- WebSocket communication with backend
- Real-time status updates and error handling

Key components:
- `PushToTalkButton` - Main interface component
- `useWebSocket` - WebSocket connection management
- `useAudioRecorder` - Audio recording functionality

### Audio Processing

The `shift_pitch.py` script:
- Accepts input/output file paths and pitch shift amount
- Uses librosa for high-quality pitch shifting
- Supports both mono and stereo audio
- Provides detailed logging and error handling

Example usage:
```bash
uv run python shift_pitch.py input.wav output.wav 4.0
```

## Technical Details

### Audio Flow

1. **Recording**: MediaRecorder API captures audio in WebM format
2. **Streaming**: Audio chunks sent to backend via WebSocket
3. **Storage**: Backend saves audio stream to temporary WAV file
4. **Processing**: Python script applies pitch shifting using librosa
5. **Playback**: Processed audio streamed back and played in browser

### WebSocket Protocol

**Client to Server:**
- JSON messages: `{"type": "start_recording"}`, `{"type": "stop_recording"}`
- Binary data: Audio chunks during recording

**Server to Client:**
- JSON messages: Status updates, errors
- Binary data: Processed audio chunks for playback

### Configuration

- **Backend Port**: 8000 (configurable in server.py)
- **Frontend Port**: 5173 (configurable in vite.config.ts)
- **Pitch Shift**: 4 semitones (configurable in shift_pitch.py)
- **Audio Format**: WebM for recording, WAV for processing

## Troubleshooting

### Common Issues

**Microphone Access Denied:**
- Ensure browser permissions are granted for microphone access
- Try refreshing the page and allowing permissions again

**WebSocket Connection Failed:**
- Verify backend server is running on port 8000
- Check that no firewall is blocking the connection

**Audio Processing Errors:**
- Ensure sufficient disk space for temporary files
- Check that all Python audio dependencies are installed

**Build/Install Issues:**
- Verify uv and just are properly installed
- Clear node_modules and .venv, then reinstall: `just clean && just install`

### Logs and Debugging

- Backend logs: Visible in terminal where `just up` was run
- Frontend console: Check browser developer tools
- Temporary files: Located in `temp_audio/` directory

### Performance Notes

- Audio processing typically takes 2-5 seconds
- Large audio files may require more processing time
- Temporary files are automatically cleaned up after processing

## Scaling Considerations

This is currently a single-user development version. For production scaling, consider:

- Horizontal scaling with load balancers
- Dedicated audio processing workers
- Persistent storage for audio files
- User authentication and session management
- Multiple voice effect options
- Real-time streaming instead of file-based processing

## Contributing

1. Follow the existing code style and patterns
2. Run linting before submitting: `just lint`
3. Test both backend and frontend functionality
4. Update documentation for any new features

## License

This project is for demonstration and learning purposes.
