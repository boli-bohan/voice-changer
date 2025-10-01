# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time voice changer with push-to-talk functionality using WebRTC streaming. The architecture consists of three components:

1. **Go API Server** (`api/main.go`) - Signalling service that proxies WebRTC SDP offers/answers between frontend and worker
2. **Python Voice Changer Worker** (`voice_changer.py`) - FastAPI service with WebRTC peer connection handling and real-time pitch shifting using librosa
3. **React TypeScript Frontend** (`frontend/src/`) - Push-to-talk UI with WebRTC client implementation

## Architecture Flow

```
Frontend (React) <--WebRTC SDP--> API Server (Go:8000) <--HTTP--> Worker (Python:8001)
                 <-----WebRTC Audio Stream (P2P)--------------------->
```

- Frontend negotiates WebRTC connection via API server's `/webrtc/offer` endpoint
- API server forwards SDP offers to worker's `/offer` endpoint and returns answers
- Once connected, audio streams directly between frontend and worker via WebRTC
- Worker applies pitch shift using `PitchShiftTrack` (librosa-based MediaStreamTrack wrapper)
- All WebRTC connection lifecycle is managed in `useWebRTC.ts` hook

## Common Commands

### Local Development (No Kubernetes)
```bash
just install    # Install all dependencies (Go, Python via uv, Node)
just up         # Start all services locally (API:8000, Worker:8001, Frontend:5173)
just down       # Stop all services
just status     # Check which services are running
```

### Kubernetes Deployment (Helm)
```bash
just helm         # Build images (unless SKIP_BUILD_IMAGES=1) and install/upgrade the Helm release
SKIP_BUILD_IMAGES=1 just helm  # Reuse previously built images when redeploying
just helm-status  # Inspect deployment, pods, and services
helm uninstall voice-changer  # Remove the Helm release when finished
```

### Individual Services
```bash
just api        # Run API server only (port 8000)
just worker     # Run worker only (port 8001)
cd frontend && npm run dev  # Run frontend only (port 5173)
```

### Building & Testing
```bash
just build      # Build Go API server binary (api/voice-changer-api)
just test       # Run pytest (backend) and vitest (frontend)
just test-browser [--record-output path]  # Playwright integration test

# Test with sample audio
uv run python test_client.py  # Requires services running via 'just up'
```

### Linting & Formatting
```bash
just lint       # Lint all (Go vet/fmt, ruff, skip frontend ESLint)
just format     # Format all (Go fmt, ruff format, prettier)
uv run ruff check .        # Python linting
uv run ruff format .       # Python formatting
cd frontend && npm run format  # Frontend formatting
```

## Key Implementation Details

### WebRTC Connection Setup
- Frontend creates RTCPeerConnection with local microphone stream
- `useWebRTC.ts` hook manages connection lifecycle with cleanup on unmount
- Frontend waits for ICE gathering completion before sending offer to API
- API server proxies offer to worker's FastAPI `/offer` endpoint
- Worker creates peer connection, adds `PitchShiftTrack`, and returns SDP answer
- Active peer connections tracked in `active_peers` set, cleaned up on connection state changes

### Audio Processing Pipeline
1. Frontend captures audio via `getUserMedia()` (browser microphone)
2. Audio sent as WebRTC stream to worker
3. Worker's `PitchShiftTrack.recv()` processes each AudioFrame:
   - Converts multi-channel to mono (averaging)
   - Normalizes to float32 range [-1.0, 1.0]
   - Applies librosa pitch shift (default 4 semitones)
   - Converts back to int16 PCM
   - Returns new AudioFrame with original timing metadata
4. Worker streams processed audio back to frontend via WebRTC
5. Frontend plays audio through HTMLAudioElement

### Testing Infrastructure
- `test_client.py`: Integration test using aiortc to simulate WebRTC client
- `test_browser.py`: Playwright-based browser automation with real microphone simulation
- Frontend tests use Vitest; backend uses pytest with pytest-asyncio
- Tests require services running (`just up`)

## Configuration

### Environment Variables
- `VOICE_WORKER_URL`: Worker endpoint (default: `http://127.0.0.1:8001`)

### Port Assignments
- `8000`: Go API server
- `8001`: Python worker service
- `5173`: Vite dev server (frontend)

### Python Dependencies
Managed via uv (see `pyproject.toml`):
- FastAPI, uvicorn for HTTP/WebSocket
- aiortc, av for WebRTC
- librosa, soundfile, numpy for audio processing
- playwright for browser testing

### Frontend Stack
- React 18 + TypeScript
- Vite build tool
- Vitest for testing
- No state management library (local React state + useWebRTC hook)

## Codebase Conventions

### Python (PEP 8 + Ruff)
- Line length: 100 characters
- Use type annotations for new code
- Ruff rules: E, F, I, N, W, UP (ignore E501)
- Known first-party: `voice_changer`

### TypeScript
- Components: PascalCase (`PushToTalkButton.tsx`)
- Hooks: camelCase with `use` prefix (`useWebRTC.ts`)
- Format with Prettier (see `frontend/.prettierrc`)
- ESLint temporarily disabled (needs config fix)

### Go
- Standard Go formatting (go fmt)
- Use go vet for linting

## Important Files

- `voice_changer.py`: Worker service with WebRTC peer handling and pitch shift track
- `api/main.go`: API server that forwards SDP offers to worker
- `frontend/src/hooks/useWebRTC.ts`: React hook managing WebRTC lifecycle
- `frontend/src/components/PushToTalkButton.tsx`: Push-to-talk UI component
- `test_client.py`: Integration test client using aiortc
- `test_browser.py`: Playwright-based browser automation test
- `Justfile`: Command runner with all common operations

## Modal Deployment

`voice_changer_modal.py` contains Modal.com serverless deployment configuration (experimental/alternate deployment path).
