# Voice Changer Implementation Plan

## Overview
Build a push-to-talk voice changer app with React frontend and FastAPI backend that processes audio files with pitch shifting.

## Architecture
- **Frontend**: React TypeScript with Vite, WebSocket client
- **Backend**: FastAPI with WebSocket, file-based audio processing
- **Audio Flow**: Record → Stream to backend → Save to disk → Process with script → Stream back

---

## Backend Implementation

### Phase 1: Core Backend Setup
- [ ] Create `pyproject.toml` with dependencies
  - FastAPI, uvicorn, websockets
  - Audio processing: librosa, soundfile, numpy, scipy
  - Development: ruff for linting
- [ ] Create `server.py` FastAPI application
  - Basic FastAPI app with health check endpoint
  - WebSocket endpoint for audio streaming
- [ ] Create `shift_pitch.py` audio processing script
  - Accept input/output file paths as arguments
  - Load audio file, apply pitch shift, save result
  - Use librosa for pitch shifting (pitch_shift function)

### Phase 2: Audio Processing Pipeline
- [ ] Implement WebSocket audio reception
  - Accept binary audio data from frontend
  - Stream data to temporary WAV file on disk
  - Handle start/stop recording signals
- [ ] Implement file processing workflow
  - Call `shift_pitch.py` script when recording stops
  - Handle subprocess execution and error handling
  - Clean up temporary files after processing
- [ ] Implement audio playback streaming
  - Stream processed audio file back over WebSocket
  - Send audio data in chunks for smooth playback
  - Handle file reading and WebSocket transmission

### Phase 3: Error Handling & Optimization
- [ ] Add comprehensive error handling
  - WebSocket connection errors
  - Audio processing failures
  - File I/O errors and cleanup
- [ ] Add logging and monitoring
  - Audio processing timing
  - File size limits and validation
  - WebSocket connection status
- [ ] Optimize performance
  - Streaming buffer sizes
  - Temporary file management
  - Memory usage optimization

---

## Frontend Implementation

### Phase 1: React App Setup
- [ ] Create `frontend/` directory structure
- [ ] Set up `package.json` with dependencies
  - React 18, TypeScript, Vite
  - Audio recording and WebSocket libraries
- [ ] Configure `tsconfig.json` and `vite.config.ts`
- [ ] Create basic App component structure

### Phase 2: Audio Recording Component
- [ ] Create `PushToTalkButton` component
  - Visual states: idle, recording, processing, playing
  - Handle mouse down/up events for push-to-talk
- [ ] Implement `useAudioRecorder` hook
  - MediaRecorder API integration
  - Handle audio data chunks and streaming
  - Start/stop recording functionality
- [ ] Create WebSocket connection hook
  - Connect to backend WebSocket endpoint
  - Send audio data during recording
  - Receive processed audio data

### Phase 3: Audio Playback & UI
- [ ] Implement audio playback functionality
  - Play received processed audio automatically
  - Handle audio blob creation and playback
- [ ] Add visual feedback components
  - Recording indicator (red dot, pulsing animation)
  - Processing status (loading spinner)
  - Simple waveform visualization (optional)
- [ ] Polish UI/UX
  - Responsive design
  - Clear instructions for push-to-talk
  - Error states and user feedback

---

## Development Tools & Configuration

### Project Setup
- [ ] Create `Justfile` with development commands
  - `install`: Install backend and frontend dependencies
  - `up`: Start both backend and frontend concurrently
  - `down`: Stop all running services
  - `lint`: Run linting on both projects
- [ ] Set up proper directory structure
  ```
  voice-changer/
  ├── Justfile
  ├── pyproject.toml
  ├── server.py
  ├── shift_pitch.py
  ├── plan.md
  └── frontend/
      ├── package.json
      ├── tsconfig.json
      ├── vite.config.ts
      └── src/
  ```

### Testing & Quality
- [ ] Add basic error handling and validation
- [ ] Test audio recording and playback in different browsers
- [ ] Verify pitch shifting works with various audio inputs
- [ ] Test WebSocket connection stability

---

## Future Enhancements (Post-MVP)
- [ ] Add different voice effects beyond pitch shifting
- [ ] Implement real-time streaming instead of file-based processing
- [ ] Add user authentication and session management
- [ ] Scale to handle multiple concurrent users
- [ ] Add audio format support beyond WAV
- [ ] Implement client-side audio visualization

---

## Technical Notes

### Audio Processing Details
- **Input Format**: WAV files from MediaRecorder API
- **Processing**: Pitch shifting using librosa's `pitch_shift` function
- **Output Format**: WAV files streamed back as binary data
- **File Management**: Temporary files cleaned up after processing

### WebSocket Protocol
- **Upload**: Binary audio data chunks during recording
- **Control**: JSON messages for start/stop recording
- **Download**: Binary processed audio data chunks
- **Status**: JSON status updates (processing, ready, error)

### Performance Considerations
- Streaming chunk sizes for smooth real-time experience
- Temporary file cleanup to prevent disk space issues
- Memory management for large audio files
- Concurrent user handling (future scaling)