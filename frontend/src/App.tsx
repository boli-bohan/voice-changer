import { useCallback, useState } from 'react'
import PushToTalkButton, { type AppState } from './components/PushToTalkButton'
import AudioWaveform, { type AudioWaveformMode } from './components/AudioWaveform'
import './App.css'

/**
 * Root application component that wires together the push-to-talk workflow.
 *
 * Manages WebRTC connection state and maps PushToTalkButton app states to
 * waveform display modes for the AudioWaveform component.
 *
 * @returns The rendered Voice Changer user interface.
 */
function App() {
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected')
  const [waveformMode, setWaveformMode] = useState<AudioWaveformMode>('inactive')

  const handleAppStateChange = useCallback((state: AppState) => {
    switch (state) {
      case 'recording':
        setWaveformMode('input')
        break
      case 'playing':
        setWaveformMode('output')
        break
      default:
        setWaveformMode('inactive')
        break
    }
  }, [])

  const waveformCaptions: Record<AudioWaveformMode, string> = {
    inactive: 'Awaiting audio',
    input: 'Capturing voice…',
    output: 'Playing response…'
  }

  return (
    <div className="App">
      <header className="app-header">
        <h1>Voice Changer</h1>
        <p>Press and hold the button to record, release to process and play back</p>
        <div className={`connection-status ${connectionStatus}`}>
          Status: {connectionStatus}
        </div>
      </header>

      <main className="app-main">
        <div className="interaction-panel">
          <AudioWaveform mode={waveformMode} caption={waveformCaptions[waveformMode]} />
          <PushToTalkButton
            onConnectionStatusChange={setConnectionStatus}
            onAppStateChange={handleAppStateChange}
          />
        </div>
      </main>

      <footer className="app-footer">
        <small>
          Signalling via <span id="backend-url">http://localhost:8000/webrtc/offer</span> · Direct WebRTC to
          worker on <span id="worker-url">http://localhost:8001</span>
        </small>
      </footer>
    </div>
  )
}

export default App
