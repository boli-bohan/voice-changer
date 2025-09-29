import { useState, useCallback, useEffect } from 'react'
import { useWebSocket } from '../hooks/useWebSocket'
import { useAudioRecorder } from '../hooks/useAudioRecorder'
import './PushToTalkButton.css'

interface PushToTalkButtonProps {
  onConnectionStatusChange: (status: 'connecting' | 'connected' | 'disconnected') => void
  onAppStateChange?: (state: AppState) => void
}

export type AppState = 'idle' | 'recording' | 'processing' | 'playing' | 'error'

const PushToTalkButton: React.FC<PushToTalkButtonProps> = ({
  onConnectionStatusChange,
  onAppStateChange,
}) => {
  const [appState, setAppState] = useState<AppState>('idle')
  const [errorMessage, setErrorMessage] = useState<string>('')

  const {
    sendAudioData,
    connect,
    disconnect,
    connectionState,
    startRecording: startWebSocketRecording,
    stopRecording: stopWebSocketRecording,
  } = useWebSocket({
    url: 'ws://localhost:8000/ws',
    onStatusChange: onConnectionStatusChange,
    onError: useCallback((error: string) => {
      setAppState('error')
      setErrorMessage(error)
    }, []),
    onPlaybackStarted: useCallback(() => {
      setAppState('playing')
    }, []),
    onPlaybackComplete: useCallback(() => {
      setAppState('idle')
    }, []),
  })

  const {
    startRecording: startAudioRecording,
    stopRecording: stopAudioRecording,
    isRecording,
  } = useAudioRecorder({
    onAudioData: sendAudioData,
    onError: useCallback((error: string) => {
      setAppState('error')
      setErrorMessage(error)
    }, []),
  })

  useEffect(() => {
    onAppStateChange?.(appState)
  }, [appState, onAppStateChange])

  const handleMouseDown = useCallback(async () => {
    if (appState !== 'idle') return

    try {
      if (connectionState !== 'connected') {
        await connect()
      }
      setErrorMessage('')
      setAppState('recording')

      // Start WebSocket recording session
      startWebSocketRecording()

      // Start audio recording
      await startAudioRecording()
    } catch (error) {
      setAppState('error')
      setErrorMessage(error instanceof Error ? error.message : 'Failed to start recording')
    }
  }, [appState, connectionState, connect, startWebSocketRecording, startAudioRecording])

  const handleMouseUp = useCallback(() => {
    if (appState === 'recording') {
      setAppState('processing')

      // Stop audio recording
      stopAudioRecording()

      // Stop WebSocket recording session and trigger processing
      stopWebSocketRecording()
    }
  }, [appState, stopAudioRecording, stopWebSocketRecording])

  const handleMouseLeave = useCallback(() => {
    if (appState === 'recording') {
      setAppState('processing')

      // Stop audio recording
      stopAudioRecording()

      // Stop WebSocket recording session and trigger processing
      stopWebSocketRecording()
    }
  }, [appState, stopAudioRecording, stopWebSocketRecording])

  const handleReset = useCallback(() => {
    setAppState('idle')
    setErrorMessage('')
    disconnect()
  }, [disconnect])

  const getButtonText = () => {
    switch (appState) {
      case 'idle':
        return 'Press & Hold to Record'
      case 'recording':
        return 'Recording... (Release to Process)'
      case 'processing':
        return 'Processing Audio...'
      case 'playing':
        return 'Playing Processed Audio...'
      case 'error':
        return 'Error Occurred'
      default:
        return 'Press & Hold to Record'
    }
  }

  const getButtonClass = () => {
    return `push-to-talk-button ${appState} ${isRecording ? 'recording' : ''}`
  }

  return (
    <div className="push-to-talk-container">
      <button
        className={getButtonClass()}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        disabled={appState === 'processing' || appState === 'playing'}
        type="button"
      >
        <div className="button-content">
          <div className="button-icon">
            {appState === 'recording' && <div className="recording-indicator" />}
            {appState === 'processing' && <div className="processing-spinner" />}
            {(appState === 'idle' || appState === 'playing') && (
              <div className="microphone-icon">🎤</div>
            )}
            {appState === 'error' && <div className="error-icon">❌</div>}
          </div>
          <div className="button-text">{getButtonText()}</div>
        </div>
      </button>

      {errorMessage && (
        <div className="error-message">
          <p>{errorMessage}</p>
          <button onClick={handleReset} className="reset-button">
            Reset
          </button>
        </div>
      )}

      <div className="instructions">
        <p>
          Hold down the button to record your voice, then release to hear it with a pitch shift!
        </p>
      </div>
    </div>
  )
}

export default PushToTalkButton
