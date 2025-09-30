import { useCallback, useEffect, useState } from 'react'
import { useWebRTC } from '../hooks/useWebRTC'
import './PushToTalkButton.css'

interface PushToTalkButtonProps {
  onConnectionStatusChange: (status: 'connecting' | 'connected' | 'disconnected') => void
  onAppStateChange?: (state: AppState) => void
}

export type AppState = 'idle' | 'recording' | 'playing' | 'error'

const PushToTalkButton: React.FC<PushToTalkButtonProps> = ({
  onConnectionStatusChange,
  onAppStateChange,
}) => {
  const [appState, setAppState] = useState<AppState>('idle')
  const [errorMessage, setErrorMessage] = useState<string>('')

  const { startStreaming, stopStreaming, connectionState, isStreaming } = useWebRTC({
    apiBaseUrl: 'http://localhost:8000',
    onStatusChange: onConnectionStatusChange,
    onError: useCallback((error: string) => {
      setAppState('error')
      setErrorMessage(error)
    }, []),
    onRemoteStarted: useCallback(() => {
      setAppState('playing')
    }, []),
    onRemoteStopped: useCallback(() => {
      setAppState('idle')
    }, []),
  })

  useEffect(() => {
    onAppStateChange?.(appState)
  }, [appState, onAppStateChange])

  const handleMouseDown = useCallback(async () => {
    if (appState !== 'idle') return

    try {
      setErrorMessage('')
      setAppState('recording')
      await startStreaming()
    } catch (error) {
      setAppState('error')
      setErrorMessage(error instanceof Error ? error.message : 'Failed to start streaming')
    }
  }, [appState, startStreaming])

  const stopSession = useCallback(() => {
    stopStreaming()
  }, [stopStreaming])

  const handleMouseUp = useCallback(() => {
    if (isStreaming || appState === 'playing' || appState === 'recording') {
      stopSession()
    }
  }, [appState, isStreaming, stopSession])

  const handleMouseLeave = useCallback(() => {
    if (isStreaming || appState === 'recording') {
      stopSession()
    }
  }, [appState, isStreaming, stopSession])

  const handleReset = useCallback(() => {
    setAppState('idle')
    setErrorMessage('')
    stopStreaming()
  }, [stopStreaming])

  const getButtonText = () => {
    switch (appState) {
      case 'idle':
        return connectionState === 'connected' ? 'Press & Hold to Speak' : 'Press to Connect'
      case 'recording':
        return 'Streaming voice… release to stop'
      case 'playing':
        return 'Listening to transformed audio…'
      case 'error':
        return 'Error occurred'
      default:
        return 'Press & Hold to Speak'
    }
  }

  const getButtonClass = () => {
    const states = [appState]
    if (isStreaming) {
      states.push('recording')
    }
    return `push-to-talk-button ${states.join(' ')}`
  }

  return (
    <div className="push-to-talk-container">
      <button
        className={getButtonClass()}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        disabled={appState === 'error'}
        type="button"
      >
        <div className="button-content">
          <div className="button-icon">
            {(appState === 'idle' || appState === 'playing') && <div className="microphone-icon">🎤</div>}
            {appState === 'recording' && <div className="recording-indicator" />}
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
          Hold down the button to stream your voice to the worker. Release to stop the WebRTC session and
          start a new one when you are ready.
        </p>
      </div>
    </div>
  )
}

export default PushToTalkButton
