import { useCallback, useEffect, useState } from 'react'
import { useWebRTC } from '../hooks/useWebRTC'
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
  const [errorMessage, setErrorMessage] = useState('')
  const [isPointerActive, setIsPointerActive] = useState(false)

  const {
    connect,
    disconnect,
    startTalking,
    stopTalking,
    connectionState,
    isTalking,
    isConnected,
  } = useWebRTC({
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

  useEffect(() => {
    if (connectionState !== 'connected') {
      setIsPointerActive(false)
      setAppState('idle')
    }
  }, [connectionState])

  useEffect(() => {
    if (appState === 'error') {
      setIsPointerActive(false)
    }
  }, [appState])

  const handleConnect = useCallback(async () => {
    setErrorMessage('')
    setAppState('idle')
    try {
      await connect()
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to connect to worker'
      setAppState('error')
      setErrorMessage(message)
    }
  }, [connect])

  const handleDisconnect = useCallback(() => {
    disconnect()
    setIsPointerActive(false)
    setAppState('idle')
    setErrorMessage('')
  }, [disconnect])

  const handlePushToTalkStart = useCallback(() => {
    if (!isConnected) {
      setErrorMessage('Connect to the worker before streaming audio.')
      setAppState('error')
      return
    }

    setErrorMessage('')
    setAppState('recording')
    setIsPointerActive(true)
    startTalking()
  }, [isConnected, startTalking])

  const handlePushToTalkStop = useCallback(() => {
    if (!isConnected) return

    setIsPointerActive(false)
    if (isTalking) {
      setAppState('processing')
    }
    stopTalking()
  }, [isConnected, isTalking, stopTalking])

  const handleReset = useCallback(() => {
    setAppState('idle')
    setErrorMessage('')
    setIsPointerActive(false)
  }, [])

  const getButtonText = () => {
    switch (appState) {
      case 'idle':
        return isConnected ? 'Hold to speak' : 'Connect first to begin'
      case 'recording':
        return 'Streaming voice… release to stop'
      case 'processing':
        return 'Processing response…'
      case 'playing':
        return 'Playing transformed audio'
      case 'error':
        return 'Error occurred'
      default:
        return 'Hold to speak'
    }
  }

  const getButtonClass = () => {
    const states = [appState]
    if (isPointerActive) {
      states.push('recording')
    }
    if (appState === 'processing') {
      states.push('processing')
    }
    if (appState === 'playing') {
      states.push('playing')
    }
    return `push-to-talk-button ${states.join(' ')}`
  }

  return (
    <div className="push-to-talk-container">
      <div className="connection-controls">
        <button
          type="button"
          className="connect-button"
          onClick={handleConnect}
          disabled={connectionState === 'connecting' || isConnected}
        >
          {connectionState === 'connecting' ? 'Connecting…' : 'Connect'}
        </button>
        <button
          type="button"
          className="disconnect-button"
          onClick={handleDisconnect}
          disabled={connectionState !== 'connected'}
        >
          Disconnect
        </button>
      </div>

      <button
        className={getButtonClass()}
        onPointerDown={(event) => {
          if (event.button !== undefined && event.button !== 0) return
          handlePushToTalkStart()
        }}
        onPointerUp={handlePushToTalkStop}
        onPointerLeave={() => {
          if (isPointerActive) {
            handlePushToTalkStop()
          }
        }}
        onPointerCancel={handlePushToTalkStop}
        disabled={!isConnected || appState === 'error'}
        type="button"
      >
        <div className="button-content">
          <div className="button-icon">
            {(appState === 'idle' || appState === 'playing') && (
              <div className="microphone-icon">🎤</div>
            )}
            {appState === 'recording' && <div className="recording-indicator" />}
            {appState === 'processing' && <div className="processing-spinner" />}
            {appState === 'error' && <div className="error-icon">❌</div>}
          </div>
          <div className="button-text">{getButtonText()}</div>
        </div>
      </button>

      {errorMessage && (
        <div className="error-message">
          <p>{errorMessage}</p>
          <button onClick={handleReset} className="reset-button" type="button">
            Reset
          </button>
        </div>
      )}

      <div className="instructions">
        <p>
          Use the connect button to negotiate the WebRTC session. While connected, hold the
          push-to-talk control to stream your microphone; release to buffer the worker output and
          play it back.
        </p>
      </div>
    </div>
  )
}

export default PushToTalkButton
