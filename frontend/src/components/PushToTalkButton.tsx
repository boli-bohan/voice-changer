import { useCallback, useEffect, useState } from 'react'
import { useWebRTC } from '../hooks/useWebRTC'
import './PushToTalkButton.css'

interface PushToTalkButtonProps {
  onConnectionStatusChange: (status: 'connecting' | 'connected' | 'disconnected') => void
  onAppStateChange?: (state: AppState) => void
}

export type AppState = 'idle' | 'recording'

const PushToTalkButton: React.FC<PushToTalkButtonProps> = ({
  onConnectionStatusChange,
  onAppStateChange,
}) => {
  const [appState, setAppState] = useState<AppState>('idle')
  const [errorMessage, setErrorMessage] = useState('')
  const [isPointerActive, setIsPointerActive] = useState(false)

  const defaultApiBaseUrl = (() => {
    const envBase = typeof import.meta !== 'undefined' && (import.meta as any).env
      ? (import.meta as any).env.VITE_API_BASE as string | undefined
      : undefined

    if (envBase && envBase.length > 0) {
      return envBase
    }

    if (typeof window !== 'undefined') {
      const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:'
      const host = window.location.hostname
      const port = window.location.port

      if (port === '3000') {
        return `${protocol}//${host}:9000`
      }

      if (port === '5173') {
        return `${protocol}//${host}:8000`
      }

      return `${protocol}//${host}:8000`
    }

    return 'http://localhost:8000'
  })()

  const {
    connect,
    disconnect,
    startTalking,
    stopTalking,
    connectionState,
    isConnected,
  } = useWebRTC({
    apiBaseUrl: defaultApiBaseUrl,
    onStatusChange: onConnectionStatusChange,
    onError: useCallback((error: string) => {
      setAppState('idle')
      setErrorMessage(error)
    }, []),
    onRemoteStarted: undefined,
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

  const handleConnect = useCallback(async () => {
    setErrorMessage('')
    setAppState('idle')
    try {
      await connect()
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to connect to worker'
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
    setAppState('idle')
    stopTalking()
  }, [isConnected, stopTalking])

  const handleReset = useCallback(() => {
    setAppState('idle')
    setErrorMessage('')
    setIsPointerActive(false)
  }, [])

  const getButtonText = () => {
    switch (appState) {
      case 'idle':
        return isConnected ? 'Push-to-talk' : 'Connect first to begin'
      default:
        return 'Push-to-talk'
    }
  }

  const getButtonClass = () => {
    const states = [appState]
    if (isPointerActive) {
      states.push('recording')
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
        disabled={!isConnected}
        type="button"
      >
        <div className="button-content">
          <div className="button-icon">
            {appState === 'idle' && (
              <div className="microphone-icon">🎤</div>
            )}
            {appState === 'recording' && <div className="recording-indicator" />}
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
          push-to-talk control to stream your microphone; release to stop streaming.
        </p>
      </div>
    </div>
  )
}

export default PushToTalkButton
