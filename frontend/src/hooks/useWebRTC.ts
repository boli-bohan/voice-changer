import { useCallback, useEffect, useRef, useState } from 'react'

interface UseWebRTCOptions {
  apiBaseUrl: string
  onStatusChange: (status: 'connecting' | 'connected' | 'disconnected') => void
  onError: (error: string) => void
  onRemoteStarted?: () => void
  onRemoteStopped?: () => void
}

/**
 * React hook that encapsulates WebRTC signalling and audio streaming logic.
 *
 * @param apiBaseUrl - Base URL of the signalling API for SDP exchange.
 * @param onStatusChange - Callback for connection state updates.
 * @param onError - Callback invoked when negotiation or playback fails.
 * @param onRemoteStarted - Optional callback when remote audio begins.
 * @param onRemoteStopped - Optional callback when remote audio ends.
 * @returns Controls and status flags for managing a WebRTC session.
 */
export const useWebRTC = ({
  apiBaseUrl,
  onStatusChange,
  onError,
  onRemoteStarted,
  onRemoteStopped,
}: UseWebRTCOptions) => {
  const peerRef = useRef<RTCPeerConnection | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const remoteAudioRef = useRef<HTMLAudioElement | null>(null)
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected')
  const [isStreaming, setIsStreaming] = useState(false)

  /**
   * Tears down active WebRTC state and optionally emits disconnect callbacks.
   *
   * @param notify - When true, notifies listeners about disconnection.
   */
  const cleanup = useCallback(
    (notify = true) => {
      peerRef.current?.close()
      peerRef.current = null

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
        streamRef.current = null
      }

      if (remoteAudioRef.current) {
        remoteAudioRef.current.pause()
        remoteAudioRef.current.srcObject = null
        remoteAudioRef.current = null
      }

      setIsStreaming(false)
      setConnectionState('disconnected')
      if (notify) {
        onStatusChange('disconnected')
        onRemoteStopped?.()
      }
    },
    [onRemoteStopped, onStatusChange],
  )

  useEffect(() => {
    return () => cleanup(false)
  }, [cleanup])

  /**
   * Waits for ICE candidate gathering to finish before sending an SDP offer.
   *
   * @param pc - Peer connection whose ICE gathering state should be awaited.
   */
  const waitForIceGathering = async (pc: RTCPeerConnection) => {
    if (pc.iceGatheringState === 'complete') return
    await new Promise<void>((resolve) => {
      const checkState = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', checkState)
          resolve()
        }
      }

      pc.addEventListener('icegatheringstatechange', checkState)
    })
  }

  /**
   * Performs offer/answer exchange with the signalling server.
   *
   * @param pc - Peer connection to negotiate.
   */
  const negotiateConnection = useCallback(
    async (pc: RTCPeerConnection) => {
      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
      await waitForIceGathering(pc)

      const response = await fetch(`${apiBaseUrl.replace(/\/$/, '')}/webrtc/offer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription?.sdp,
          type: pc.localDescription?.type,
        }),
      })

      if (!response.ok) {
        throw new Error(`Signalling failed: ${response.status} ${response.statusText}`)
      }

      const answer = await response.json()
      await pc.setRemoteDescription(answer)
    },
    [apiBaseUrl],
  )

  /**
   * Starts microphone capture, negotiates the peer connection, and streams audio.
   */
  const startStreaming = useCallback(async () => {
    if (peerRef.current) return

    onStatusChange('connecting')
    setConnectionState('connecting')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      streamRef.current = stream

      const pc = new RTCPeerConnection()
      peerRef.current = pc

      stream.getTracks().forEach((track) => pc.addTrack(track, stream))

      pc.ontrack = (event) => {
        const [remoteStream] = event.streams
        if (!remoteStream) return

        let element = remoteAudioRef.current
        if (!element) {
          element = new Audio()
          element.autoplay = true
          remoteAudioRef.current = element
        }

        element.srcObject = remoteStream
        element
          .play()
          .then(() => {
            onRemoteStarted?.()
          })
          .catch((error) => {
            console.warn('Failed to autoplay remote audio', error)
            onError('Unable to start remote audio playback')
          })

        remoteStream.getAudioTracks().forEach((track) => {
          track.onended = () => {
            cleanup()
          }
        })
      }

      pc.onconnectionstatechange = () => {
        const state = pc.connectionState
        if (state === 'connected') {
          setConnectionState('connected')
          onStatusChange('connected')
        } else if (state === 'disconnected' || state === 'failed' || state === 'closed') {
          cleanup()
        }
      }

      await negotiateConnection(pc)

      setIsStreaming(true)
    } catch (error) {
      console.error('Failed to start WebRTC session', error)
      cleanup()
      onError(error instanceof Error ? error.message : 'Failed to start WebRTC session')
    }
  }, [cleanup, negotiateConnection, onError, onRemoteStarted, onStatusChange])

  /**
   * Stops streaming and cleans up any active peer connection.
   */
  const stopStreaming = useCallback(() => {
    cleanup()
  }, [cleanup])

  return {
    startStreaming,
    stopStreaming,
    connectionState,
    isStreaming,
  }
}
