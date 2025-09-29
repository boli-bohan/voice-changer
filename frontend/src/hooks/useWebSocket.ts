import { useCallback, useEffect, useRef, useState } from 'react'

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected'

interface UseWebSocketProps {
  url: string
  onStatusChange: (status: ConnectionStatus) => void
  onError: (error: string) => void
  onPlaybackComplete?: () => void
  onPlaybackStarted?: () => void
}

interface ServerMessage {
  type: string
  message?: string
  sdp?: string
  sdp_type?: string
  candidate?: RTCIceCandidateInit
}

const waitForIceGatheringComplete = async (pc: RTCPeerConnection): Promise<void> => {
  if (pc.iceGatheringState === 'complete') {
    return
  }

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

export const useWebSocket = ({
  url,
  onStatusChange,
  onError,
  onPlaybackComplete,
  onPlaybackStarted,
}: UseWebSocketProps) => {
  const wsRef = useRef<WebSocket | null>(null)
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null)
  const remoteStreamRef = useRef<MediaStream | null>(null)
  const isInitiatingRef = useRef(false)
  const playbackStartedRef = useRef(false)

  const [connectionState, setConnectionState] = useState<ConnectionStatus>('disconnected')
  const [streamingSupported, setStreamingSupported] = useState(false)
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null)

  const cleanupPeerConnection = useCallback(() => {
    const peerConnection = peerConnectionRef.current
    if (peerConnection) {
      peerConnection.ontrack = null
      peerConnection.onicecandidate = null
      peerConnection.onconnectionstatechange = null
      peerConnection.close()
    }

    peerConnectionRef.current = null
    isInitiatingRef.current = false

    if (remoteStreamRef.current) {
      remoteStreamRef.current.getTracks().forEach((track) => track.stop())
    }

    remoteStreamRef.current = null
    setRemoteStream(null)
    playbackStartedRef.current = false
    setStreamingSupported(false)
  }, [])

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    cleanupPeerConnection()
    setConnectionState('disconnected')
    onStatusChange('disconnected')
  }, [cleanupPeerConnection, onStatusChange])

  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  const handleRemoteTrack = useCallback(
    (event: RTCTrackEvent) => {
      const [firstStream] = event.streams

      if (firstStream) {
        remoteStreamRef.current = firstStream
        setRemoteStream(firstStream)
      } else {
        if (!remoteStreamRef.current) {
          remoteStreamRef.current = new MediaStream()
        }
        remoteStreamRef.current.addTrack(event.track)
        setRemoteStream(remoteStreamRef.current.clone())
      }

      const track = event.track
      track.onunmute = () => {
        if (!playbackStartedRef.current) {
          playbackStartedRef.current = true
          onPlaybackStarted?.()
        }
      }

      track.onended = () => {
        playbackStartedRef.current = false
        onPlaybackComplete?.()
      }
    },
    [onPlaybackComplete, onPlaybackStarted]
  )

  const initiateWebRTC = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return
    }

    if (peerConnectionRef.current || isInitiatingRef.current) {
      return
    }

    isInitiatingRef.current = true

    try {
      const peerConnection = new RTCPeerConnection()
      peerConnectionRef.current = peerConnection

      peerConnection.ontrack = handleRemoteTrack
      peerConnection.onicecandidate = (event) => {
        if (event.candidate && wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({
              type: 'webrtc_ice_candidate',
              candidate: event.candidate.toJSON(),
            })
          )
        }
      }

      peerConnection.onconnectionstatechange = () => {
        if (peerConnection.connectionState === 'failed') {
          onError('WebRTC connection failed')
          cleanupPeerConnection()
        }
      }

      const offer = await peerConnection.createOffer({ offerToReceiveAudio: true })
      await peerConnection.setLocalDescription(offer)
      await waitForIceGatheringComplete(peerConnection)

      const localDescription = peerConnection.localDescription
      if (!localDescription || wsRef.current?.readyState !== WebSocket.OPEN) {
        throw new Error('Missing local description for WebRTC offer')
      }

      wsRef.current.send(
        JSON.stringify({
          type: 'webrtc_offer',
          sdp: localDescription.sdp,
          sdp_type: localDescription.type,
        })
      )

      setStreamingSupported(true)
    } catch (error) {
      cleanupPeerConnection()
      if (error instanceof Error) {
        onError(error.message)
      } else {
        onError('Failed to establish WebRTC connection')
      }
    } finally {
      isInitiatingRef.current = false
    }
  }, [cleanupPeerConnection, handleRemoteTrack, onError])

  const handleServerMessage = useCallback(
    async (data: string) => {
      try {
        const message: ServerMessage = JSON.parse(data)

        switch (message.type) {
          case 'webrtc_answer': {
            const peerConnection = peerConnectionRef.current
            if (!peerConnection) {
              onError('Received WebRTC answer without an active connection')
              return
            }

            const description = new RTCSessionDescription({
              type: (message.sdp_type as RTCSdpType) ?? 'answer',
              sdp: message.sdp ?? '',
            })

            await peerConnection.setRemoteDescription(description)
            break
          }

          case 'webrtc_ice_candidate':
            if (message.candidate && peerConnectionRef.current) {
              try {
                await peerConnectionRef.current.addIceCandidate(message.candidate)
              } catch (error) {
                onError('Failed to add remote ICE candidate')
              }
            }
            break

          case 'streaming_completed':
          case 'done':
            playbackStartedRef.current = false
            onPlaybackComplete?.()
            break

          case 'error':
            onError(message.message || 'Unknown server error')
            break

          default:
            break
        }
      } catch (error) {
        onError('Failed to process server message')
      }
    },
    [onError, onPlaybackComplete]
  )

  const connect = useCallback((): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        resolve()
        return
      }

      setConnectionState('connecting')
      onStatusChange('connecting')

      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        setConnectionState('connected')
        onStatusChange('connected')
        void initiateWebRTC()
        resolve()
      }

      ws.onmessage = async (event) => {
        if (typeof event.data === 'string') {
          await handleServerMessage(event.data)
        }
      }

      ws.onerror = () => {
        setConnectionState('disconnected')
        onStatusChange('disconnected')
        onError('WebSocket connection failed')
        cleanupPeerConnection()
        reject(new Error('WebSocket connection failed'))
      }

      ws.onclose = () => {
        setConnectionState('disconnected')
        onStatusChange('disconnected')
        cleanupPeerConnection()
      }
    })
  }, [cleanupPeerConnection, handleServerMessage, initiateWebRTC, onError, onStatusChange, url])

  const sendMessage = useCallback(
    (message: unknown) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(message))
      } else {
        onError('WebSocket not connected')
      }
    },
    [onError]
  )

  const sendAudioData = useCallback(
    (audioBlob: Blob) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(audioBlob)
      } else {
        onError('WebSocket not connected')
      }
    },
    [onError]
  )

  const startRecording = useCallback(() => {
    sendMessage({ type: 'start_recording' })
  }, [sendMessage])

  const stopRecording = useCallback(() => {
    sendMessage({ type: 'stop_recording' })
  }, [sendMessage])

  return {
    connect,
    disconnect,
    sendMessage,
    sendAudioData,
    startRecording,
    stopRecording,
    connectionState,
    streamingSupported,
    remoteStream,
  }
}
