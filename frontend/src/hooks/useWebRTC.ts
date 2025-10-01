import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'

interface UseWebRTCOptions {
  apiBaseUrl: string
  onStatusChange: (status: 'connecting' | 'connected' | 'disconnected') => void
  onError: (error: string) => void
  onRemoteStarted?: () => void
  onRemoteStopped?: () => void
  remoteAudioRef?: RefObject<HTMLAudioElement>
}

interface UseWebRTCControls {
  connect: () => Promise<void>
  disconnect: () => void
  startTalking: () => void
  stopTalking: () => void
  connectionState: 'connecting' | 'connected' | 'disconnected'
  isTalking: boolean
  isConnected: boolean
}

export const useWebRTC = ({
  apiBaseUrl,
  onStatusChange,
  onError,
  onRemoteStarted,
  onRemoteStopped,
  remoteAudioRef,
}: UseWebRTCOptions): UseWebRTCControls => {
  const peerRef = useRef<RTCPeerConnection | null>(null)
  const localStreamRef = useRef<MediaStream | null>(null)
  const localTracksRef = useRef<MediaStreamTrack[]>([])
  const remoteStreamRef = useRef<MediaStream | null>(null)
  const remoteAudioElementRef = useRef<HTMLAudioElement | null>(null)

  const [connectionState, setConnectionState] = useState<
    'connecting' | 'connected' | 'disconnected'
  >('disconnected')
  const [isTalking, setIsTalking] = useState(false)

  const updateStatus = useCallback(
    (status: 'connecting' | 'connected' | 'disconnected') => {
      setConnectionState(status)
      onStatusChange(status)
    },
    [onStatusChange]
  )

  const cleanup = useCallback(
    (notify = true) => {
      peerRef.current?.close()
      peerRef.current = null

      localTracksRef.current.forEach((track) => {
        track.stop()
      })
      localTracksRef.current = []

      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach((track) => track.stop())
        localStreamRef.current = null
      }

      if (remoteStreamRef.current) {
        remoteStreamRef.current.getTracks().forEach((track) => track.stop())
        remoteStreamRef.current = null
      }

      if (remoteAudioElementRef.current) {
        remoteAudioElementRef.current.srcObject = null
      }

      setIsTalking(false)
      updateStatus('disconnected')
      if (notify) {
        onRemoteStopped?.()
      }
    },
    [onRemoteStopped, updateStatus]
  )

  useEffect(() => {
    return () => {
      cleanup(false)
    }
  }, [cleanup])

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

  const negotiateConnection = useCallback(
    async (pc: RTCPeerConnection) => {
      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
      await waitForIceGathering(pc)

      const endpoint = `${apiBaseUrl.replace(/\/$/, '')}/webrtc/offer`
      const response = await fetch(endpoint, {
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
    [apiBaseUrl]
  )

  const ensureRemoteAudioElement = useCallback(() => {
    if (remoteAudioRef?.current) {
      remoteAudioElementRef.current = remoteAudioRef.current
      return remoteAudioRef.current
    }

    if (!remoteAudioElementRef.current) {
      const audio = new Audio()
      audio.autoplay = true
      audio.controls = false
      audio.crossOrigin = 'anonymous'
      remoteAudioElementRef.current = audio
    }
    return remoteAudioElementRef.current
  }, [remoteAudioRef])

  const connect = useCallback(async () => {
    if (peerRef.current) {
      return
    }

    updateStatus('connecting')

    try {
      const localStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      localStreamRef.current = localStream
      localTracksRef.current = localStream.getAudioTracks()
      localTracksRef.current.forEach((track) => {
        track.enabled = false
      })

      const pc = new RTCPeerConnection()
      peerRef.current = pc

      localStream.getTracks().forEach((track) => {
        pc.addTrack(track, localStream)
      })

      pc.ontrack = (event) => {
        const [stream] = event.streams
        remoteStreamRef.current = stream ?? new MediaStream([event.track])
        const audioEl = ensureRemoteAudioElement()
        if (audioEl.srcObject !== remoteStreamRef.current) {
          audioEl.srcObject = remoteStreamRef.current
        }
        audioEl
          .play()
          .then(() => onRemoteStarted?.())
          .catch((error) => {
            console.warn('Failed to start remote playback', error)
            onError('Unable to start remote audio playback')
          })
      }

      pc.onconnectionstatechange = () => {
        const state = pc.connectionState
        if (state === 'connected') {
          updateStatus('connected')
        }
        if (state === 'disconnected' || state === 'failed' || state === 'closed') {
          cleanup()
        }
      }

      await negotiateConnection(pc)
      updateStatus('connected')
    } catch (error) {
      console.error('Failed to establish WebRTC session', error)
      cleanup(false)
      onError('Failed to establish WebRTC session')
      throw error
    }
  }, [cleanup, ensureRemoteAudioElement, negotiateConnection, onError, onRemoteStarted, updateStatus])

  const disconnect = useCallback(() => {
    cleanup()
  }, [cleanup])

  const setTalkingState = useCallback((enabled: boolean) => {
    localTracksRef.current.forEach((track) => {
      track.enabled = enabled
    })
    setIsTalking(enabled)
  }, [])

  const startTalking = useCallback(() => {
    if (!peerRef.current) {
      void connect().catch(() => undefined)
    }
    setTalkingState(true)
  }, [connect, setTalkingState])

  const stopTalking = useCallback(() => {
    setTalkingState(false)
  }, [setTalkingState])

  return {
    connect,
    disconnect,
    startTalking,
    stopTalking,
    connectionState,
    isTalking,
    isConnected: connectionState === 'connected',
  }
}
