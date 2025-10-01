import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

interface UseWebRTCOptions {
  apiBaseUrl: string
  onStatusChange: (status: 'connecting' | 'connected' | 'disconnected') => void
  onError: (error: string) => void
  onRemoteStarted?: () => void
  onRemoteStopped?: () => void
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

const REMOTE_FLUSH_DELAY_MS = 250

const selectRecorderOptions = () => {
  if (typeof window === 'undefined' || typeof MediaRecorder === 'undefined') {
    return null
  }

  const preferred = ['audio/webm;codecs=opus', 'audio/webm']
  for (const mimeType of preferred) {
    if (MediaRecorder.isTypeSupported(mimeType)) {
      return { mimeType }
    }
  }

  return {}
}

export const useWebRTC = ({
  apiBaseUrl,
  onStatusChange,
  onError,
  onRemoteStarted,
  onRemoteStopped,
}: UseWebRTCOptions): UseWebRTCControls => {
  const peerRef = useRef<RTCPeerConnection | null>(null)
  const localStreamRef = useRef<MediaStream | null>(null)
  const localTracksRef = useRef<MediaStreamTrack[]>([])
  const remoteStreamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const recordedChunksRef = useRef<BlobPart[]>([])
  const recordedMimeRef = useRef<string>('audio/webm')
  const playbackAudioRef = useRef<HTMLAudioElement | null>(null)
  const playbackUrlRef = useRef<string | null>(null)
  const wantRecordingRef = useRef(false)
  const playbackRequestedRef = useRef(false)
  const recorderOptions = useMemo(selectRecorderOptions, [])

  const [connectionState, setConnectionState] = useState<
    'connecting' | 'connected' | 'disconnected'
  >('disconnected')
  const [isTalking, setIsTalking] = useState(false)

  const releasePlaybackUrl = useCallback(() => {
    if (playbackUrlRef.current && typeof URL.revokeObjectURL === 'function') {
      URL.revokeObjectURL(playbackUrlRef.current)
    }
    playbackUrlRef.current = null
  }, [])

  const finalizePlayback = useCallback(() => {
    if (!playbackRequestedRef.current) {
      recordedChunksRef.current = []
      return
    }

    const chunks = recordedChunksRef.current
    recordedChunksRef.current = []
    playbackRequestedRef.current = false

    if (!chunks.length) {
      onRemoteStopped?.()
      return
    }

    releasePlaybackUrl()

    const blob = new Blob(chunks, { type: recordedMimeRef.current })
    if (typeof URL.createObjectURL !== 'function') {
      onError('This environment cannot create audio buffers for playback')
      onRemoteStopped?.()
      return
    }
    if (typeof Audio === 'undefined') {
      onError('Audio playback is not supported in this environment')
      onRemoteStopped?.()
      return
    }

    const audioEl = playbackAudioRef.current ?? new Audio()
    audioEl.autoplay = false
    audioEl.controls = false
    audioEl.crossOrigin = 'anonymous'
    audioEl.currentTime = 0

    const objectUrl = URL.createObjectURL(blob)
    playbackUrlRef.current = objectUrl
    audioEl.src = objectUrl

    audioEl.onended = () => {
      audioEl.onended = null
      releasePlaybackUrl()
      onRemoteStopped?.()
    }

    audioEl.onerror = () => {
      audioEl.onerror = null
      releasePlaybackUrl()
      onError('Unable to play buffered audio')
      onRemoteStopped?.()
    }

    playbackAudioRef.current = audioEl

    audioEl
      .play()
      .then(() => {
        onRemoteStarted?.()
      })
      .catch((error) => {
        console.warn('Buffered playback failed', error)
        releasePlaybackUrl()
        onError('Unable to play buffered audio')
        onRemoteStopped?.()
      })
  }, [onError, onRemoteStarted, onRemoteStopped, releasePlaybackUrl])

  const stopRecorder = useCallback((options?: { allowPlayback: boolean }) => {
    const recorder = recorderRef.current
    if (!recorder) return

    if (options?.allowPlayback === false) {
      playbackRequestedRef.current = false
    }

    if (recorder.state !== 'inactive') {
      try {
        recorder.stop()
      } catch (error) {
        console.warn('Failed to stop media recorder cleanly', error)
      }
    }

    recorderRef.current = null
  }, [])

  const cleanupPlayback = useCallback(() => {
    if (playbackAudioRef.current) {
      playbackAudioRef.current.pause()
      playbackAudioRef.current.removeAttribute('src')
      playbackAudioRef.current.load()
    }
    releasePlaybackUrl()
  }, [releasePlaybackUrl])

  const cleanup = useCallback(
    (notify = true) => {
      wantRecordingRef.current = false
      playbackRequestedRef.current = false

      stopRecorder({ allowPlayback: false })
      recordedChunksRef.current = []
      cleanupPlayback()

      peerRef.current?.close()
      peerRef.current = null

      localTracksRef.current.forEach((track) => {
        track.enabled = false
        track.stop()
      })
      localTracksRef.current = []

      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach((track) => {
          track.enabled = false
          track.stop()
        })
        localStreamRef.current = null
      }

      remoteStreamRef.current = null

      setIsTalking(false)
      setConnectionState('disconnected')
      if (notify) {
        onStatusChange('disconnected')
        onRemoteStopped?.()
      }
    },
    [cleanupPlayback, onRemoteStopped, onStatusChange, stopRecorder]
  )

  useEffect(() => {
    return () => cleanup(false)
  }, [cleanup])

  const waitForIceGathering = async (pc: RTCPeerConnection) => {
    if (pc.iceGatheringState === 'complete') return
    await new Promise<void>((resolve) => {
      const handleStateChange = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', handleStateChange)
          resolve()
        }
      }
      pc.addEventListener('icegatheringstatechange', handleStateChange)
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

  const startRecorder = useCallback(() => {
    if (!wantRecordingRef.current) return
    if (!remoteStreamRef.current) return
    if (!recorderOptions) {
      onError('MediaRecorder is not supported; buffered playback is unavailable')
      return
    }

    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      return
    }

    try {
      const recorder = new MediaRecorder(remoteStreamRef.current, recorderOptions)
      recorderRef.current = recorder
      recordedMimeRef.current =
        recorder.mimeType || (recorderOptions.mimeType as string) || 'audio/webm'
      recordedChunksRef.current = []

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunksRef.current.push(event.data)
        }
      }

      recorder.onstop = () => {
        recorder.ondataavailable = null
        recorder.onstop = null
        recorderRef.current = null
        finalizePlayback()
      }

      recorder.start()
    } catch (error) {
      console.error('Failed to start media recorder', error)
      onError('Unable to buffer remote audio in this browser')
    }
  }, [finalizePlayback, onError, recorderOptions])

  const connect = useCallback(async () => {
    if (peerRef.current) return

    onStatusChange('connecting')
    setConnectionState('connecting')

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
        const [remoteStream] = event.streams
        if (remoteStream) {
          remoteStreamRef.current = remoteStream
        } else {
          remoteStreamRef.current = new MediaStream([event.track])
        }

        event.track.onended = () => {
          if (!wantRecordingRef.current) {
            return
          }
          stopRecorder()
        }

        if (wantRecordingRef.current) {
          startRecorder()
        }
      }

      pc.onconnectionstatechange = () => {
        const state = pc.connectionState
        if (state === 'connected') {
          setConnectionState('connected')
          onStatusChange('connected')
        } else if (state === 'failed' || state === 'disconnected' || state === 'closed') {
          cleanup()
        }
      }

      await negotiateConnection(pc)
    } catch (error) {
      console.error('Failed to establish WebRTC connection', error)
      cleanup()
      onError(error instanceof Error ? error.message : 'Unable to establish WebRTC connection')
    }
  }, [cleanup, negotiateConnection, onError, onStatusChange, startRecorder, stopRecorder])

  const disconnect = useCallback(() => {
    cleanup()
  }, [cleanup])

  const startTalking = useCallback(() => {
    if (!peerRef.current || connectionState !== 'connected') {
      onError('Connect to the worker before speaking')
      return
    }

    wantRecordingRef.current = true
    playbackRequestedRef.current = false
    recordedChunksRef.current = []
    cleanupPlayback()

    setIsTalking(true)

    localTracksRef.current.forEach((track) => {
      track.enabled = true
    })

    if (remoteStreamRef.current) {
      startRecorder()
    }
  }, [cleanupPlayback, connectionState, onError, startRecorder])

  const stopTalking = useCallback(() => {
    if (!peerRef.current) return

    wantRecordingRef.current = false
    playbackRequestedRef.current = true

    localTracksRef.current.forEach((track) => {
      track.enabled = false
    })

    setIsTalking(false)

    if (recorderRef.current && recorderRef.current.state === 'recording') {
      setTimeout(() => {
        if (recorderRef.current && recorderRef.current.state === 'recording') {
          stopRecorder()
        } else {
          finalizePlayback()
        }
      }, REMOTE_FLUSH_DELAY_MS)
    } else {
      finalizePlayback()
    }
  }, [finalizePlayback, stopRecorder])

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
