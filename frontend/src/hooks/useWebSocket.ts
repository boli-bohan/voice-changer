import { useRef, useCallback, useState, useEffect } from 'react'

interface UseWebSocketProps {
  url: string
  onStatusChange: (status: 'connecting' | 'connected' | 'disconnected') => void
  onProcessedAudio: (audioBlob: Blob) => void
  onError: (error: string) => void
  onPlaybackComplete?: () => void
  onPlaybackStarted?: () => void
}

interface AudioFormat {
  sampleRate: number
  channels: number
  bitsPerSample: number
  encoding: string
}

const DEFAULT_AUDIO_FORMAT: AudioFormat = {
  sampleRate: 44100,
  channels: 1,
  bitsPerSample: 16,
  encoding: 'pcm_s16le',
}

const createWavBlobFromPcm = (pcmData: Uint8Array, format: AudioFormat): Blob => {
  const { sampleRate, channels, bitsPerSample } = format
  const headerSize = 44
  const dataLength = pcmData.length
  const buffer = new ArrayBuffer(headerSize + dataLength)
  const view = new DataView(buffer)
  const wavBytes = new Uint8Array(buffer)

  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) {
      wavBytes[offset + i] = value.charCodeAt(i)
    }
  }

  const blockAlign = (channels * bitsPerSample) / 8
  const byteRate = sampleRate * blockAlign

  writeString(0, 'RIFF')
  view.setUint32(4, 36 + dataLength, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, channels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, byteRate, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, bitsPerSample, true)
  writeString(36, 'data')
  view.setUint32(40, dataLength, true)

  wavBytes.set(pcmData, headerSize)

  return new Blob([buffer], { type: 'audio/wav' })
}

export const useWebSocket = ({
  url,
  onStatusChange,
  onProcessedAudio,
  onError,
  onPlaybackComplete,
  onPlaybackStarted,
}: UseWebSocketProps) => {
  const wsRef = useRef<WebSocket | null>(null)
  const [connectionState, setConnectionState] = useState<
    'connecting' | 'connected' | 'disconnected'
  >('disconnected')
  const audioChunksRef = useRef<Uint8Array[]>([])
  const hasPendingAudioRef = useRef(false)
  const audioFormatRef = useRef<AudioFormat>(DEFAULT_AUDIO_FORMAT)
  const audioContextRef = useRef<AudioContext | null>(null)
  const playbackTimeRef = useRef(0)
  const activeSourcesRef = useRef(new Set<AudioBufferSourceNode>())
  const playbackStartedRef = useRef(false)
  const completionPendingRef = useRef(false)
  const streamingSupportedRef = useRef(false)
  const isMountedRef = useRef(true)
  const [streamingSupported, setStreamingSupported] = useState(false)

  const updateStreamingSupported = useCallback((supported: boolean) => {
    streamingSupportedRef.current = supported
    if (isMountedRef.current) {
      setStreamingSupported((prev) => (prev === supported ? prev : supported))
    }
  }, [])

  const cleanupAudioPlayback = useCallback(() => {
    activeSourcesRef.current.forEach((source) => {
      try {
        source.stop()
      } catch (error) {
        // Ignore errors caused by stopping already finished sources
      }
    })
    activeSourcesRef.current.clear()
    playbackTimeRef.current = 0
    playbackStartedRef.current = false
    completionPendingRef.current = false

    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {
        /* swallow close errors */
      })
      audioContextRef.current = null
    }
  }, [])

  useEffect(
    () => () => {
      isMountedRef.current = false
      cleanupAudioPlayback()
    },
    [cleanupAudioPlayback],
  )

  const ensureAudioContext = useCallback(async (): Promise<AudioContext | null> => {
    if (typeof window === 'undefined') {
      updateStreamingSupported(false)
      return null
    }

    const AudioContextConstructor =
      window.AudioContext ||
      (window as typeof window & { webkitAudioContext?: typeof window.AudioContext }).webkitAudioContext

    if (!AudioContextConstructor) {
      updateStreamingSupported(false)
      return null
    }

    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContextConstructor()
    }

    if (audioContextRef.current.state === 'suspended') {
      try {
        await audioContextRef.current.resume()
      } catch (error) {
        updateStreamingSupported(false)
        return null
      }
    }

    updateStreamingSupported(true)
    return audioContextRef.current
  }, [updateStreamingSupported])

  const schedulePcmPlayback = useCallback(
    async (pcmChunk: Uint8Array) => {
      const format = audioFormatRef.current
      const bytesPerSample = format.bitsPerSample / 8

      if (bytesPerSample !== 2 || pcmChunk.length % bytesPerSample !== 0) {
        updateStreamingSupported(false)
        return
      }

      const audioContext = await ensureAudioContext()
      if (!audioContext) {
        return
      }

      const totalSamples = pcmChunk.length / bytesPerSample
      const frameCount = totalSamples / format.channels

      if (!Number.isFinite(frameCount) || frameCount <= 0) {
        return
      }

      const audioBuffer = audioContext.createBuffer(format.channels, frameCount, format.sampleRate)
      const dataView = new DataView(pcmChunk.buffer, pcmChunk.byteOffset, pcmChunk.byteLength)
      let offset = 0

      for (let frame = 0; frame < frameCount; frame++) {
        for (let channel = 0; channel < format.channels; channel++) {
          const sample = dataView.getInt16(offset, true)
          offset += bytesPerSample
          const normalizedSample = Math.max(-1, Math.min(1, sample / 32768))
          audioBuffer.getChannelData(channel)[frame] = normalizedSample
        }
      }

      const source = audioContext.createBufferSource()
      source.buffer = audioBuffer
      source.connect(audioContext.destination)

      const startTime = Math.max(playbackTimeRef.current, audioContext.currentTime)
      source.start(startTime)
      playbackTimeRef.current = startTime + audioBuffer.duration
      activeSourcesRef.current.add(source)

      source.onended = () => {
        activeSourcesRef.current.delete(source)

        if (activeSourcesRef.current.size === 0 && audioContextRef.current) {
          playbackTimeRef.current = audioContextRef.current.currentTime

          if (completionPendingRef.current) {
            completionPendingRef.current = false
            onPlaybackComplete?.()
          }
        }
      }

      if (!playbackStartedRef.current) {
        playbackStartedRef.current = true
        onPlaybackStarted?.()
      }
    },
    [ensureAudioContext, onPlaybackComplete, onPlaybackStarted, updateStreamingSupported],
  )

  const flushAudioBuffer = useCallback(() => {
    if (!hasPendingAudioRef.current || audioChunksRef.current.length === 0) {
      return
    }

    const totalLength = audioChunksRef.current.reduce((sum, chunk) => sum + chunk.length, 0)
    const combinedArray = new Uint8Array(totalLength)
    let offset = 0

    for (const chunk of audioChunksRef.current) {
      combinedArray.set(chunk, offset)
      offset += chunk.length
    }

    const audioBlob = createWavBlobFromPcm(combinedArray, audioFormatRef.current)
    onProcessedAudio(audioBlob)
    audioChunksRef.current = []
    hasPendingAudioRef.current = false
  }, [onProcessedAudio])

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
        resolve()
      }

      ws.onmessage = async (event) => {
        try {
          if (event.data instanceof Blob) {
            // Binary audio data
            const arrayBuffer = await event.data.arrayBuffer()
            const uint8Array = new Uint8Array(arrayBuffer)
            audioChunksRef.current.push(uint8Array)
            hasPendingAudioRef.current = true
            await schedulePcmPlayback(uint8Array)
          } else {
            // JSON message
            const message = JSON.parse(event.data)

            if (message.type === 'audio_complete') {
              flushAudioBuffer()
            } else if (message.type === 'audio_format') {
              const { sample_rate, channels, bits_per_sample, encoding } = message
              audioFormatRef.current = {
                sampleRate: sample_rate ?? DEFAULT_AUDIO_FORMAT.sampleRate,
                channels: channels ?? DEFAULT_AUDIO_FORMAT.channels,
                bitsPerSample: bits_per_sample ?? DEFAULT_AUDIO_FORMAT.bitsPerSample,
                encoding: encoding ?? DEFAULT_AUDIO_FORMAT.encoding,
              }
            } else if (message.type === 'done') {
              // Server has finished playing back all audio
              flushAudioBuffer()
              if (!streamingSupportedRef.current) {
                completionPendingRef.current = false
                onPlaybackComplete?.()
              } else if (activeSourcesRef.current.size === 0) {
                completionPendingRef.current = false
                onPlaybackComplete?.()
              } else {
                completionPendingRef.current = true
              }
            } else if (message.type === 'error') {
              onError(message.message || 'Unknown server error')
            }
          }
        } catch (error) {
          onError('Failed to process server message')
        }
      }

      ws.onerror = () => {
        setConnectionState('disconnected')
        onStatusChange('disconnected')
        flushAudioBuffer()
        cleanupAudioPlayback()
        reject(new Error('WebSocket connection failed'))
      }

      ws.onclose = () => {
        setConnectionState('disconnected')
        onStatusChange('disconnected')
        flushAudioBuffer()
        cleanupAudioPlayback()
      }
    })
  }, [
    url,
    onStatusChange,
    flushAudioBuffer,
    onError,
    schedulePcmPlayback,
    cleanupAudioPlayback,
  ])

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setConnectionState('disconnected')
    onStatusChange('disconnected')
    flushAudioBuffer()
    audioChunksRef.current = []
    hasPendingAudioRef.current = false
    cleanupAudioPlayback()
  }, [cleanupAudioPlayback, flushAudioBuffer, onStatusChange])

  const sendMessage = useCallback(
    (message: any) => {
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
    void ensureAudioContext()
    sendMessage({ type: 'start_recording' })
  }, [ensureAudioContext, sendMessage])

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
  }
}
