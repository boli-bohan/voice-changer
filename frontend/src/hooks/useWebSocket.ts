import { useRef, useCallback, useState } from 'react'

interface UseWebSocketProps {
  url: string
  onStatusChange: (status: 'connecting' | 'connected' | 'disconnected') => void
  onProcessedAudio: (audioBlob: Blob) => void
  onError: (error: string) => void
}

export const useWebSocket = ({
  url,
  onStatusChange,
  onProcessedAudio,
  onError,
}: UseWebSocketProps) => {
  const wsRef = useRef<WebSocket | null>(null)
  const [connectionState, setConnectionState] = useState<
    'connecting' | 'connected' | 'disconnected'
  >('disconnected')
  const audioChunksRef = useRef<Uint8Array[]>([])
  const hasPendingAudioRef = useRef(false)

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

    const audioBlob = new Blob([combinedArray], { type: 'audio/wav' })
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
          } else {
            // JSON message
            const message = JSON.parse(event.data)

            if (message.type === 'audio_complete') {
              flushAudioBuffer()
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
        reject(new Error('WebSocket connection failed'))
      }

      ws.onclose = () => {
        setConnectionState('disconnected')
        onStatusChange('disconnected')
        flushAudioBuffer()
      }
    })
  }, [url, onStatusChange, flushAudioBuffer, onError])

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
  }, [flushAudioBuffer, onStatusChange])

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
  }
}
