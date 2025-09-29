import { useRef, useCallback, useState } from 'react'

interface UseAudioRecorderProps {
  onAudioData: (audioBlob: Blob) => void
  onError: (error: string) => void
}

export const useAudioRecorder = ({ onAudioData, onError }: UseAudioRecorderProps) => {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const [isRecording, setIsRecording] = useState(false)

  const startRecording = useCallback(async () => {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 48000,
        },
      })

      streamRef.current = stream
      audioChunksRef.current = []

      // Create MediaRecorder
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType,
        audioBitsPerSecond: 128000,
      })

      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
          // Send audio data in real-time chunks
          onAudioData(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        setIsRecording(false)
        // Send final audio blob when recording stops
        if (audioChunksRef.current.length > 0) {
          const finalBlob = new Blob(audioChunksRef.current, { type: mimeType })
          onAudioData(finalBlob)
        }
      }

      mediaRecorder.onerror = (event) => {
        setIsRecording(false)
        onError('MediaRecorder error: ' + (event.error?.message || 'Unknown error'))
      }

      // Start recording with data available every 100ms
      mediaRecorder.start(100)
      setIsRecording(true)
    } catch (error) {
      setIsRecording(false)
      if (error instanceof Error) {
        if (error.name === 'NotAllowedError') {
          onError('Microphone access denied. Please allow microphone access and try again.')
        } else if (error.name === 'NotFoundError') {
          onError('No microphone found. Please connect a microphone and try again.')
        } else {
          onError('Failed to access microphone: ' + error.message)
        }
      } else {
        onError('Failed to access microphone: Unknown error')
      }
    }
  }, [onAudioData, onError])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
    }

    // Stop all media tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop()
      })
      streamRef.current = null
    }

    mediaRecorderRef.current = null
    setIsRecording(false)
  }, [isRecording])

  return {
    startRecording,
    stopRecording,
    isRecording,
  }
}
