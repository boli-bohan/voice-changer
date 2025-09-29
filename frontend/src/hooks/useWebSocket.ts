import { useRef, useCallback, useState, useEffect, useMemo } from 'react'

declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext
  }
}

interface UseWebSocketProps {
  url: string
  onStatusChange: (status: 'connecting' | 'connected' | 'disconnected') => void
  onError: (error: string) => void
  onPlaybackComplete?: () => void
  onPlaybackStarted?: () => void
}

const SAMPLE_RATE = 44100
const CHANNELS = 1
const JITTER_MS = 200

const createWorkletUrl = () => {
  const code = `
class PCMQueue {
  constructor() {
    this.chunks = []
    this.headIndex = 0
    this.offset = 0
    this.totalFrames = 0
  }
  push(arr) {
    if (arr && arr.length) {
      this.chunks.push(arr)
      this.totalFrames += arr.length
    }
  }
  popFrames(n) {
    const out = new Float32Array(n)
    let i = 0
    while (i < n) {
      const cur = this.chunks[this.headIndex]
      if (!cur) break
      const rem = cur.length - this.offset
      const take = Math.min(rem, n - i)
      out.set(cur.subarray(this.offset, this.offset + take), i)
      i += take
      this.offset += take
      if (this.offset >= cur.length) {
        this.headIndex++
        this.offset = 0
        if (this.headIndex > 32) {
          this.chunks = this.chunks.slice(this.headIndex)
          this.headIndex = 0
        }
      }
    }
    this.totalFrames -= i
    return { filled: i, out }
  }
  clear() {
    this.chunks = []
    this.headIndex = 0
    this.offset = 0
    this.totalFrames = 0
  }
}

class PCMPlayerProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    this.channels = options?.processorOptions?.channels || 1
    this.jitterFrames = options?.processorOptions?.jitterFrames || 0
    this.started = false
    this.underruns = 0
    this.tick = 0
    this.queues = Array.from({ length: this.channels }, () => new PCMQueue())

    this.port.onmessage = (e) => {
      const msg = e.data || {}
      if (msg.type === 'push') {
        const bufs = msg.buffers || []
        for (let c = 0; c < this.channels; c++) {
          const arr = bufs[c] ? new Float32Array(bufs[c]) : null
          if (arr) this.queues[c].push(arr)
        }
      } else if (msg.type === 'flush') {
        for (let q of this.queues) q.clear()
        this.started = false
        this.underruns = 0
      }
    }
  }

  process(_, outputs) {
    const output = outputs[0]
    const frames = output[0].length
    const bufferedFrames = this.queues[0].totalFrames

    if (!this.started) {
      if (bufferedFrames >= this.jitterFrames) {
        this.started = true
      } else {
        for (let c = 0; c < output.length; c++) output[c].fill(0)
        if ((this.tick++ & 15) === 0) {
          this.port.postMessage({
            type: 'status',
            bufferedFrames,
            underruns: this.underruns,
          })
        }
        return true
      }
    }

    let hadData = true
    for (let c = 0; c < output.length; c++) {
      const { filled, out } = this.queues[c].popFrames(frames)
      if (filled < frames) hadData = false
      output[c].set(out)
    }
    if (!hadData) this.underruns++

    if ((this.tick++ & 15) === 0) {
      this.port.postMessage({
        type: 'status',
        bufferedFrames: this.queues[0].totalFrames,
        underruns: this.underruns,
      })
    }

    return true
  }
}

registerProcessor('pcm-player', PCMPlayerProcessor)
`

  const blob = new Blob([code], { type: 'text/javascript' })
  return URL.createObjectURL(blob)
}

const interleavedToPlanar = (data: Float32Array, channels: number) => {
  if (channels === 1) {
    return [data]
  }

  const frames = Math.floor(data.length / channels)
  return Array.from({ length: channels }, (_, ch) => {
    const channelData = new Float32Array(frames)
    for (let i = 0; i < frames; i++) {
      channelData[i] = data[i * channels + ch]
    }
    return channelData
  })
}

const linearResample = (input: Float32Array, srcRate: number, dstRate: number) => {
  if (srcRate === dstRate || input.length === 0) return input
  const ratio = dstRate / srcRate
  const outLen = Math.max(1, Math.floor(input.length * ratio))
  const out = new Float32Array(outLen)
  const scale = (input.length - 1) / (outLen - 1)

  for (let i = 0; i < outLen; i++) {
    const pos = i * scale
    const idx = Math.floor(pos)
    const frac = pos - idx
    const a = input[idx]
    const b = input[Math.min(idx + 1, input.length - 1)]
    out[i] = a + (b - a) * frac
  }

  return out
}

export const useWebSocket = ({
  url,
  onStatusChange,
  onError,
  onPlaybackComplete,
  onPlaybackStarted,
}: UseWebSocketProps) => {
  const wsRef = useRef<WebSocket | null>(null)
  const [connectionState, setConnectionState] = useState<
    'connecting' | 'connected' | 'disconnected'
  >('disconnected')

  const ctxRef = useRef<AudioContext | null>(null)
  const nodeRef = useRef<AudioWorkletNode | null>(null)
  const workletReadyRef = useRef(false)
  const playbackStartedRef = useRef(false)
  const completionPendingRef = useRef(false)

  const workletUrl = useMemo(() => createWorkletUrl(), [])

  const cleanupAudioPlayback = useCallback(() => {
    playbackStartedRef.current = false
    completionPendingRef.current = false

    try {
      nodeRef.current?.port.postMessage({ type: 'flush' })
    } catch {}

    try {
      nodeRef.current?.disconnect()
    } catch {}

    nodeRef.current = null
    workletReadyRef.current = false

    if (ctxRef.current) {
      const ctx = ctxRef.current
      ctxRef.current = null
      void ctx.close()
    }
  }, [])

  useEffect(() => {
    return () => {
      cleanupAudioPlayback()
      URL.revokeObjectURL(workletUrl)
    }
  }, [cleanupAudioPlayback, workletUrl])

  const ensureAudioGraph = useCallback(async () => {
    if (nodeRef.current && ctxRef.current) {
      return nodeRef.current
    }

    const AudioCtx = window.AudioContext ?? window.webkitAudioContext
    if (!AudioCtx) {
      onError('Web Audio API is not supported in this browser')
      return null
    }

    const ctx = new AudioCtx()
    ctxRef.current = ctx

    if (ctx.state === 'suspended') {
      await ctx.resume()
    }

    try {
      if (!workletReadyRef.current) {
        await ctx.audioWorklet.addModule(workletUrl)
        workletReadyRef.current = true
      }

      const jitterFrames = Math.floor((JITTER_MS / 1000) * ctx.sampleRate)
      const node = new AudioWorkletNode(ctx, 'pcm-player', {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [CHANNELS],
        processorOptions: { channels: CHANNELS, jitterFrames },
      })

      node.port.onmessage = (event: MessageEvent) => {
        const msg = event.data || {}
        if (msg.type === 'status') {
          if (completionPendingRef.current && (msg.bufferedFrames ?? 0) === 0) {
            completionPendingRef.current = false
            onPlaybackComplete?.()
          }
        }
      }

      node.connect(ctx.destination)
      nodeRef.current = node
      return node
    } catch (error) {
      console.error('Failed to initialise AudioWorklet', error)
      cleanupAudioPlayback()
      onError('Audio playback is not supported in this browser')
      return null
    }
  }, [cleanupAudioPlayback, onError, onPlaybackComplete, workletUrl])

  const handleAudioChunk = useCallback(
    async (arrayBuffer: ArrayBuffer) => {
      const node = await ensureAudioGraph()
      if (!node || !ctxRef.current) return

      completionPendingRef.current = false

      const sourceRate = SAMPLE_RATE
      const destRate = ctxRef.current.sampleRate

      const interleaved = new Float32Array(arrayBuffer)
      const planar = interleavedToPlanar(interleaved, CHANNELS).map((channel) =>
        sourceRate === destRate ? channel : linearResample(channel, sourceRate, destRate),
      )

      const buffers = planar.map((channel) => channel.buffer)
      node.port.postMessage({ type: 'push', buffers }, buffers)

      if (!playbackStartedRef.current) {
        playbackStartedRef.current = true
        onPlaybackStarted?.()
      }
    },
    [ensureAudioGraph, onPlaybackStarted],
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
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws

      ws.onopen = () => {
        setConnectionState('connected')
        onStatusChange('connected')
        resolve()
      }

      ws.onmessage = async (event) => {
        if (event.data instanceof ArrayBuffer) {
          await handleAudioChunk(event.data)
          return
        }

        try {
          const message = JSON.parse(event.data as string)
          if (message.type === 'done') {
            if (nodeRef.current) {
              completionPendingRef.current = true
            } else {
              onPlaybackComplete?.()
            }
          } else if (message.type === 'error') {
            onError(message.message || 'Unknown server error')
          }
        } catch {
          // Ignore malformed messages
        }
      }

      ws.onerror = () => {
        setConnectionState('disconnected')
        onStatusChange('disconnected')
        cleanupAudioPlayback()
        reject(new Error('WebSocket connection failed'))
      }

      ws.onclose = () => {
        setConnectionState('disconnected')
        onStatusChange('disconnected')
        cleanupAudioPlayback()
      }
    })
  }, [
    cleanupAudioPlayback,
    handleAudioChunk,
    onError,
    onPlaybackComplete,
    onStatusChange,
    url,
  ])

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setConnectionState('disconnected')
    onStatusChange('disconnected')
    cleanupAudioPlayback()
  }, [cleanupAudioPlayback, onStatusChange])

  const sendMessage = useCallback(
    (message: any) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(message))
      } else {
        onError('WebSocket not connected')
      }
    },
    [onError],
  )

  const sendAudioData = useCallback(
    (audioBlob: Blob) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(audioBlob)
      } else {
        onError('WebSocket not connected')
      }
    },
    [onError],
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
