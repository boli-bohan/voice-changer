import { describe, it, expect } from 'vitest'
import wrtc from '@koush/wrtc'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { fileURLToPath } from 'node:url'
import { decode as decodeWav } from 'node-wav'

const { RTCPeerConnection, RTCSessionDescription, nonstandard } = wrtc
const { RTCAudioSource } = nonstandard

const SAMPLE_RATE = 48_000
const FRAME_SIZE = 480
const CHANNEL_COUNT = 1
const BITS_PER_SAMPLE = 16

const API_BASE = process.env.VOICE_CHANGER_API_BASE ?? 'http://localhost:8000'
const OFFER_PATH = process.env.VOICE_CHANGER_OFFER_PATH ?? '/webrtc/offer'
const ENABLE_E2E = process.env.VOICE_CHANGER_E2E === '1'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

function loadInputSamples(): Int16Array {
  const wavPath = path.resolve(__dirname, '../../../data/test_input_opus.wav')
  const buffer = readFileSync(wavPath)
  const decoded = decodeWav(buffer)
  if (decoded.sampleRate !== SAMPLE_RATE) {
    throw new Error(`Unexpected sample rate ${decoded.sampleRate}`)
  }
  if (decoded.channelData.length < 1) {
    throw new Error('Decoded WAV file is missing channel data')
  }
  const channel = decoded.channelData[0]
  const pcm = new Int16Array(channel.length)
  for (let i = 0; i < channel.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, channel[i]))
    pcm[i] = Math.round(sample * 32767)
  }
  return pcm
}

async function waitForIceCompletion(pc: RTCPeerConnection): Promise<void> {
  while (pc.iceGatheringState !== 'complete') {
    await delay(50)
  }
}

async function streamSamples(
  source: InstanceType<typeof RTCAudioSource>,
  samples: Int16Array,
  frameSize: number
): Promise<void> {
  const frameDurationMs = (frameSize / SAMPLE_RATE) * 1000
  for (let offset = 0; offset < samples.length; offset += frameSize) {
    const slice = samples.subarray(offset, Math.min(offset + frameSize, samples.length))
    const frame = new Int16Array(frameSize)
    frame.set(slice)
    source.onData({
      samples: frame,
      sampleRate: SAMPLE_RATE,
      bitsPerSample: BITS_PER_SAMPLE,
      channelCount: CHANNEL_COUNT,
      numberOfFrames: frameSize,
    })
    await delay(frameDurationMs)
  }
}

describe.skipIf(!ENABLE_E2E)('voice-changer webrtc worker', () => {
  it('streams audio frames through the worker and receives shifted output', async () => {
    const samples = loadInputSamples()

    const pc = new RTCPeerConnection()
    const source = new RTCAudioSource()
    const track = source.createTrack()
    pc.addTrack(track)

    let receivedTrack = false
    const remoteAudioPromise = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('Timed out waiting for remote audio')),
        10_000
      )
      pc.ontrack = () => {
        receivedTrack = true
        clearTimeout(timeout)
        resolve()
      }
    })

    try {
      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
      await waitForIceCompletion(pc)

      const localDescription = pc.localDescription
      if (!localDescription) {
        throw new Error('Peer connection missing local description')
      }

      const offerUrl = `${API_BASE.replace(/\/$/, '')}${OFFER_PATH}`
      const response = await fetch(offerUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sdp: localDescription.sdp, type: localDescription.type }),
      })

      if (!response.ok) {
        const body = await response.text()
        throw new Error(`Offer exchange failed: ${response.status} ${body}`)
      }

      const answer = await response.json()
      await pc.setRemoteDescription(new RTCSessionDescription(answer))

      await Promise.all([streamSamples(source, samples, FRAME_SIZE), remoteAudioPromise])

      expect(receivedTrack).toBe(true)
    } finally {
      track.stop()
      pc.close()
    }
  }, 30_000)
})
