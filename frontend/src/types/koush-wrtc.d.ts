declare module '@koush/wrtc' {
  class RTCPeerConnection {
    constructor(configuration?: unknown)
    createOffer(options?: unknown): Promise<any>
    setLocalDescription(description?: any): Promise<void>
    setRemoteDescription(description: any): Promise<void>
    addTrack(track: any, ...streams: any[]): any
    getTransceivers(): any[]
    close(): void
    readonly localDescription: any
    readonly iceGatheringState: string
    ontrack: ((event: any) => void) | null
  }

  class RTCSessionDescription {
    constructor(descriptionInitDict: any)
  }

  const nonstandard: {
    RTCAudioSource: new () => {
      createTrack(): any
      onData(data: {
        samples: Int16Array
        sampleRate: number
        bitsPerSample: number
        channelCount: number
        numberOfFrames: number
      }): void
    }
  }

  const wrtc: {
    RTCPeerConnection: typeof RTCPeerConnection
    RTCSessionDescription: typeof RTCSessionDescription
    nonstandard: typeof nonstandard
  }

  export default wrtc
}
