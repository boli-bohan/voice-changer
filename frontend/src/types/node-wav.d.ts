declare module 'node-wav' {
  export function decode(buffer: ArrayBuffer | ArrayLike<number> | Buffer): {
    sampleRate: number
    channelData: Float32Array[]
  }
}
