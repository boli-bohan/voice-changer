import type { CSSProperties } from 'react'
import './AudioWaveform.css'

type AudioWaveformMode = 'inactive' | 'input' | 'output'

interface AudioWaveformProps {
  mode: AudioWaveformMode
  caption?: string
}

const modeCaptions: Record<AudioWaveformMode, string> = {
  inactive: 'Waveform idle',
  input: 'Listening…',
  output: 'Playing back…',
}

const BAR_COUNT = 12

const AudioWaveform: React.FC<AudioWaveformProps> = ({ mode, caption }) => {
  return (
    <div className={`audio-waveform mode-${mode}`}>
      <div className="audio-waveform-bars" aria-hidden>
        {Array.from({ length: BAR_COUNT }).map((_, index) => (
          <span
            key={index}
            className="waveform-bar"
            style={{ '--wave-index': index } as CSSProperties}
          />
        ))}
      </div>
      <p className="audio-waveform-caption">{caption ?? modeCaptions[mode]}</p>
    </div>
  )
}

export type { AudioWaveformMode }
export default AudioWaveform
