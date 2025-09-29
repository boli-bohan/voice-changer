package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
)

// AudioFormat represents audio format parameters
type AudioFormat struct {
	SampleRate    int
	Channels      int
	BitsPerSample int
}

// StreamingAudioProcessor handles WebM -> PCM -> WAV conversion
type StreamingAudioProcessor struct {
	format     AudioFormat
	webmBuffer *bytes.Buffer
	wavWriter  *WAVWriter
	mu         sync.Mutex
	inputFile  *os.File
	tempDir    string
}

// NewStreamingAudioProcessor creates a new audio processor
func NewStreamingAudioProcessor(sessionID string) (*StreamingAudioProcessor, error) {
	tempDir := filepath.Join("temp_audio", sessionID)
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return nil, err
	}

	// Create input file for WebM accumulation
	inputFile, err := os.CreateTemp(tempDir, "input_*.webm")
	if err != nil {
		return nil, err
	}

	processor := &StreamingAudioProcessor{
		format: AudioFormat{
			SampleRate:    48000,
			Channels:      1,
			BitsPerSample: 16,
		},
		webmBuffer: &bytes.Buffer{},
		tempDir:    tempDir,
		inputFile:  inputFile,
	}

	// Initialize WAV writer for output
	outputFile := filepath.Join(tempDir, "output.wav")
	wavWriter, err := NewWAVWriter(outputFile, processor.format)
	if err != nil {
		return nil, err
	}
	processor.wavWriter = wavWriter

	return processor, nil
}

// StoreWebMChunk persists incoming WebM audio data for later decoding
func (p *StreamingAudioProcessor) StoreWebMChunk(webmData []byte) error {
	if len(webmData) == 0 {
		return nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if _, err := p.webmBuffer.Write(webmData); err != nil {
		return err
	}

	if _, err := p.inputFile.Write(webmData); err != nil {
		return err
	}

	return nil
}

// DecodeFullWebMToPCM converts the stored WebM data to raw PCM samples
func (p *StreamingAudioProcessor) DecodeFullWebMToPCM() ([]int16, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.webmBuffer.Len() == 0 {
		return nil, fmt.Errorf("no webm audio buffered")
	}

	if p.inputFile != nil {
		if err := p.inputFile.Sync(); err != nil {
			return nil, err
		}
	}

	pcmData, err := p.decodeWebMToPCM()
	if err != nil {
		return nil, err
	}

	return pcmData, nil
}

// decodeWebMToPCM uses FFmpeg to convert WebM to raw PCM data
func (p *StreamingAudioProcessor) decodeWebMToPCM() ([]int16, error) {
	// Create temporary output file for PCM
	pcmFile, err := os.CreateTemp(p.tempDir, "temp_*.pcm")
	if err != nil {
		return nil, err
	}
	defer os.Remove(pcmFile.Name())
	defer pcmFile.Close()

	// Use FFmpeg to convert WebM to raw PCM
	cmd := exec.Command("ffmpeg",
		"-y", // Overwrite output
		"-i", p.inputFile.Name(),
		"-f", "s16le", // 16-bit signed little endian
		"-acodec", "pcm_s16le",
		"-ar", fmt.Sprintf("%d", p.format.SampleRate),
		"-ac", fmt.Sprintf("%d", p.format.Channels),
		pcmFile.Name(),
	)

	// Suppress FFmpeg output
	cmd.Stdout = nil
	cmd.Stderr = nil

	if err := cmd.Run(); err != nil {
		return nil, err
	}

	// Read PCM data
	pcmData, err := os.ReadFile(pcmFile.Name())
	if err != nil {
		return nil, err
	}

	// Convert bytes to int16 samples
	samples := make([]int16, len(pcmData)/2)
	for i := 0; i < len(samples); i++ {
		samples[i] = int16(binary.LittleEndian.Uint16(pcmData[i*2:]))
	}

	return samples, nil
}

// ProcessPCMChunk processes processed PCM data and writes to WAV
func (p *StreamingAudioProcessor) ProcessPCMChunk(pcmData []int16) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	return p.wavWriter.WriteSamples(pcmData)
}

// Close finalizes the audio processing and saves files
func (p *StreamingAudioProcessor) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	var inputFileName string

	// Close input file and store filename for cleanup
	if p.inputFile != nil {
		inputFileName = p.inputFile.Name()
		p.inputFile.Close()
	}

	// Close WAV writer
	if p.wavWriter != nil {
		p.wavWriter.Close()
	}

	// Save final input as WAV for comparison
	inputWAVPath := filepath.Join(p.tempDir, "input.wav")
	err := p.convertWebMToWAV(inputFileName, inputWAVPath)

	// Remove temporary input file
	if inputFileName != "" {
		os.Remove(inputFileName)
	}

	return err
}

// convertWebMToWAV converts the final WebM file to WAV
func (p *StreamingAudioProcessor) convertWebMToWAV(webmPath, wavPath string) error {
	cmd := exec.Command("ffmpeg",
		"-y", // Overwrite output
		"-i", webmPath,
		"-acodec", "pcm_s16le",
		"-ar", fmt.Sprintf("%d", p.format.SampleRate),
		"-ac", fmt.Sprintf("%d", p.format.Channels),
		wavPath,
	)

	// Suppress FFmpeg output
	cmd.Stdout = nil
	cmd.Stderr = nil

	return cmd.Run()
}

// WAVWriter handles streaming WAV file writing
type WAVWriter struct {
	file         *os.File
	format       AudioFormat
	samplesCount int64
	headerPos    int64
}

// NewWAVWriter creates a new WAV file writer
func NewWAVWriter(filename string, format AudioFormat) (*WAVWriter, error) {
	file, err := os.Create(filename)
	if err != nil {
		return nil, err
	}

	writer := &WAVWriter{
		file:   file,
		format: format,
	}

	// Write initial WAV header (will be updated when closed)
	if err := writer.writeWAVHeader(0); err != nil {
		file.Close()
		return nil, err
	}

	return writer, nil
}

// WriteSamples writes PCM samples to the WAV file
func (w *WAVWriter) WriteSamples(samples []int16) error {
	for _, sample := range samples {
		if err := binary.Write(w.file, binary.LittleEndian, sample); err != nil {
			return err
		}
		w.samplesCount++
	}
	return nil
}

// Close finalizes the WAV file by updating the header with correct sizes
func (w *WAVWriter) Close() error {
	if w.file == nil {
		return nil
	}

	// Calculate data size
	dataSize := w.samplesCount * int64(w.format.BitsPerSample/8) * int64(w.format.Channels)

	// Seek back to header and update with final sizes
	if _, err := w.file.Seek(0, 0); err != nil {
		return err
	}

	if err := w.writeWAVHeader(dataSize); err != nil {
		return err
	}

	return w.file.Close()
}

// writeWAVHeader writes the WAV file header
func (w *WAVWriter) writeWAVHeader(dataSize int64) error {
	blockAlign := int16(w.format.Channels * w.format.BitsPerSample / 8)
	byteRate := int32(w.format.SampleRate * int(blockAlign))

	fileSize := int32(36 + dataSize)

	// RIFF header
	w.file.WriteString("RIFF")
	binary.Write(w.file, binary.LittleEndian, fileSize)
	w.file.WriteString("WAVE")

	// fmt chunk
	w.file.WriteString("fmt ")
	binary.Write(w.file, binary.LittleEndian, int32(16)) // chunk size
	binary.Write(w.file, binary.LittleEndian, int16(1))  // PCM format
	binary.Write(w.file, binary.LittleEndian, int16(w.format.Channels))
	binary.Write(w.file, binary.LittleEndian, int32(w.format.SampleRate))
	binary.Write(w.file, binary.LittleEndian, byteRate)
	binary.Write(w.file, binary.LittleEndian, blockAlign)
	binary.Write(w.file, binary.LittleEndian, int16(w.format.BitsPerSample))

	// data chunk
	w.file.WriteString("data")
	binary.Write(w.file, binary.LittleEndian, int32(dataSize))

	return nil
}
