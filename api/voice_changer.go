package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// Message types for WebSocket communication
type Message struct {
	Type    string `json:"type"`
	Message string `json:"message,omitempty"`
}

// VoiceChangerClient manages connection to the voice_changer.py worker
type VoiceChangerClient struct {
	conn      *websocket.Conn
	connected bool
	mu        sync.RWMutex
}

// NewVoiceChangerClient creates a new client connection to the worker
func NewVoiceChangerClient() *VoiceChangerClient {
	return &VoiceChangerClient{}
}

// Connect establishes connection to the voice changer worker
func (vc *VoiceChangerClient) Connect() error {
	vc.mu.Lock()
	defer vc.mu.Unlock()

	if vc.connected {
		return nil
	}

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.Dial("ws://127.0.0.1:8001/process", nil)
	if err != nil {
		log.Printf("❌ Failed to connect to voice changer worker: %v", err)
		return err
	}

	vc.conn = conn
	vc.connected = true
	log.Println("🔗 Connected to voice changer worker: ws://127.0.0.1:8001/process")
	return nil
}

// SendAudioChunk sends audio data to the worker
func (vc *VoiceChangerClient) SendAudioChunk(data []byte) error {
	vc.mu.RLock()
	defer vc.mu.RUnlock()

	if !vc.connected || vc.conn == nil {
		return fmt.Errorf("not connected to worker")
	}

	err := vc.conn.WriteMessage(websocket.BinaryMessage, data)
	if err != nil {
		log.Printf("❌ Failed to send audio chunk to worker: %v", err)
		vc.connected = false
		return err
	}
	return nil
}

// SendFlush sends flush command to the worker
func (vc *VoiceChangerClient) SendFlush() error {
	vc.mu.RLock()
	defer vc.mu.RUnlock()

	if !vc.connected || vc.conn == nil {
		return fmt.Errorf("not connected to worker")
	}

	flushMsg := Message{Type: "flush"}
	err := vc.conn.WriteJSON(flushMsg)
	if err != nil {
		log.Printf("❌ Failed to send flush to worker: %v", err)
		return err
	}
	return nil
}

// ReceiveAudioChunk receives processed audio from the worker
func (vc *VoiceChangerClient) ReceiveAudioChunk() ([]byte, bool, error) {
	vc.mu.RLock()
	defer vc.mu.RUnlock()

	if !vc.connected || vc.conn == nil {
		return nil, false, fmt.Errorf("not connected to worker")
	}

	messageType, data, err := vc.conn.ReadMessage()
	if err != nil {
		log.Printf("❌ Failed to receive from worker: %v", err)
		vc.connected = false
		return nil, false, err
	}

	if messageType == websocket.BinaryMessage {
		return data, false, nil
	} else if messageType == websocket.TextMessage {
		var msg Message
		if err := json.Unmarshal(data, &msg); err == nil && msg.Type == "done" {
			log.Println("✅ Voice changer worker finished processing")
			return nil, true, nil
		}
	}

	return nil, false, nil
}

// Close closes the connection to the worker
func (vc *VoiceChangerClient) Close() error {
	vc.mu.Lock()
	defer vc.mu.Unlock()

	if vc.conn != nil {
		err := vc.conn.Close()
		vc.conn = nil
		vc.connected = false
		log.Println("🔌 Disconnected from voice changer worker")
		return err
	}
	return nil
}

// ClientConnection represents a WebSocket connection from the frontend
type ClientConnection struct {
	ID              string
	Conn            *websocket.Conn
	VoiceChanger    *VoiceChangerClient
	Recording       bool
	InputBuffer     []byte
	ProcessedChunks [][]byte
	StreamingActive bool
	AudioProcessor  *StreamingAudioProcessor
	mu              sync.RWMutex
}

// ConnectionManager manages all client connections
type ConnectionManager struct {
	connections map[string]*ClientConnection
	mu          sync.RWMutex
	idCounter   int
}

// NewConnectionManager creates a new connection manager
func NewConnectionManager() *ConnectionManager {
	return &ConnectionManager{
		connections: make(map[string]*ClientConnection),
	}
}

// AddConnection adds a new client connection
func (cm *ConnectionManager) AddConnection(conn *websocket.Conn) *ClientConnection {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.idCounter++
	id := fmt.Sprintf("conn_%d", cm.idCounter)

	// Initialize audio processor
	audioProcessor, err := NewStreamingAudioProcessor(id)
	if err != nil {
		log.Printf("Failed to create audio processor for %s: %v", id, err)
		audioProcessor = nil
	}

	client := &ClientConnection{
		ID:              id,
		Conn:            conn,
		VoiceChanger:    NewVoiceChangerClient(),
		Recording:       false,
		InputBuffer:     make([]byte, 0),
		ProcessedChunks: make([][]byte, 0),
		StreamingActive: false,
		AudioProcessor:  audioProcessor,
	}

	cm.connections[id] = client
	log.Printf("New connection established: %s", id)
	return client
}

// RemoveConnection removes a client connection
func (cm *ConnectionManager) RemoveConnection(id string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if client, exists := cm.connections[id]; exists {
		client.VoiceChanger.Close()
		if client.AudioProcessor != nil {
			client.AudioProcessor.Close()
		}
		delete(cm.connections, id)
		log.Printf("Connection disconnected: %s", id)
	}
}

// GetConnection retrieves a connection by ID
func (cm *ConnectionManager) GetConnection(id string) (*ClientConnection, bool) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	conn, exists := cm.connections[id]
	return conn, exists
}

// GetConnectionCount returns the number of active connections
func (cm *ConnectionManager) GetConnectionCount() int {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return len(cm.connections)
}

// sendMessage sends a JSON message to the client
func sendMessage(client *ClientConnection, msgType, message string) {
	msg := Message{Type: msgType, Message: message}
	client.Conn.WriteJSON(msg)
}

// sendAudioData sends binary audio data to the client
func sendAudioData(client *ClientConnection, data []byte) {
	client.Conn.WriteMessage(websocket.BinaryMessage, data)
}

// StartRecording initiates recording session
func StartRecording(client *ClientConnection) error {
	client.mu.Lock()
	defer client.mu.Unlock()

	if client.Recording {
		log.Printf("Connection %s already recording", client.ID)
		return nil
	}

	client.Recording = true
	client.InputBuffer = client.InputBuffer[:0]         // Reset input buffer
	client.ProcessedChunks = client.ProcessedChunks[:0] // Reset processed chunks

	// Connect to voice changer worker
	if err := client.VoiceChanger.Connect(); err != nil {
		log.Printf("Failed to connect to voice changer worker for %s: %v", client.ID, err)
		return err
	}

	client.StreamingActive = true
	log.Printf("✅ Started recording for connection %s", client.ID)
	log.Printf("🎵 Streaming mode enabled for %s", client.ID)

	// Start receiver goroutine
	go ContinuousReceiver(client)

	return nil
}

// AddAudioChunk adds audio chunk and processes through the new pipeline
func AddAudioChunk(client *ClientConnection, audioData []byte) {
	client.mu.Lock()
	client.InputBuffer = append(client.InputBuffer, audioData...)
	bufferSize := len(client.InputBuffer)
	client.mu.Unlock()

	log.Printf("📊 Added audio chunk to %s: %d bytes, buffer total: %d bytes",
		client.ID, len(audioData), bufferSize)

	// Process through new audio pipeline
	if client.StreamingActive && client.AudioProcessor != nil {
		// Convert WebM to PCM
		pcmSamples, err := client.AudioProcessor.ProcessWebMChunk(audioData)
		if err != nil {
			log.Printf("Failed to process WebM chunk for %s: %v", client.ID, err)
			return
		}

		if pcmSamples != nil {
			// Convert PCM samples to bytes for sending to Python worker
			pcmBytes := make([]byte, len(pcmSamples)*2)
			for i, sample := range pcmSamples {
				pcmBytes[i*2] = byte(sample)
				pcmBytes[i*2+1] = byte(sample >> 8)
			}

			// Send PCM data to Python worker
			if err := client.VoiceChanger.SendAudioChunk(pcmBytes); err != nil {
				log.Printf("Failed to send PCM chunk to voice changer for %s: %v", client.ID, err)
			}
		}
	}
}

// ContinuousReceiver handles incoming data from voice changer worker
func ContinuousReceiver(client *ClientConnection) {
	log.Printf("🎧 Started continuous receiver for %s", client.ID)
	playbackStarted := false

	for client.VoiceChanger.connected {
		// Receive processed PCM chunk
		processedChunk, done, err := client.VoiceChanger.ReceiveAudioChunk()
		if err != nil {
			log.Printf("Error in continuous receiver for %s: %v", client.ID, err)
			break
		}

		if done {
			log.Printf("✅ Voice changer finished processing for %s", client.ID)
			break
		}

		if len(processedChunk) > 0 {
			// Convert PCM bytes back to int16 samples
			sampleCount := len(processedChunk) / 2
			pcmSamples := make([]int16, sampleCount)
			for i := 0; i < sampleCount; i++ {
				pcmSamples[i] = int16(processedChunk[i*2]) | (int16(processedChunk[i*2+1]) << 8)
			}

			// Write to WAV file for output
			if client.AudioProcessor != nil {
				client.AudioProcessor.ProcessPCMChunk(pcmSamples)
			}

			// Convert to WAV format for streaming to client
			wavBytes := convertPCMToWAVChunk(pcmSamples)

			// Add to processed chunks for streaming
			client.mu.Lock()
			client.ProcessedChunks = append(client.ProcessedChunks, wavBytes)
			client.mu.Unlock()

			// Start streaming playback if this is the first chunk
			if !playbackStarted {
				go StartStreamingPlayback(client)
				playbackStarted = true
			}

			log.Printf("🎵 Received processed PCM chunk for %s: %d samples -> %d WAV bytes",
				client.ID, len(pcmSamples), len(wavBytes))
		}
	}

	log.Printf("🎧 Stopped continuous receiver for %s", client.ID)
}

// convertPCMToWAVChunk converts PCM samples to a WAV audio chunk for streaming
func convertPCMToWAVChunk(samples []int16) []byte {
	wavData := make([]byte, len(samples)*2)
	for i, sample := range samples {
		wavData[i*2] = byte(sample)
		wavData[i*2+1] = byte(sample >> 8)
	}
	return wavData
}

// StartStreamingPlayback streams processed audio back to client
func StartStreamingPlayback(client *ClientConnection) {
	sendMessage(client, "streaming_started", "Starting audio playback")
	log.Printf("🎵 Started streaming playback for %s", client.ID)

	go StreamChunksToClient(client)
}

// StreamChunksToClient streams processed audio chunks to client
func StreamChunksToClient(client *ClientConnection) {
	chunkIndex := 0

	for {
		client.mu.RLock()
		recording := client.Recording
		chunksLen := len(client.ProcessedChunks)
		client.mu.RUnlock()

		// Stream available chunks
		for chunkIndex < chunksLen {
			client.mu.RLock()
			if chunkIndex < len(client.ProcessedChunks) {
				chunk := client.ProcessedChunks[chunkIndex]
				client.mu.RUnlock()
				sendAudioData(client, chunk)
				chunkIndex++
				time.Sleep(10 * time.Millisecond) // Small delay to prevent overwhelming
			} else {
				client.mu.RUnlock()
				break
			}
		}

		// Exit if recording stopped AND all chunks have been streamed
		if !recording && chunkIndex >= chunksLen {
			break
		}

		// Wait before checking for new chunks
		time.Sleep(100 * time.Millisecond)
	}

	// Send completion message
	sendMessage(client, "streaming_completed", "Audio streaming completed")
	log.Printf("✅ Completed streaming playback for %s", client.ID)

	// Send done message to signal client that all audio playback has finished
	sendMessage(client, "done", "Audio playback finished")
	log.Printf("🏁 Sent done message to %s", client.ID)
}

// StopRecordingAndProcess stops recording and processes the audio
func StopRecordingAndProcess(client *ClientConnection) error {
	log.Printf("🔄 Starting stop_recording_and_process for %s", client.ID)

	client.mu.Lock()
	if !client.Recording {
		client.mu.Unlock()
		log.Printf("Connection %s not recording in stop_recording_and_process", client.ID)
		return nil
	}

	client.Recording = false
	client.mu.Unlock()
	log.Printf("✅ Stopped recording for %s", client.ID)

	// Send flush command to voice changer worker
	if client.StreamingActive {
		log.Printf("🔄 Sending flush command to voice changer for %s", client.ID)
		if err := client.VoiceChanger.SendFlush(); err != nil {
			log.Printf("Failed to send flush to voice changer for %s: %v", client.ID, err)
		}

		// Give some time for final processing
		time.Sleep(2 * time.Second)

		// Disconnect from worker
		client.VoiceChanger.Close()
	}

	// Save session files for debugging
	go SaveSessionFiles(client)

	return nil
}

// SaveSessionFiles saves input and output audio files for debugging
func SaveSessionFiles(client *ClientConnection) {
	tempDir := "temp_audio"
	sessionDir := filepath.Join(tempDir, client.ID)

	// Create session directory
	if err := os.MkdirAll(sessionDir, 0755); err != nil {
		log.Printf("Failed to create session directory for %s: %v", client.ID, err)
		return
	}

	client.mu.RLock()
	inputBuffer := make([]byte, len(client.InputBuffer))
	copy(inputBuffer, client.InputBuffer)

	var outputBuffer []byte
	for _, chunk := range client.ProcessedChunks {
		outputBuffer = append(outputBuffer, chunk...)
	}
	client.mu.RUnlock()

	// Save input file as WAV
	inputFile := filepath.Join(sessionDir, "input.wav")
	if err := os.WriteFile(inputFile, inputBuffer, 0644); err != nil {
		log.Printf("Failed to save input file for %s: %v", client.ID, err)
	} else {
		log.Printf("💿 Saved input audio: %s (%d bytes)", inputFile, len(inputBuffer))
	}

	// Save output file if we have processed audio
	if len(outputBuffer) > 0 {
		outputFile := filepath.Join(sessionDir, "output.wav")
		if err := os.WriteFile(outputFile, outputBuffer, 0644); err != nil {
			log.Printf("Failed to save output file for %s: %v", client.ID, err)
		} else {
			log.Printf("💾 Saved output audio: %s (%d bytes)", outputFile, len(outputBuffer))
		}
	}

	log.Printf("🗂️ Session files saved to: %s", sessionDir)
}
