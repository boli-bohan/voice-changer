package main

import (
	"encoding/binary"
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
	Type      string        `json:"type"`
	Message   string        `json:"message,omitempty"`
	SDP       string        `json:"sdp,omitempty"`
	SDPType   string        `json:"sdp_type,omitempty"`
	Candidate *ICECandidate `json:"candidate,omitempty"`
}

// ICECandidate represents a trickled ICE candidate from the client
type ICECandidate struct {
	Candidate     string  `json:"candidate"`
	SDPMid        *string `json:"sdpMid,omitempty"`
	SDPMLineIndex *uint16 `json:"sdpMLineIndex,omitempty"`
}

// AudioFormatMessage communicates PCM stream metadata to clients
type AudioFormatMessage struct {
	Type          string `json:"type"`
	SampleRate    int    `json:"sample_rate"`
	Channels      int    `json:"channels"`
	BitsPerSample int    `json:"bits_per_sample"`
	Encoding      string `json:"encoding"`
}

const (
	defaultSampleRate = 48000
	defaultChannels   = 1
	bitsPerSample     = 16
)

// VoiceChangerClient manages connection to the voice_changer.py worker
type VoiceChangerClient struct {
	conn        *websocket.Conn
	connected   bool
	mu          sync.RWMutex
	audioChan   chan []byte
	controlChan chan Message
	done        chan struct{}
	closeOnce   sync.Once
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
	vc.audioChan = make(chan []byte, 32)
	vc.controlChan = make(chan Message, 32)
	vc.done = make(chan struct{})
	vc.closeOnce = sync.Once{}
	log.Println("🔗 Connected to voice changer worker: ws://127.0.0.1:8001/process")
	go vc.readLoop()
	return nil
}

func (vc *VoiceChangerClient) readLoop() {
	for {
		messageType, data, err := vc.conn.ReadMessage()
		if err != nil {
			vc.mu.Lock()
			vc.connected = false
			vc.mu.Unlock()
			vc.signalClosed()
			log.Printf("❌ Failed to receive from worker: %v", err)
			return
		}

		switch messageType {
		case websocket.BinaryMessage:
			select {
			case vc.audioChan <- data:
			default:
				log.Printf("⚠️ Dropping audio chunk from worker: channel full")
			}
		case websocket.TextMessage:
			var msg Message
			if err := json.Unmarshal(data, &msg); err != nil {
				log.Printf("⚠️ Ignoring malformed worker message: %v", err)
				continue
			}
			select {
			case vc.controlChan <- msg:
			default:
				log.Printf("⚠️ Dropping worker control message %s: channel full", msg.Type)
			}
		}
	}
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

func (vc *VoiceChangerClient) SendWebRTCOffer(sdp, sdpType string) error {
	vc.mu.RLock()
	defer vc.mu.RUnlock()

	if !vc.connected || vc.conn == nil {
		return fmt.Errorf("not connected to worker")
	}

	msg := Message{Type: "webrtc_offer", SDP: sdp, SDPType: sdpType}
	return vc.conn.WriteJSON(msg)
}

func (vc *VoiceChangerClient) SendICECandidate(candidate ICECandidate) error {
	vc.mu.RLock()
	defer vc.mu.RUnlock()

	if !vc.connected || vc.conn == nil {
		return fmt.Errorf("not connected to worker")
	}

	msg := Message{Type: "webrtc_ice_candidate", Candidate: &candidate}
	return vc.conn.WriteJSON(msg)
}

func (vc *VoiceChangerClient) AudioChannel() <-chan []byte {
	return vc.audioChan
}

func (vc *VoiceChangerClient) ControlChannel() <-chan Message {
	return vc.controlChan
}

func (vc *VoiceChangerClient) Done() <-chan struct{} {
	return vc.done
}

func (vc *VoiceChangerClient) signalClosed() {
	vc.closeOnce.Do(func() {
		if vc.done != nil {
			close(vc.done)
		}
		if vc.audioChan != nil {
			close(vc.audioChan)
		}
		if vc.controlChan != nil {
			close(vc.controlChan)
		}
	})
}

// Close closes the connection to the worker
func (vc *VoiceChangerClient) Close() error {
	vc.mu.Lock()
	defer vc.mu.Unlock()

	if vc.conn != nil {
		err := vc.conn.Close()
		vc.conn = nil
		vc.connected = false
		vc.signalClosed()
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
	workerAnswers   chan Message
	receiverActive  bool
	writeMu         sync.Mutex
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
		workerAnswers:   make(chan Message, 1),
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
	sendJSON(client, msg)
}

func sendJSON(client *ClientConnection, payload interface{}) error {
	client.writeMu.Lock()
	defer client.writeMu.Unlock()
	return client.Conn.WriteJSON(payload)
}

// sendAudioFormat communicates the PCM stream format to the client UI
func sendAudioFormat(client *ClientConnection) {
	var format AudioFormat
	if client.AudioProcessor != nil {
		format = client.AudioProcessor.format
	} else {
		format = AudioFormat{SampleRate: defaultSampleRate, Channels: defaultChannels, BitsPerSample: bitsPerSample}
	}

	msg := AudioFormatMessage{
		Type:          "audio_format",
		SampleRate:    format.SampleRate,
		Channels:      format.Channels,
		BitsPerSample: format.BitsPerSample,
		Encoding:      "pcm_s16le",
	}

	sendJSON(client, msg)
}

func ensureWorkerConnection(client *ClientConnection) error {
	if err := client.VoiceChanger.Connect(); err != nil {
		return err
	}

	startWorkerReceiver(client)

	return nil
}

func startWorkerReceiver(client *ClientConnection) {
	client.mu.Lock()
	if client.receiverActive {
		client.mu.Unlock()
		return
	}
	client.receiverActive = true
	client.mu.Unlock()

	go func() {
		defer func() {
			client.mu.Lock()
			client.receiverActive = false
			client.mu.Unlock()
		}()
		ContinuousReceiver(client)
	}()
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
	if err := ensureWorkerConnection(client); err != nil {
		log.Printf("Failed to connect to voice changer worker for %s: %v", client.ID, err)
		return err
	}

	client.StreamingActive = true
	log.Printf("✅ Started recording for connection %s", client.ID)
	log.Printf("🎵 Streaming mode enabled for %s", client.ID)

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

	if client.AudioProcessor != nil {
		if err := client.AudioProcessor.StoreWebMChunk(audioData); err != nil {
			log.Printf("Failed to store WebM chunk for %s: %v", client.ID, err)
		}
	}
}

// ContinuousReceiver handles incoming data from voice changer worker
func ContinuousReceiver(client *ClientConnection) {
	log.Printf("🎧 Started continuous receiver for %s", client.ID)

	audioCh := client.VoiceChanger.AudioChannel()
	controlCh := client.VoiceChanger.ControlChannel()
	doneCh := client.VoiceChanger.Done()
	playbackNotified := false
	finished := false

	for audioCh != nil || controlCh != nil || doneCh != nil {
		select {
		case <-doneCh:
			doneCh = nil
		case chunk, ok := <-audioCh:
			if !ok {
				audioCh = nil
				continue
			}

			if len(chunk) == 0 {
				continue
			}

			if len(chunk)%2 != 0 {
				log.Printf("Discarding misaligned PCM chunk for %s: %d bytes", client.ID, len(chunk))
				continue
			}

			sampleCount := len(chunk) / 2
			pcmSamples := make([]int16, sampleCount)
			for i := 0; i < sampleCount; i++ {
				pcmSamples[i] = int16(binary.LittleEndian.Uint16(chunk[i*2:]))
			}

			if client.AudioProcessor != nil {
				client.AudioProcessor.ProcessPCMChunk(pcmSamples)
			}

			chunkCopy := make([]byte, len(chunk))
			copy(chunkCopy, chunk)

			client.mu.Lock()
			client.ProcessedChunks = append(client.ProcessedChunks, chunkCopy)
			client.mu.Unlock()

			if !playbackNotified {
				sendMessage(client, "streaming_started", "Starting audio playback")
				sendAudioFormat(client)
				playbackNotified = true
			}

			log.Printf("🎵 Received processed PCM chunk for %s: %d samples -> %d bytes",
				client.ID, len(pcmSamples), len(chunk))

		case msg, ok := <-controlCh:
			if !ok {
				controlCh = nil
				continue
			}

			switch msg.Type {
			case "webrtc_answer":
				select {
				case client.workerAnswers <- msg:
				default:
					log.Printf("⚠️ Dropping redundant worker answer for %s", client.ID)
				}
			case "webrtc_ice_candidate":
				if msg.Candidate != nil && msg.Candidate.Candidate != "" {
					sendJSON(client, Message{Type: "webrtc_ice_candidate", Candidate: msg.Candidate})
				}
			case "streaming_started":
				sendMessage(client, "streaming_started", msg.Message)
				if !playbackNotified {
					sendAudioFormat(client)
					playbackNotified = true
				}
			case "streaming_completed":
				sendMessage(client, "streaming_completed", msg.Message)
			case "done":
				if !finished {
					sendMessage(client, "done", msg.Message)
					finished = true
				}
			case "error":
				sendMessage(client, "error", msg.Message)
			default:
				log.Printf("ℹ️ Worker message for %s: %s", client.ID, msg.Type)
			}
		}
	}

	if !finished {
		sendMessage(client, "done", "Audio playback finished")
	}
	if !playbackNotified {
		sendMessage(client, "streaming_completed", "Audio streaming completed")
	}
	log.Printf("🎧 Stopped continuous receiver for %s", client.ID)
}

// ProcessWebRTCOffer forwards the SDP offer to the worker and waits for an answer
func ProcessWebRTCOffer(client *ClientConnection, offerSDP, sdpType string) (*Message, error) {
	if offerSDP == "" {
		return nil, fmt.Errorf("empty SDP offer")
	}

	if err := ensureWorkerConnection(client); err != nil {
		return nil, err
	}

	select {
	case <-client.workerAnswers:
	default:
	}

	if err := client.VoiceChanger.SendWebRTCOffer(offerSDP, sdpType); err != nil {
		return nil, err
	}

	select {
	case answer := <-client.workerAnswers:
		return &answer, nil
	case <-time.After(10 * time.Second):
		return nil, fmt.Errorf("timeout waiting for worker answer")
	}
}

// AddWebRTCCandidate adds a remote ICE candidate to the client's session
func AddWebRTCCandidate(client *ClientConnection, candidate ICECandidate) error {
	if err := ensureWorkerConnection(client); err != nil {
		return err
	}

	return client.VoiceChanger.SendICECandidate(candidate)
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

	// Decode buffered audio and forward raw PCM to the worker
	if client.StreamingActive && client.AudioProcessor != nil {
		log.Printf("🎛️ Decoding buffered WebM audio for %s", client.ID)
		pcmSamples, err := client.AudioProcessor.DecodeFullWebMToPCM()
		if err != nil {
			log.Printf("Failed to decode WebM audio for %s: %v", client.ID, err)
			sendMessage(client, "error", "Failed to decode recorded audio")
		} else if len(pcmSamples) > 0 {
			log.Printf("📥 Decoded %d PCM samples for %s", len(pcmSamples), client.ID)
			chunkSamples := client.AudioProcessor.format.SampleRate / 10
			if chunkSamples <= 0 {
				chunkSamples = defaultSampleRate / 10
			}

			for start := 0; start < len(pcmSamples); start += chunkSamples {
				end := start + chunkSamples
				if end > len(pcmSamples) {
					end = len(pcmSamples)
				}

				chunk := pcmSamples[start:end]
				pcmBytes := make([]byte, len(chunk)*2)
				for i, sample := range chunk {
					binary.LittleEndian.PutUint16(pcmBytes[i*2:], uint16(sample))
				}

				if err := client.VoiceChanger.SendAudioChunk(pcmBytes); err != nil {
					log.Printf("Failed to send PCM chunk to voice changer for %s: %v", client.ID, err)
					break
				}

				log.Printf("📤 Sent PCM chunk to voice changer for %s: %d samples", client.ID, len(chunk))

				time.Sleep(10 * time.Millisecond)
			}
		} else {
			log.Printf("No PCM samples decoded for %s", client.ID)
		}
	}

	// Send flush command to voice changer worker
	if client.StreamingActive {
		log.Printf("🔄 Sending flush command to voice changer for %s", client.ID)
		if err := client.VoiceChanger.SendFlush(); err != nil {
			log.Printf("Failed to send flush to voice changer for %s: %v", client.ID, err)
		}

		client.StreamingActive = false
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

	// Save input file as WebM
	inputFile := filepath.Join(sessionDir, "input.webm")
	if err := os.WriteFile(inputFile, inputBuffer, 0644); err != nil {
		log.Printf("Failed to save input file for %s: %v", client.ID, err)
	} else {
		log.Printf("💿 Saved input audio: %s (%d bytes)", inputFile, len(inputBuffer))
	}

	// Save output file if we have processed audio
	if len(outputBuffer) > 0 {
		outputFile := filepath.Join(sessionDir, "output.pcm")
		if err := os.WriteFile(outputFile, outputBuffer, 0644); err != nil {
			log.Printf("Failed to save output file for %s: %v", client.ID, err)
		} else {
			log.Printf("💾 Saved output PCM audio: %s (%d bytes)", outputFile, len(outputBuffer))
		}
	}

	log.Printf("🗂️ Session files saved to: %s", sessionDir)
}
