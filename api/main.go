package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/rs/cors"
)

// Global connection manager
var connManager = NewConnectionManager()

// WebSocket upgrader
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		origin := r.Header.Get("Origin")
		return origin == "http://localhost:5173" || origin == "http://127.0.0.1:5173"
	},
}

// handleWebSocket handles WebSocket connections
func handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	client := connManager.AddConnection(conn)
	defer connManager.RemoveConnection(client.ID)

	for {
		messageType, data, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error for %s: %v", client.ID, err)
			}
			break
		}

		if messageType == websocket.TextMessage {
			var msg Message
			if err := json.Unmarshal(data, &msg); err != nil {
				log.Printf("Invalid JSON received from %s", client.ID)
				sendMessage(client, "error", "Invalid JSON format")
				continue
			}

			switch msg.Type {
			case "start_recording":
				log.Printf("🎙️ Received start_recording for %s", client.ID)
				if err := StartRecording(client); err != nil {
					sendMessage(client, "error", "Failed to start recording")
				} else {
					sendMessage(client, "recording_started", "Recording started")
					log.Printf("✅ Sent recording_started confirmation to %s", client.ID)
				}

			case "stop_recording":
				log.Printf("⏹️ Received stop_recording for %s", client.ID)
				sendMessage(client, "processing_started", "Processing audio...")
				log.Printf("📤 Sent processing_started message to %s", client.ID)

				if err := StopRecordingAndProcess(client); err != nil {
					log.Printf("❌ Audio processing failed for %s: %v", client.ID, err)
					sendMessage(client, "error", "Audio processing failed")
				} else {
					log.Printf("✅ Audio processing completed successfully for %s", client.ID)
				}

			default:
				log.Printf("Unknown message type: %s", msg.Type)
			}

		} else if messageType == websocket.BinaryMessage {
			log.Printf("Received audio chunk for %s: %d bytes", client.ID, len(data))
			AddAudioChunk(client, data)
		}
	}
}

// handleRoot serves the root endpoint with API information
func handleRoot(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"service":   "Voice Changer API",
		"status":    "running",
		"version":   "0.1.0",
		"timestamp": time.Now().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleHealth serves the health check endpoint
func handleHealth(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"status":             "healthy",
		"active_connections": connManager.GetConnectionCount(),
		"temp_directory":     "temp_audio",
		"timestamp":          time.Now().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func main() {
	// Create temp directory
	if err := os.MkdirAll("temp_audio", 0755); err != nil {
		log.Printf("Warning: Failed to create temp_audio directory: %v", err)
	}

	// Create router
	router := mux.NewRouter()

	// Routes
	router.HandleFunc("/", handleRoot).Methods("GET")
	router.HandleFunc("/health", handleHealth).Methods("GET")
	router.HandleFunc("/ws", handleWebSocket)

	// CORS configuration
	c := cors.New(cors.Options{
		AllowedOrigins:   []string{"http://localhost:5173", "http://127.0.0.1:5173"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"*"},
		AllowCredentials: true,
	})

	handler := c.Handler(router)

	// Start server
	log.Println("🚀 Voice Changer API Server starting...")
	log.Println("📡 Server will be available at http://127.0.0.1:8000")
	log.Println("🔌 WebSocket endpoint at ws://127.0.0.1:8000/ws")
	log.Println("⚡ Press Ctrl+C to stop")

	if err := http.ListenAndServe(":8000", handler); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}
