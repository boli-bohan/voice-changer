package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/rs/cors"
)

type offerRequest struct {
	SDP  string `json:"sdp"`
	Type string `json:"type"`
}

type offerResponse struct {
	SDP  string `json:"sdp"`
	Type string `json:"type"`
}

var (
	workerURL  string
	httpClient = &http.Client{Timeout: 15 * time.Second}
)

// init resolves the worker URL from environment and normalises the value.
func init() {
        workerURL = os.Getenv("VOICE_WORKER_URL")
        if workerURL == "" {
                workerURL = "http://127.0.0.1:8001"
        }
        workerURL = strings.TrimRight(workerURL, "/")
}

// handleRoot responds with a basic service descriptor for observability checks.
func handleRoot(w http.ResponseWriter, r *http.Request) {
        response := map[string]interface{}{
                "service":   "Voice Changer API",
                "status":    "running",
                "version":   "2.0.0",
		"timestamp": time.Now().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleHealth reports the status of the API service and downstream worker URL.
func handleHealth(w http.ResponseWriter, r *http.Request) {
        response := map[string]interface{}{
                "status":     "healthy",
                "worker_url": workerURL,
                "timestamp":  time.Now().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(response)
}

// handleConfig exposes runtime configuration values required by the frontend.
func handleConfig(w http.ResponseWriter, r *http.Request) {
        response := map[string]string{
                "worker_url": workerURL,
        }
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(response)
}

// forwardOfferToWorker proxies an SDP offer to the worker and returns the answer.
func forwardOfferToWorker(r *http.Request, offer offerRequest) (*offerResponse, int, error) {
        payload, err := json.Marshal(offer)
        if err != nil {
                return nil, http.StatusInternalServerError, fmt.Errorf("failed to encode offer: %w", err)
        }

	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, workerURL+"/offer", bytes.NewReader(payload))
	if err != nil {
		return nil, http.StatusInternalServerError, fmt.Errorf("failed to create worker request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, http.StatusBadGateway, fmt.Errorf("worker request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, http.StatusBadGateway, fmt.Errorf("failed to read worker response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, resp.StatusCode, fmt.Errorf("worker responded with %s: %s", resp.Status, string(body))
	}

	var answer offerResponse
	if err := json.Unmarshal(body, &answer); err != nil {
		return nil, http.StatusBadGateway, fmt.Errorf("invalid worker response: %w", err)
	}

        return &answer, http.StatusOK, nil
}

// handleWebRTCOffer accepts incoming offers and relays them to the worker service.
func handleWebRTCOffer(w http.ResponseWriter, r *http.Request) {
        var offer offerRequest
        if err := json.NewDecoder(r.Body).Decode(&offer); err != nil {
                http.Error(w, "invalid JSON payload", http.StatusBadRequest)
                return
	}

	answer, status, err := forwardOfferToWorker(r, offer)
	if err != nil {
		log.Printf("❌ Failed to forward offer: %v", err)
		http.Error(w, err.Error(), status)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(answer); err != nil {
                http.Error(w, "failed to encode response", http.StatusInternalServerError)
        }
}

// main wires the HTTP routes, applies CORS, and starts the signalling server.
func main() {
        router := mux.NewRouter()

	router.HandleFunc("/", handleRoot).Methods(http.MethodGet)
	router.HandleFunc("/health", handleHealth).Methods(http.MethodGet)
	router.HandleFunc("/config", handleConfig).Methods(http.MethodGet)
	router.HandleFunc("/webrtc/offer", handleWebRTCOffer).Methods(http.MethodPost)

	corsHandler := cors.New(cors.Options{
		AllowedOrigins:   []string{"http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:30000", "http://127.0.0.1:30000"},
		AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders:   []string{"*"},
		AllowCredentials: true,
	}).Handler(router)

	log.Println("🚀 Voice Changer API Server starting...")
	log.Printf("📡 Signalling endpoint at http://127.0.0.1:8000/webrtc/offer -> %s/offer", workerURL)
	log.Println("⚡ Press Ctrl+C to stop")

	if err := http.ListenAndServe(":8000", corsHandler); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}
