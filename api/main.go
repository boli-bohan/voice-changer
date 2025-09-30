package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
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

// WorkerInfo tracks the state of a registered worker instance.
type WorkerInfo struct {
	URL             string    `json:"url"`
	LastSeen        time.Time `json:"last_seen"`
	ConnectionCount int       `json:"connection_count"`
	MaxConnections  int       `json:"max_connections"`
}

// HeartbeatRequest represents a worker heartbeat payload.
type HeartbeatRequest struct {
	WorkerID        string `json:"worker_id"`
	WorkerURL       string `json:"worker_url"`
	ConnectionCount int    `json:"connection_count"`
	MaxConnections  int    `json:"max_connections"`
}

var (
	httpClient = &http.Client{Timeout: 15 * time.Second}
	workersMux sync.RWMutex
	workers    = make(map[string]*WorkerInfo)
)

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

// handleHealth reports the status of the API service and registered workers.
func handleHealth(w http.ResponseWriter, r *http.Request) {
	workersMux.RLock()
	workerCount := len(workers)
	workersMux.RUnlock()

	response := map[string]interface{}{
		"status":       "healthy",
		"worker_count": workerCount,
		"timestamp":    time.Now().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleConfig exposes runtime configuration values required by the frontend.
func handleConfig(w http.ResponseWriter, r *http.Request) {
	workersMux.RLock()
	workerCount := len(workers)
	workersMux.RUnlock()

	response := map[string]interface{}{
		"worker_count":   workerCount,
		"load_balancing": "enabled",
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// forwardOfferToWorker proxies an SDP offer to a specific worker and returns the answer.
func forwardOfferToWorker(r *http.Request, offer offerRequest, workerURL string) (*offerResponse, int, error) {
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

// selectWorker finds an available worker with the lowest connection count.
func selectWorker() (*WorkerInfo, error) {
	workersMux.RLock()
	defer workersMux.RUnlock()

	if len(workers) == 0 {
		return nil, fmt.Errorf("no workers available")
	}

	var bestWorker *WorkerInfo
	minLoad := int(^uint(0) >> 1) // max int

	for id, info := range workers {
		// Only consider workers that have capacity
		if info.ConnectionCount < info.MaxConnections {
			if info.ConnectionCount < minLoad {
				minLoad = info.ConnectionCount
				bestWorker = info
				log.Printf("🎯 Candidate worker %s: %d/%d connections",
					id, info.ConnectionCount, info.MaxConnections)
			}
		}
	}

	if bestWorker == nil {
		return nil, fmt.Errorf("all workers at capacity")
	}

	return bestWorker, nil
}

// handleHeartbeat receives and processes heartbeat messages from workers.
func handleHeartbeat(w http.ResponseWriter, r *http.Request) {
	var hb HeartbeatRequest
	if err := json.NewDecoder(r.Body).Decode(&hb); err != nil {
		http.Error(w, "invalid payload", http.StatusBadRequest)
		return
	}

	workersMux.Lock()
	workers[hb.WorkerID] = &WorkerInfo{
		URL:             hb.WorkerURL,
		LastSeen:        time.Now(),
		ConnectionCount: hb.ConnectionCount,
		MaxConnections:  hb.MaxConnections,
	}
	workersMux.Unlock()

	// log.Printf("💓 Heartbeat from worker %s: %d/%d connections",
	//	hb.WorkerID, hb.ConnectionCount, hb.MaxConnections)

	w.WriteHeader(http.StatusOK)
}

// handleWorkers returns the current worker registry for debugging.
func handleWorkers(w http.ResponseWriter, r *http.Request) {
	workersMux.RLock()
	defer workersMux.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(workers)
}

// cleanupStaleWorkers removes workers that haven't sent a heartbeat recently.
func cleanupStaleWorkers() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		workersMux.Lock()
		now := time.Now()
		for id, info := range workers {
			if now.Sub(info.LastSeen) > 30*time.Second {
				log.Printf("⚠️ Removing stale worker: %s (last seen: %v)", id, info.LastSeen)
				delete(workers, id)
			}
		}
		workersMux.Unlock()
	}
}

// handleWebRTCOffer accepts incoming offers and relays them to an available worker.
func handleWebRTCOffer(w http.ResponseWriter, r *http.Request) {
	var offer offerRequest
	if err := json.NewDecoder(r.Body).Decode(&offer); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}

	worker, err := selectWorker()
	if err != nil {
		log.Printf("❌ Failed to select worker: %v", err)
		http.Error(w, fmt.Sprintf("no available workers: %v", err), http.StatusServiceUnavailable)
		return
	}

	log.Printf("📤 Forwarding offer to worker at %s", worker.URL)

	answer, status, err := forwardOfferToWorker(r, offer, worker.URL)
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
	// Start worker cleanup goroutine
	go cleanupStaleWorkers()

	router := mux.NewRouter()

	router.HandleFunc("/", handleRoot).Methods(http.MethodGet)
	router.HandleFunc("/health", handleHealth).Methods(http.MethodGet)
	router.HandleFunc("/config", handleConfig).Methods(http.MethodGet)
	router.HandleFunc("/workers", handleWorkers).Methods(http.MethodGet)
	router.HandleFunc("/heartbeat", handleHeartbeat).Methods(http.MethodPost)
	router.HandleFunc("/webrtc/offer", handleWebRTCOffer).Methods(http.MethodPost)

	corsHandler := cors.New(cors.Options{
		AllowedOrigins:   []string{"http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:30000", "http://127.0.0.1:30000"},
		AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders:   []string{"*"},
		AllowCredentials: true,
	}).Handler(router)

	log.Println("🚀 Voice Changer API Server starting...")
	log.Println("📡 Signalling endpoint at http://127.0.0.1:8000/webrtc/offer")
	log.Println("🔄 Load balancing enabled with dynamic worker discovery")
	log.Println("💓 Heartbeat endpoint at http://127.0.0.1:8000/heartbeat")
	log.Println("⚡ Press Ctrl+C to stop")

	if err := http.ListenAndServe(":8000", corsHandler); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}
