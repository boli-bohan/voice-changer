set shell := ["bash", "-cu"]

default: help

help:
	@just --list

install:
	@echo "Installing Go dependencies..."
	cd api && go mod tidy
	@echo "Installing Python worker dependencies..."
	uv sync
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

lint:
	@echo "Linting Go API server..."
	cd api && go vet ./... && go fmt ./...
	@echo "Linting Python worker..."
	uv run ruff check .
	@echo "Linting frontend..."
	@echo "Frontend linting temporarily disabled - ESLint config needs fixing"
	# cd frontend && npm run lint

format:
	@echo "Formatting Go API server..."
	cd api && go fmt ./...
	@echo "Formatting Python worker..."
	uv run ruff format .
	@echo "Formatting frontend..."
	cd frontend && npm run format

up:
	@echo "Stopping any existing services..."
	@just down
	@echo "Starting all services..."
	@echo "Go API Server will be available at http://127.0.0.1:8000"
	@echo "Voice Changer Worker will be available at http://127.0.0.1:8001"
	@echo "Frontend will be available at http://localhost:5173"
	@echo "Press Ctrl+C to stop all services"
	cd api && go run . &
	uv run uvicorn voice_changer:app --reload --host 127.0.0.1 --port 8001 --log-level info &
	cd frontend && npm run dev &
	wait

down:
	@echo "Stopping all services..."
	@pkill -f "go run main.go" 2>/dev/null || true
	@pkill -f "main.go" 2>/dev/null || true
	@pkill -f "uvicorn voice_changer:app" 2>/dev/null || true
	@pkill -f "voice_changer:app" 2>/dev/null || true
	@pkill -f "npm run dev" 2>/dev/null || true
	@pkill -f "vite.*--port 5173" 2>/dev/null || true
	@pkill -f "vite" 2>/dev/null || true
	@lsof -ti:8000 2>/dev/null | xargs -r kill 2>/dev/null || true
	@lsof -ti:8001 2>/dev/null | xargs -r kill 2>/dev/null || true
	@lsof -ti:5173 2>/dev/null | xargs -r kill 2>/dev/null || true
	@echo "Services stopped"

status:
	@echo "Checking service status..."
	@echo "Port 8000 (API Server):"
	@lsof -ti:8000 2>/dev/null && echo "  ✅ Running" || echo "  ❌ Not running"
	@echo "Port 8001 (Voice Changer Worker):"
	@lsof -ti:8001 2>/dev/null && echo "  ✅ Running" || echo "  ❌ Not running"
	@echo "Port 5173 (Frontend):"
	@lsof -ti:5173 2>/dev/null && echo "  ✅ Running" || echo "  ❌ Not running"
	@echo ""
	@echo "Running processes:"
	@pgrep -fl "uvicorn|vite|npm run dev" || echo "  No related processes found"

worker:
	@echo "Starting Voice Changer PCM Worker only..."
	@echo "Worker will be available at http://127.0.0.1:8001"
	uv run uvicorn voice_changer:app --reload --host 127.0.0.1 --port 8001 --log-level info

api:
	@echo "Starting Go API Server only..."
	@echo "Go API Server will be available at http://127.0.0.1:8000"
	cd api && go run .

build:
	@echo "Building Go API server..."
	cd api && go build -o voice-changer-api .
	@echo "Build complete: api/voice-changer-api"

clean:
	@echo "Cleaning temporary files..."
	rm -rf temp_audio/
	rm -rf frontend/dist/
	rm -rf frontend/node_modules/.vite/
	rm -rf api/voice-changer-api
	@echo "Cleanup complete"

test:
	@echo "Running backend tests..."
	uv run pytest
	@echo "Running frontend tests..."
	cd frontend && npm run test