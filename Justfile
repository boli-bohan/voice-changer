set shell := ["bash", "-cu"]

default: help

help:
	@just --list

install:
	@echo "Installing Go dependencies..."
	cd api && go mod tidy
	@echo "Installing Python dependencies (including dev/test dependencies)..."
	uv sync --all-extras
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

# Local Development (no Kubernetes)
up:
	@echo "Stopping any existing services..."
	@just down
	@echo "Starting all services (manual mode, no k8s)..."
	@echo ""
	@echo "🌐 Services will be available at:"
	@echo "  • Go API Server: http://127.0.0.1:8000"
	@echo "  • Voice Changer Worker: http://127.0.0.1:8001"
	@echo "  • Frontend: http://localhost:5173"
	@echo ""
	@echo "Press Ctrl+C to stop all services"
	@echo ""
	@bash -c ' \
		trap "echo \"\" && echo \"Stopping services...\" && just down && exit" INT TERM; \
		cd api && go run . 2>&1 | sed "s/^/[API] /" & \
		API_URL=http://127.0.0.1:8000 WORKER_PORT=8001 uv run uvicorn voice_changer:app --reload --host 127.0.0.1 --port 8001 --log-level info 2>&1 | sed "s/^/[WORKER] /" & \
		cd frontend && npm run dev 2>&1 | sed "s/^/[FRONTEND] /" \
	'

up-echo:
	@echo "Stopping any existing services..."
	@just down
	@echo "Starting all services with echo worker (manual mode, no k8s)..."
	@echo ""
	@echo "🌐 Services will be available at:"
	@echo "  • Go API Server: http://127.0.0.1:8000"
	@echo "  • Echo Worker: http://127.0.0.1:8001"
	@echo "  • Frontend: http://localhost:5173"
	@echo ""
	@echo "Press Ctrl+C to stop all services"
	@echo ""
	@bash -c ' \
		trap "echo \"\" && echo \"Stopping services...\" && just down && exit" INT TERM; \
		cd api && go run . 2>&1 | sed "s/^/[API] /" & \
		API_URL=http://127.0.0.1:8000 WORKER_PORT=8001 uv run uvicorn echo:app --reload --host 127.0.0.1 --port 8001 --log-level info 2>&1 | sed "s/^/[ECHO] /" & \
		cd frontend && npm run dev 2>&1 | sed "s/^/[FRONTEND] /" \
	'

down:
	@echo "Stopping all services..."
	@pkill -f "[g]o run main.go" 2>/dev/null || true
	@pkill -f "[m]ain.go" 2>/dev/null || true
	@pkill -f "[u]vicorn voice_changer:app" 2>/dev/null || true
	@pkill -f "[u]vicorn echo:app" 2>/dev/null || true
	@pkill -f "[v]oice_changer:app" 2>/dev/null || true
	@pkill -f "[n]pm run dev" 2>/dev/null || true
	@pkill -f "[v]ite.*--port 5173" 2>/dev/null || true
	@pkill -f "[v]ite" 2>/dev/null || true
	@lsof -ti:8000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
	@lsof -ti:8001 2>/dev/null | xargs -r kill -9 2>/dev/null || true
	@lsof -ti:5173 2>/dev/null | xargs -r kill -9 2>/dev/null || true
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
	API_URL=http://127.0.0.1:8000 WORKER_PORT=8001 uv run uvicorn voice_changer:app --reload --host 127.0.0.1 --port 8001 --log-level info

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
	rm -rf api/temp_audio/
	@echo "Cleanup complete"

test:
        @echo "Running backend tests..."
        uv run pytest
        @echo "Running frontend tests..."
        cd frontend && npm run test

test-browser *args:
        @echo "Running headless browser integration test..."
        uv run python test_browser.py {{args}}

# Kubernetes Deployment (alternative to 'just up')

helm:
	@echo "🚀 Preparing voice-changer Helm deployment..."
	@driver="${MINIKUBE_DRIVER:-docker}"; \
		if ! minikube status >/dev/null 2>&1; then \
			echo "Starting minikube with driver '${driver}' (expanded node port range)..."; \
			minikube start --driver="${driver}" --extra-config=apiserver.service-node-port-range=1-65535; \
		fi
	@echo "✅ Minikube is running"
	@echo ""
	@if [ "${SKIP_BUILD_IMAGES:-0}" != "1" ]; then \
		echo "Building Docker images inside minikube..."; \
		./scripts/build-images.sh; \
	else \
		echo "Skipping image build (SKIP_BUILD_IMAGES=${SKIP_BUILD_IMAGES})"; \
	fi
	@echo ""
	@echo "Installing Helm release..."
	helm upgrade --install voice-changer helm/voice-changer
	@echo ""
	@echo "Waiting for pods to become ready..."
	kubectl wait --for=condition=ready pod -l app=voice-changer --timeout=180s
	@echo "✅ Helm deployment complete"
	@echo ""
	@echo "🌐 Access services via NodePort:"
	@echo "  Frontend: http://$$(minikube ip):3000"
	@echo "  API:      http://$$(minikube ip):9000"
	@echo "  Worker:   http://$$(minikube ip):9001"

helm-status:
	@echo "📊 Helm Deployment Status"
	@echo ""
	@if ! minikube status >/dev/null 2>&1; then \
		echo "❌ Minikube is not running"; \
		exit 0; \
	fi
	@helm status voice-changer 2>/dev/null || echo "❌ Helm release 'voice-changer' not found"
	@echo ""
	@echo "Deployments:"; kubectl get deployments -l app=voice-changer 2>/dev/null || echo "No deployments found"
	@echo ""
	@echo "Pods:"; kubectl get pods -l app=voice-changer 2>/dev/null || echo "No pods found"
	@echo ""
	@echo "Services:"; kubectl get services -l app=voice-changer 2>/dev/null || echo "No services found"

helm-uninstall:
	@echo "🛑 Uninstalling voice-changer Helm release..."
	helm uninstall voice-changer || echo "Release not found"
	@echo "✅ Helm release removed"
