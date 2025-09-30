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
		API_PID=$$!; \
		API_URL=http://127.0.0.1:8000 uv run uvicorn voice_changer:app --reload --host 127.0.0.1 --port 8001 --log-level info 2>&1 | sed "s/^/[WORKER] /" & \
		WORKER_PID=$$!; \
		cd frontend && npm run dev 2>&1 | sed "s/^/[FRONTEND] /" & \
		FRONTEND_PID=$$!; \
		wait $${API_PID} $${WORKER_PID} $${FRONTEND_PID}; \
	'

down:
	@echo "Stopping all services..."
	@pkill -f "[g]o run main.go" 2>/dev/null || true
	@pkill -f "[m]ain.go" 2>/dev/null || true
	@pkill -f "[u]vicorn voice_changer:app" 2>/dev/null || true
	@pkill -f "[v]oice_changer:app" 2>/dev/null || true
	@pkill -f "[n]pm run dev" 2>/dev/null || true
	@pkill -f "[v]ite.*--port 5173" 2>/dev/null || true
	@pkill -f "[v]ite" 2>/dev/null || true
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
	API_URL=http://127.0.0.1:8000 uv run uvicorn voice_changer:app --reload --host 127.0.0.1 --port 8001 --log-level info

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
k8s:
	@echo "🚀 Starting Kubernetes deployment..."
	@echo "Checking minikube status..."
	@minikube status >/dev/null 2>&1 || (echo "Starting minikube..." && minikube start --driver=qemu)
	@echo "✅ Minikube is running"
	@echo ""
	@echo "Building Docker images in minikube..."
	@./scripts/build-images.sh
	@echo ""
	@echo "Applying Kubernetes manifests..."
	kubectl apply -f k8s/
	@echo ""
	@echo "Waiting for pods to be ready..."
	kubectl wait --for=condition=ready pod -l app=voice-changer --timeout=120s
	@echo ""
	@echo "✅ Deployment complete!"
	@echo ""
	@echo "📊 Status:"
	@kubectl get pods -l app=voice-changer
	@echo ""
	@echo "🌐 Access URLs (via NodePort):"
	@echo "  API Server: http://$$(minikube ip):30900"
	@echo "  Worker:     http://$$(minikube ip):30901"
	@echo "  Frontend:   http://$$(minikube ip):30000"
	@echo ""
	@echo "💡 To access via localhost, run in separate terminals:"
	@echo "  kubectl port-forward svc/voice-changer-api 9000:8000"
	@echo "  kubectl port-forward svc/voice-changer-worker 9001:8001"
	@echo "  kubectl port-forward svc/voice-changer-frontend 3000:80"
	@echo ""
	@echo "📝 View logs with: just k8s-logs"
	@echo "🛑 Stop with: just k8s-down"

k8s-down:
	@echo "🛑 Stopping Kubernetes deployment..."
	kubectl delete -f k8s/ || true
	@echo "✅ Kubernetes resources deleted"

k8s-logs *args:
	@if [ -z "{{args}}" ]; then \
		echo "📝 Available pods:"; \
		kubectl get pods -l app=voice-changer; \
		echo ""; \
		echo "Usage: just k8s-logs [pod-name]"; \
		echo "Or view all logs with: just k8s-logs all"; \
	elif [ "{{args}}" = "all" ]; then \
		echo "📝 Logs for all voice-changer pods:"; \
		for pod in $$(kubectl get pods -l app=voice-changer -o jsonpath='{.items[*].metadata.name}'); do \
			echo ""; \
			echo "=== $$pod ==="; \
			kubectl logs $$pod --tail=50; \
		done; \
	else \
		kubectl logs {{args}} -f; \
	fi

k8s-status:
	@echo "📊 Kubernetes Status"
	@echo ""
	@echo "Cluster:"
	@minikube status || echo "❌ Minikube is not running"
	@echo ""
	@echo "Deployments:"
	@kubectl get deployments -l app=voice-changer 2>/dev/null || echo "No deployments found"
	@echo ""
	@echo "Pods:"
	@kubectl get pods -l app=voice-changer 2>/dev/null || echo "No pods found"
	@echo ""
	@echo "Services:"
	@kubectl get services -l app=voice-changer 2>/dev/null || echo "No services found"
	@echo ""
	@echo "Access URLs (via NodePort):"
	@echo "  API Server: http://$$(minikube ip):30900" 2>/dev/null || echo "  ❌ Minikube not running"
	@echo "  Worker:     http://$$(minikube ip):30901" 2>/dev/null || echo "  ❌ Minikube not running"
	@echo "  Frontend:   http://$$(minikube ip):30000" 2>/dev/null || echo "  ❌ Minikube not running"

k8s-rebuild component:
	@echo "🔄 Rebuilding {{component}}..."
	@bash -c 'eval $$(minikube -p minikube docker-env) && \
		if [ "{{component}}" = "api" ]; then \
			docker build -t voice-changer-api:latest -f api/Dockerfile api/ && \
			kubectl rollout restart deployment/voice-changer-api; \
		elif [ "{{component}}" = "worker" ]; then \
			docker build -t voice-changer-worker:latest -f Dockerfile . && \
			kubectl rollout restart deployment/voice-changer-worker; \
		elif [ "{{component}}" = "frontend" ]; then \
			docker build -t voice-changer-frontend:latest -f frontend/Dockerfile frontend/ && \
			kubectl rollout restart deployment/voice-changer-frontend; \
		else \
			echo "❌ Unknown component: {{component}}"; \
			echo "Valid components: api, worker, frontend"; \
			exit 1; \
		fi'
	@echo "✅ {{component}} rebuild complete"

# Helm Deployment (alternative to 'kubectl apply -f k8s/')
helm-install:
	@echo "🚀 Installing voice-changer Helm chart..."
	@echo "Checking minikube status..."
	@minikube status >/dev/null 2>&1 || (echo "Starting minikube..." && minikube start --driver=qemu)
	@echo "✅ Minikube is running"
	@echo ""
	@echo "Building Docker images in minikube..."
	@./scripts/build-images.sh
	@echo ""
	@echo "Installing Helm chart..."
	helm install voice-changer helm/voice-changer
	@echo ""
	@echo "Waiting for pods to be ready..."
	kubectl wait --for=condition=ready pod -l app=voice-changer --timeout=120s
	@echo ""

helm-upgrade:
	@echo "🔄 Upgrading voice-changer Helm release..."
	helm upgrade voice-changer helm/voice-changer
	@echo ""
	@echo "Waiting for pods to be ready..."
	kubectl wait --for=condition=ready pod -l app=voice-changer --timeout=120s
	@echo "✅ Upgrade complete!"

helm-uninstall:
	@echo "🛑 Uninstalling voice-changer Helm release..."
	helm uninstall voice-changer || true
	@echo "✅ Helm release uninstalled"

helm-template:
	@echo "📝 Rendering Helm templates..."
	helm template voice-changer helm/voice-changer

helm-status:
	@echo "📊 Helm Release Status"
	@echo ""
	@helm status voice-changer 2>/dev/null || echo "❌ Helm release 'voice-changer' not found"
	@echo ""
	@echo "Deployments:"
	@kubectl get deployments -l app=voice-changer 2>/dev/null || echo "No deployments found"
	@echo ""
	@echo "Pods:"
	@kubectl get pods -l app=voice-changer 2>/dev/null || echo "No pods found"
	@echo ""
	@echo "Services:"
	@kubectl get services -l app=voice-changer 2>/dev/null || echo "No services found"
