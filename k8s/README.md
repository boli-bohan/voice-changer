# Kubernetes Deployment for Voice Changer

This directory contains Kubernetes manifests for deploying the voice changer application locally using minikube.

## Prerequisites

- **minikube**: `brew install minikube` ✅ Installed
- **kubectl**: `brew install kubectl` ✅ Installed
- **Docker Desktop**: `brew install --cask docker` ✅ Installed (needs to be started)
- **QEMU**: `brew install qemu` ✅ Installed

## Quick Start

1. **Start Docker Desktop**:
   ```bash
   open /Applications/Docker.app
   ```
   Wait for Docker to fully start (check menu bar icon shows "Running")

2. **Deploy to Kubernetes**:
   ```bash
   just k8s
   ```

This command will:
- Start minikube if not running
- Build Docker images for all three services
- Load images into minikube
- Apply Kubernetes manifests
- Wait for pods to be ready
- Display access URLs

## Access the Application

After deployment, the services are accessible via minikube's IP:

- **API Server**: `http://$(minikube ip):30800`
- **Frontend**: `http://$(minikube ip):30173`

### Port Forwarding (Alternative Access)

For localhost access, run in separate terminals:

```bash
# API Server on localhost:8000
kubectl port-forward svc/voice-changer-api 8000:8000

# Frontend on localhost:5173
kubectl port-forward svc/voice-changer-frontend 5173:80
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Minikube Cluster                      │
│                                                          │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │   Frontend    │  │  API Server  │  │   Worker    │  │
│  │ (nginx:80)    │  │   (Go:8000)  │  │ (Python:8001│  │
│  │               │  │              │  │              │  │
│  └───────────────┘  └──────────────┘  └─────────────┘  │
│         │                  │                  │         │
│  NodePort:30173    NodePort:30800      ClusterIP       │
│                            │                  │         │
│                            └──────────────────┘         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

- **Worker**: ClusterIP service (internal only)
- **API**: NodePort service (external access on port 30800)
- **Frontend**: NodePort service (external access on port 30173)

## Available Commands

### Deployment
```bash
just k8s              # Deploy all services
just k8s-down         # Remove all K8s resources
just k8s-status       # Check deployment status
```

### Logs
```bash
just k8s-logs                    # List pods
just k8s-logs <pod-name>        # Follow logs for specific pod
just k8s-logs all               # Show logs for all pods
```

### Rebuild
```bash
just k8s-rebuild api      # Rebuild and restart API server
just k8s-rebuild worker   # Rebuild and restart worker
just k8s-rebuild frontend # Rebuild and restart frontend
```

## Testing

Run the test client against the K8s deployment:

```bash
# Via NodePort
uv run test_client.py --api-base http://$(minikube ip):30800

# Via port-forward (in another terminal, run kubectl port-forward first)
uv run test_client.py --api-base http://localhost:8000
```

## Troubleshooting

### Pods not starting
```bash
kubectl get pods -l app=voice-changer
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### Image pull errors
Images use `imagePullPolicy: Never` so they must be loaded into minikube:
```bash
./scripts/build-images.sh
```

### Docker not available
Make sure Docker Desktop is running:
```bash
open /Applications/Docker.app
docker info  # Should succeed when Docker is ready
```

### Minikube issues
```bash
minikube stop
minikube delete
minikube start --driver=qemu
```

## Configuration

### Environment Variables
- **API Server**: `VOICE_WORKER_URL=http://voice-changer-worker:8001`

### Resource Limits
- **Worker**: 512Mi-1Gi memory, 500m-1000m CPU
- **API**: 128Mi-256Mi memory, 100m-500m CPU
- **Frontend**: 64Mi-128Mi memory, 50m-200m CPU

### Health Probes
All services have liveness and readiness probes configured:
- **Worker**: `/health` endpoint
- **API**: `/health` endpoint
- **Frontend**: `/` endpoint
