# Voice Changer

A real-time voice changing application with push-to-talk functionality.

## Overview

This project implements a push-to-talk voice changer where users can:

1. Press and hold a button to start recording
2. Release the button to stop recording and automatically process the audio
3. Hear the pitch-shifted version of their voice played back

## Architecture

![Architecture](assets/arch.png)

## Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) (Python package installer)
- [Just](https://github.com/casey/just) (Command runner)
- go
- helm
- k8s
- minikube

## Quick Start

```
just install # install dependencies
just build # build the go api server
just up # start frontend, api server, and voice changer worker
```

## Testing

```
uv sync --all-extras # install additional test dependencies
just up # start services
uv run test_client.py # tests the api with a sample audio file data/test_input.wav
```

To exercise the frontend end-to-end flow in a real browser, first install the Playwright
Chromium binary and then run the automated test while the stack is up:

```
uv run playwright install chromium
uv run python test_browser.py --record-output temp_audio/remote.webm
```

## Kubernetes/Helm Deployment

For production-like deployments, you can run the application in Kubernetes using Helm:

### Prerequisites

- [Minikube](https://minikube.sigs.k8s.io/) (or any Kubernetes cluster)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Helm](https://helm.sh/) (v3+)

### Deploy with Helm

```bash
just helm-install  # Build images and install Helm chart
```

This will:
1. Start minikube (if not running)
2. Build Docker images for all components (API, Worker, Frontend)
3. Install the `voice-changer` Helm release
4. Wait for all pods to be ready

### Access the Application

Services are exposed via NodePort:
- **Frontend**: `http://$(minikube ip):30000`
- **API Server**: `http://$(minikube ip):30900`
- **Worker**: `http://$(minikube ip):30901`

Alternatively, use port-forwarding for localhost access:
```bash
kubectl port-forward svc/voice-changer-api 9000:8000
kubectl port-forward svc/voice-changer-worker 9001:8001
kubectl port-forward svc/voice-changer-frontend 3000:80
```

### Helm Management

```bash
just helm-status      # Check deployment status
just helm-upgrade     # Apply configuration changes
just helm-template    # Preview rendered templates
just helm-uninstall   # Remove deployment
```

### Configuration

Customize deployment by editing `helm/voice-changer/values.yaml`:
- Replica counts (API: 1, Worker: 3, Frontend: 1)
- Resource limits/requests
- NodePort numbers
- Health check settings
