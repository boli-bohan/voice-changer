#!/usr/bin/env bash
set -e

echo "Building Docker images..."
echo "NOTE: This requires Docker Desktop to be running."
echo "Please start Docker Desktop from /Applications/Docker.app if not running."
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker command not found. Please:"
    echo "   1. Open Docker Desktop from /Applications/Docker.app"
    echo "   2. Wait for it to start (check menu bar icon)"
    echo "   3. Run 'just k8s' again"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running. Please:"
    echo "   1. Open Docker Desktop from /Applications/Docker.app"
    echo "   2. Wait for it to start (check menu bar icon)"
    echo "   3. Run 'just k8s' again"
    exit 1
fi

echo "  📦 Building API server..."
docker build -t voice-changer-api:latest -f api/Dockerfile api/
minikube image load voice-changer-api:latest

echo "  📦 Building worker..."
docker build -t voice-changer-worker:latest -f Dockerfile .
minikube image load voice-changer-worker:latest

echo "  📦 Building frontend..."
docker build -t voice-changer-frontend:latest -f frontend/Dockerfile frontend/
minikube image load voice-changer-frontend:latest

echo "✅ All images built successfully"
