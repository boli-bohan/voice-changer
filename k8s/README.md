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

## Simulate public and private nodes with kind

The repository ships with a kind configuration that exposes the control-plane
node to your host while keeping the worker nodes private. This lets you mimic a
public ingress node for the API and TURN server alongside private worker nodes.

```bash
./setup.sh              # Installs kind alongside the other prerequisites
kind create cluster --config k8s/kind-public-private.yaml

# Label nodes to target them with Helm node affinity rules
kubectl label node kind-control-plane topology=public
kubectl label node kind-worker topology=private
kubectl label node kind-worker2 topology=private
```

Update your Helm values to pin services to the desired nodes. For example, to
schedule the API and TURN pods on the public control plane and all workers on
the private nodes:

```yaml
api:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: topology
              operator: In
              values:
                - public

turn:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: topology
              operator: In
              values:
                - public

worker:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: topology
              operator: In
              values:
                - private

frontend:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: topology
              operator: In
              values:
                - public
```

Apply the chart with the updated values file:

```bash
helm upgrade --install voice-changer ./helm/voice-changer -f my-kind-values.yaml
```

## Expose LoadBalancer services on kind with MetalLB

When you need stable IPs for services such as WebRTC media endpoints you can
pair kind with Calico and MetalLB. MetalLB allocates addresses from your local
network so that `LoadBalancer` services become directly reachable on your LAN.

1. **Create a kind cluster prepared for Calico and MetalLB**

   ```bash
   kind create cluster --name webrtc --config k8s/kind-metallb-config.yaml
   ```

2. **Install Calico for pod networking**

   ```bash
   kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/calico.yaml
   kubectl wait --namespace calico-system --for=condition=available deployment/calico-kube-controllers --timeout=120s
   ```

3. **Install MetalLB**

   ```bash
   kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.14.7/config/manifests/metallb-native.yaml
   kubectl wait --namespace metallb-system --for=condition=available deployment/controller --timeout=120s
   kubectl wait --namespace metallb-system --for=condition=ready pod -l component=speaker --timeout=120s
   ```

4. **Configure an address pool**

   Edit `k8s/metallb-config.yaml` and replace the example `192.168.1.240-192.168.1.250`
   range with IPs that are unused on your LAN. Apply the configuration:

   ```bash
   kubectl apply -f k8s/metallb-config.yaml
   ```

5. **Expose services with LoadBalancer**

   Update your service manifests to use `type: LoadBalancer`. MetalLB assigns
   one of the configured IPs, allowing STUN to discover the address without a
   TURN relay.

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
