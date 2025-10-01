# Voice Changer Helm Chart

This chart deploys the three core services that make up the voice changer stack:

- **API** – Go signalling service responsible for forwarding SDP offers to the worker.
- **Worker** – Python FastAPI service that applies the pitch-shift transform and streams
  audio back to the caller.
- **Frontend** – Vite/React interface that exposes the push-to-talk controls.

## Usage

```bash
# Build container images inside Minikube (optional, see SKIP_BUILD_IMAGES)
./scripts/build-images.sh

# Install or upgrade the release
just helm
```

The helper target starts Minikube if needed, loads the locally-built images, and
calls `helm upgrade --install voice-changer helm/voice-changer`.

Inspect the deployed objects with:

```bash
just helm-status
```

Remove the release when you are done:

```bash
just helm-uninstall
```

## Configuration

Override values via a custom `values.yaml` file or `--set` flags. Key options:

- `api.service.nodePort`, `worker.service.nodePort`, `frontend.service.nodePort`
  – Adjust NodePort exposure for each service when running on Minikube.
- `worker.replicaCount` – Number of worker pods to run (defaults to 3).
- `worker.env.apiUrl` – Internal URL the worker uses to reach the API service.
- `frontend.image`, `api.image`, `worker.image` – Override repository/tag when
  publishing images to a remote registry.

See `values.yaml` for the complete list of configurable settings.

