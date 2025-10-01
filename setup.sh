#!/usr/bin/env bash
# shellcheck shell=bash

set -euo pipefail

DOCKER_AVAILABLE=1

INSTALL_ROOT="${HOME}/.local"
BIN_DIR="${INSTALL_ROOT}/bin"
CACHE_DIR="${INSTALL_ROOT}/cache/setup"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${BIN_DIR}" "${CACHE_DIR}"

export PATH="${BIN_DIR}:${PATH}"

require_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "Missing required command: ${cmd}. Please install it in your environment." >&2
        exit 1
    fi
}

check_prerequisites() {
    local dependencies=(curl python3 tar)

    for dependency in "${dependencies[@]}"; do
        require_command "${dependency}"
    done

    if ! command -v docker >/dev/null 2>&1; then
        DOCKER_AVAILABLE=0
        cat <<'WARNING' >&2
⚠️ Docker was not detected. Install Docker to use the default Minikube driver and run --verify.
You can override the Minikube driver by exporting MINIKUBE_DRIVER before running setup.sh.
WARNING
    fi
}

append_path_hint() {
    local profile="$1"
    local marker="# voice-changer setup PATH"

    if [[ -f "${profile}" ]] && grep -Fq "${marker}" "${profile}"; then
        return 0
    fi

    cat <<PROFILE >>"${profile}"
${marker}
export PATH="\${HOME}/.local/bin:\${HOME}/bin:\${PATH}"
PROFILE
}

ensure_path_exports() {
    case "${SHELL:-}" in
        */zsh)
            append_path_hint "${HOME}/.zshrc"
            ;;
        *)
            append_path_hint "${HOME}/.bashrc"
            ;;
    esac
}

install_kubectl() {
    if command -v kubectl >/dev/null 2>&1; then
        return 0
    fi

    local version
    version="${KUBECTL_VERSION:-$(curl -fsSL https://dl.k8s.io/release/stable.txt)}"
    local url="https://dl.k8s.io/release/${version}/bin/linux/amd64/kubectl"

    curl -fsSLo "${BIN_DIR}/kubectl" "${url}"
    chmod +x "${BIN_DIR}/kubectl"
}

install_minikube() {
    if command -v minikube >/dev/null 2>&1; then
        return 0
    fi

    local version
    version="${MINIKUBE_VERSION:-latest}"
    local url="https://storage.googleapis.com/minikube/releases/${version}/minikube-linux-amd64"

    curl -fsSLo "${BIN_DIR}/minikube" "${url}"
    chmod +x "${BIN_DIR}/minikube"
}

install_helm() {
    if command -v helm >/dev/null 2>&1; then
        return 0
    fi

    local version
    version="${HELM_VERSION:-latest}"
    if [[ "${version}" == "latest" ]]; then
        version=$(curl -fsSL https://api.github.com/repos/helm/helm/releases/latest \
            | python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])')
    fi

    local archive="${CACHE_DIR}/helm-${version}.tar.gz"
    local extract_dir
    extract_dir="${CACHE_DIR}/helm-${version}"

    rm -rf "${extract_dir}"
    mkdir -p "${extract_dir}"
    curl -fsSLo "${archive}" "https://get.helm.sh/helm-${version}-linux-amd64.tar.gz"
    tar -xzf "${archive}" -C "${extract_dir}" --strip-components=1
    mv "${extract_dir}/helm" "${BIN_DIR}/helm"
    chmod +x "${BIN_DIR}/helm"
}

install_just() {
    if command -v just >/dev/null 2>&1; then
        return 0
    fi

    local version
    version="${JUST_VERSION:-latest}"
    if [[ "${version}" == "latest" ]]; then
        version=$(curl -fsSL https://api.github.com/repos/casey/just/releases/latest \
            | python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])')
    fi

    local machine
    machine="$(uname -m)"
    local target=""
    case "${machine}" in
        x86_64|amd64)
            target="x86_64-unknown-linux-musl"
            ;;
        aarch64|arm64)
            target="aarch64-unknown-linux-musl"
            ;;
        *)
            echo "Unsupported architecture for just: ${machine}" >&2
            exit 1
            ;;
    esac

    local archive
    archive="${CACHE_DIR}/just-${version}-${target}.tar.gz"
    local extract_dir
    extract_dir="${CACHE_DIR}/just-${version}-${target}"
    local asset
    asset="just-${version}-${target}.tar.gz"

    rm -rf "${extract_dir}"
    mkdir -p "${extract_dir}"
    curl -fsSLo "${archive}" "https://github.com/casey/just/releases/download/${version}/${asset}"
    tar -xzf "${archive}" -C "${extract_dir}"
    install -m 0755 "${extract_dir}/just" "${BIN_DIR}/just"
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi

    curl -fsSL https://astral.sh/uv/install.sh | sh -s -- --bin-dir "${BIN_DIR}" --quiet
}

install_python_dependencies() {
    if [[ "${SKIP_PYTHON_DEPS:-0}" == "1" ]]; then
        echo "Skipping Python dependency installation (SKIP_PYTHON_DEPS=1)." >&2
        return 0
    fi

    pushd "${REPO_ROOT}" >/dev/null
    uv sync --all-extras --dev
    popd >/dev/null
}

ensure_minikube_running() {
    if minikube status >/dev/null 2>&1; then
        return 0
    fi

    local driver
    driver="${MINIKUBE_DRIVER:-docker}"
    if [[ "${driver}" == "docker" && "${DOCKER_AVAILABLE}" -ne 1 ]]; then
        cat <<'ERROR' >&2
Docker is required for the Minikube docker driver but was not detected.
Install Docker or set MINIKUBE_DRIVER to a supported alternative before rerunning --verify.
ERROR
        exit 1
    fi
    echo "🚜 Starting Minikube with driver '${driver}'"
    minikube start --driver="${driver}"
}

verify_install() {
    echo "📦 Deploying Helm chart via just..."
    pushd "${REPO_ROOT}" >/dev/null

    helm uninstall voice-changer >/dev/null 2>&1 || true
    ensure_minikube_running
    ./scripts/build-images.sh
    SKIP_BUILD_IMAGES=1 just helm

    local cluster_ip
    cluster_ip="$(minikube ip)"
    local api_base
    api_base="http://${cluster_ip}:30900"
    local output_file
    output_file="${CACHE_DIR}/minikube-output.wav"

    echo "🔍 Running regression client against ${api_base}"
    uv run ./test_client.py --api-base "${api_base}" --output "${output_file}"

    popd >/dev/null
}

main() {
    local run_verify=0

    while (($#)); do
        case "$1" in
            --verify)
                run_verify=1
                shift
                ;;
            --help|-h)
                cat <<'USAGE'
Usage: ./setup.sh [--verify]

Installs Minikube, kubectl, Helm, uv, just, and Python dependencies into $HOME.

  --verify   Build images, deploy with `just helm`, and run the regression client.
USAGE
                return 0
                ;;
            *)
                echo "Unknown argument: $1" >&2
                echo "Try './setup.sh --help' for usage." >&2
                return 1
                ;;
        esac
    done

    check_prerequisites
    ensure_path_exports
    install_uv
    install_python_dependencies
    install_kubectl
    install_minikube
    install_helm
    install_just

    if (( run_verify )); then
        verify_install
    fi

    cat <<'SUMMARY'
✅ Prerequisites installed in $HOME.
Ensure your Docker daemon is running and that the following directory is on your PATH:
  export PATH="$HOME/.local/bin:$HOME/bin:$PATH"

To start Minikube against the local Docker daemon:
  minikube start --driver=docker

Build Kubernetes images, install the Helm chart, and run the regression client:
  ./scripts/build-images.sh
  just helm
  uv run ./test_client.py --api-base "http://$(minikube ip):30900"

Re-run verification any time with:
  ./setup.sh --verify
SUMMARY
}

main "$@"
