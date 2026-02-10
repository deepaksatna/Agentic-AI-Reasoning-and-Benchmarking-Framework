#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# ARBM - Reasoning-Agent-Benchmark Quickstart Script
# ═══════════════════════════════════════════════════════════════════════════════
# Usage:
#   ./quickstart.sh vllm      - Start vLLM server
#   ./quickstart.sh benchmark - Run all benchmark tracks
#   ./quickstart.sh track 01  - Run specific track
#   ./quickstart.sh report    - Generate report and plots
#   ./quickstart.sh all       - Full pipeline (vllm + benchmark + report)
# ═══════════════════════════════════════════════════════════════════════════════
# Author: Deepak Soni
# License: MIT
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Source environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS AND FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK PREREQUISITES
# ═══════════════════════════════════════════════════════════════════════════════

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found. Please install Python3."
        exit 1
    fi

    # Check NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        log_success "Found ${GPU_COUNT} GPU(s)"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv
    else
        log_warn "nvidia-smi not found. GPU benchmarks may not work."
    fi

    # Check required Python packages
    log_info "Checking Python packages..."
    python3 -c "import requests, yaml" 2>/dev/null || {
        log_warn "Installing required packages..."
        pip3 install requests pyyaml numpy matplotlib --quiet
    }

    log_success "Prerequisites check complete"
}

# ═══════════════════════════════════════════════════════════════════════════════
# START VLLM SERVER
# ═══════════════════════════════════════════════════════════════════════════════

start_vllm() {
    log_info "Starting vLLM server..."
    log_info "Model: ${LOCAL_MODEL_NAME}"
    log_info "TP Size: ${DEFAULT_TP_SIZE}"
    log_info "Port: ${VLLM_PORT}"

    # Check if vLLM is already running
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        log_success "vLLM server already running on port ${VLLM_PORT}"
        return 0
    fi

    # Start vLLM server
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${LOCAL_MODEL_NAME}" \
        --tensor-parallel-size ${DEFAULT_TP_SIZE} \
        --max-model-len ${VLLM_MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --dtype ${VLLM_DTYPE} \
        --trust-remote-code \
        --host ${VLLM_HOST} \
        --port ${VLLM_PORT} \
        2>&1 | tee "${LOG_DIR}/vllm_server.log" &

    VLLM_PID=$!
    echo ${VLLM_PID} > "${PROJECT_DIR}/vllm.pid"

    log_info "Waiting for vLLM server to start (PID: ${VLLM_PID})..."

    # Wait for server to be ready
    for i in {1..60}; do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            log_success "vLLM server is ready!"
            return 0
        fi
        echo -n "."
        sleep 5
    done

    log_error "vLLM server failed to start within 5 minutes"
    return 1
}

# ═══════════════════════════════════════════════════════════════════════════════
# STOP VLLM SERVER
# ═══════════════════════════════════════════════════════════════════════════════

stop_vllm() {
    log_info "Stopping vLLM server..."

    if [ -f "${PROJECT_DIR}/vllm.pid" ]; then
        VLLM_PID=$(cat "${PROJECT_DIR}/vllm.pid")
        if kill -0 ${VLLM_PID} 2>/dev/null; then
            kill ${VLLM_PID}
            log_success "vLLM server stopped (PID: ${VLLM_PID})"
        fi
        rm -f "${PROJECT_DIR}/vllm.pid"
    else
        # Try to find and kill vLLM process
        pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
        log_info "vLLM server stopped"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# RUN BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

run_benchmarks() {
    local track="${1:-all}"

    log_info "Running ARBM benchmarks (track: ${track})..."

    # Check if vLLM is running
    if ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        log_warn "vLLM server not running. Starting it first..."
        start_vllm
    fi

    # Create output directory
    mkdir -p "${BENCHMARKS_DIR}/results"

    # Run benchmarks
    if [ "${track}" = "all" ]; then
        log_info "Running all benchmark tracks..."
        python3 "${SCRIPTS_DIR}/run_all_tracks.py" \
            --config "${CONFIGS_DIR}/benchmark_config.yaml" \
            --track all \
            --provider vllm \
            --endpoint "http://localhost:${VLLM_PORT}" \
            --output "${BENCHMARKS_DIR}/results"
    else
        log_info "Running track ${track}..."
        python3 "${SCRIPTS_DIR}/run_all_tracks.py" \
            --config "${CONFIGS_DIR}/benchmark_config.yaml" \
            --track "${track}" \
            --provider vllm \
            --endpoint "http://localhost:${VLLM_PORT}" \
            --output "${BENCHMARKS_DIR}/results"
    fi

    log_success "Benchmarks complete!"
}

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

generate_reports() {
    log_info "Generating reports and plots..."

    mkdir -p "${REPORT_DIR}/plots"

    # Generate plots
    log_info "Generating visualization plots..."
    cd "${PROJECT_DIR}"
    python3 "${SCRIPTS_DIR}/generate_plots.py"

    # Generate ASCII report
    log_info "Generating ASCII report..."
    python3 "${SCRIPTS_DIR}/generate_report.py"

    log_success "Reports generated in ${REPORT_DIR}"
}

# ═══════════════════════════════════════════════════════════════════════════════
# SHOW STATUS
# ═══════════════════════════════════════════════════════════════════════════════

show_status() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  ARBM - Reasoning-Agent-Benchmark Status"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""

    # GPU Status
    echo "  GPU Status:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | sed 's/^/    /'
    else
        echo "    No GPU detected"
    fi
    echo ""

    # vLLM Status
    echo "  vLLM Server:"
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "    Status: Running on port ${VLLM_PORT}"
    else
        echo "    Status: Not running"
    fi
    echo ""

    # Results
    echo "  Benchmark Results:"
    if [ -d "${BENCHMARKS_DIR}/results" ]; then
        RESULT_COUNT=$(ls -la "${BENCHMARKS_DIR}/results"/*.json 2>/dev/null | wc -l)
        echo "    ${RESULT_COUNT} result files found"
    else
        echo "    No results yet"
    fi
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

print_usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  vllm [start|stop]  - Start/stop vLLM server"
    echo "  benchmark [track]  - Run benchmarks (track: 01-05 or all)"
    echo "  report             - Generate reports and plots"
    echo "  status             - Show current status"
    echo "  all                - Run full pipeline"
    echo ""
    echo "Examples:"
    echo "  $0 vllm start      - Start vLLM server"
    echo "  $0 benchmark all   - Run all benchmark tracks"
    echo "  $0 benchmark 01    - Run only track 01 (Reasoning Quality)"
    echo "  $0 report          - Generate reports"
    echo "  $0 all             - Full pipeline"
}

case "${1:-help}" in
    vllm)
        check_prerequisites
        case "${2:-start}" in
            start) start_vllm ;;
            stop) stop_vllm ;;
            *) start_vllm ;;
        esac
        ;;
    benchmark)
        check_prerequisites
        run_benchmarks "${2:-all}"
        ;;
    report)
        generate_reports
        ;;
    status)
        show_status
        ;;
    all)
        check_prerequisites
        start_vllm
        run_benchmarks "all"
        generate_reports
        show_status
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        log_error "Unknown command: $1"
        print_usage
        exit 1
        ;;
esac
