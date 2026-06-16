# ===================== Default Configuration =====================
# These values will be used if not provided as command line arguments

# Multi-server configuration - supports connecting to multiple Linux servers
# Format: "IP:PORT,IP:PORT,IP:PORT"
DEFAULT_LINUX_SERVERS="100.65.247.100:8081"
DEFAULT_LIGHTROOM_API_PORT="7777"
DEFAULT_LIGHTROOM_PLUGIN_PORT="7878"
DEFAULT_API_LIGHTROOM_PATH="./"
DEFAULT_PYTHON_BIN=""
DEFAULT_WORKDIR_BASE="$HOME/lrc_client_workdir"
DEFAULT_ENABLE_LIGHTROOM_RECOVERY="0"

# Client configuration
DEFAULT_CLIENT_ID="mac_$(hostname)_$(date +%s)"
DEFAULT_POLL_INTERVAL="1.0"
DEFAULT_CONNECTION_RETRY_DELAY="3.0"
DEFAULT_MAX_CONSECUTIVE_FAILURES="5"
DEFAULT_HEALTH_CHECK_INTERVAL="30.0"
DEFAULT_MAX_EMPTY_POLLS="50"
DEFAULT_LONG_POLL_WAIT="20.0"
DEFAULT_LOCAL_CACHE_LIMIT="100"

# ===================== Command Line Arguments =====================
# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --servers)
      LINUX_SERVERS="$2"
      shift 2
      ;;
    --api-port)
      LIGHTROOM_API_PORT="$2"
      shift 2
      ;;
    --plugin-port)
      LIGHTROOM_PLUGIN_PORT="$2"
      shift 2
      ;;
    --api-path)
      API_LIGHTROOM_PATH="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --client-id)
      CLIENT_ID="$2"
      shift 2
      ;;
    --poll-interval)
      POLL_INTERVAL="$2"
      shift 2
      ;;
    --retry-delay)
      CONNECTION_RETRY_DELAY="$2"
      shift 2
      ;;
    --max-failures)
      MAX_CONSECUTIVE_FAILURES="$2"
      shift 2
      ;;
    --health-interval)
      HEALTH_CHECK_INTERVAL="$2"
      shift 2
      ;;
    --max-empty-polls)
      MAX_EMPTY_POLLS="$2"
      shift 2
      ;;
    --long-poll-wait)
      LONG_POLL_WAIT="$2"
      shift 2
      ;;
    --local-cache-limit)
      LOCAL_CACHE_LIMIT="$2"
      shift 2
      ;;
    --workdir-base)
      WORKDIR_BASE="$2"
      shift 2
      ;;
    --enable-lightroom-recovery)
      ENABLE_LIGHTROOM_RECOVERY="1"
      shift 1
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --servers SERVERS       Linux servers (format: IP:PORT,IP:PORT)"
      echo "  --api-port PORT         Local Lightroom API port"
      echo "  --plugin-port PORT      Lightroom plugin socket port"
      echo "  --api-path PATH         API_Lightroom project path"
      echo "  --python-bin PATH       Python executable"
      echo "  --client-id ID          Client ID"
      echo "  --poll-interval SEC     Polling interval in seconds"
      echo "  --retry-delay SEC       Connection retry delay in seconds"
      echo "  --max-failures NUM      Maximum consecutive failures"
      echo "  --health-interval SEC   Health check interval in seconds"
      echo "  --max-empty-polls NUM   Consecutive empty polls threshold"
      echo "  --long-poll-wait SEC    Server long-poll wait hint in seconds"
      echo "  --local-cache-limit NUM Number of local lightroom_task_* dirs to keep"
      echo "  --workdir-base PATH     Directory for local lightroom_task_* work dirs"
      echo "  --enable-lightroom-recovery Enable Lightroom restart hook after bridge export timeout"
      echo "  --help                  Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Set default values if not provided
LINUX_SERVERS=${LINUX_SERVERS:-$DEFAULT_LINUX_SERVERS}
LIGHTROOM_API_PORT=${LIGHTROOM_API_PORT:-$DEFAULT_LIGHTROOM_API_PORT}
LIGHTROOM_PLUGIN_PORT=${LIGHTROOM_PLUGIN_PORT:-$DEFAULT_LIGHTROOM_PLUGIN_PORT}
API_LIGHTROOM_PATH=${API_LIGHTROOM_PATH:-$DEFAULT_API_LIGHTROOM_PATH}
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}
CLIENT_ID=${CLIENT_ID:-$DEFAULT_CLIENT_ID}
POLL_INTERVAL=${POLL_INTERVAL:-$DEFAULT_POLL_INTERVAL}
CONNECTION_RETRY_DELAY=${CONNECTION_RETRY_DELAY:-$DEFAULT_CONNECTION_RETRY_DELAY}
MAX_CONSECUTIVE_FAILURES=${MAX_CONSECUTIVE_FAILURES:-$DEFAULT_MAX_CONSECUTIVE_FAILURES}
HEALTH_CHECK_INTERVAL=${HEALTH_CHECK_INTERVAL:-$DEFAULT_HEALTH_CHECK_INTERVAL}
MAX_EMPTY_POLLS=${MAX_EMPTY_POLLS:-$DEFAULT_MAX_EMPTY_POLLS}
LONG_POLL_WAIT=${LONG_POLL_WAIT:-$DEFAULT_LONG_POLL_WAIT}
LOCAL_CACHE_LIMIT=${LOCAL_CACHE_LIMIT:-$DEFAULT_LOCAL_CACHE_LIMIT}
WORKDIR_BASE=${WORKDIR_BASE:-$DEFAULT_WORKDIR_BASE}
ENABLE_LIGHTROOM_RECOVERY=${ENABLE_LIGHTROOM_RECOVERY:-$DEFAULT_ENABLE_LIGHTROOM_RECOVERY}
# ================================================

echo "🍎 === Mac Lightroom Client Startup ==="
echo "Connection target: $LINUX_SERVERS"
echo "Local API port: $LIGHTROOM_API_PORT"
echo "Lightroom plugin port: $LIGHTROOM_PLUGIN_PORT"
echo "Python: ${PYTHON_BIN:-auto}"
echo "Workdir base: $WORKDIR_BASE"
echo "Client ID: $CLIENT_ID"
echo "================================================"

# Check API_Lightroom project path
if [ ! -d "$API_LIGHTROOM_PATH" ]; then
    echo "❌ Error: API_Lightroom project path does not exist: $API_LIGHTROOM_PATH"
    echo "💡 Please modify the API_LIGHTROOM_PATH variable in the script"
    exit 1
fi

cd "$API_LIGHTROOM_PATH"

if [ -z "$PYTHON_BIN" ]; then
    if [ -x ".venv/bin/python" ]; then
        PYTHON_BIN=".venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        PYTHON_BIN="python"
    fi
fi

# Check required files
if [ ! -f "lr_task_client.py" ]; then
    echo "❌ Error: lr_task_client.py file not found"
    echo "💡 Please ensure you are in the correct API_Lightroom directory: $API_LIGHTROOM_PATH"
    exit 1
fi

if [ ! -f "agent_to_lightroom/lrc_api_server.py" ]; then
    echo "❌ Error: agent_to_lightroom/lrc_api_server.py file not found"
    echo "💡 Please ensure you are in the correct API_Lightroom directory: $API_LIGHTROOM_PATH"
    exit 1
fi

if [ "$ENABLE_LIGHTROOM_RECOVERY" = "1" ]; then
    RECOVERY_SCRIPT="$(pwd)/recover_lightroom_mac.sh"
    if [ -f "$RECOVERY_SCRIPT" ]; then
        chmod +x "$RECOVERY_SCRIPT" 2>/dev/null || true
        export LIGHTROOM_BRIDGE_RECOVERY_COMMAND="$RECOVERY_SCRIPT"
        echo "Lightroom recovery hook: enabled"
    else
        echo "⚠️ Recovery requested but script not found: $RECOVERY_SCRIPT"
    fi
fi

# Function: Check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function: Wait for process to start
wait_for_service() {
    local port=$1
    local service_name=$2
    local max_wait=30
    
    echo "⏳ Waiting for $service_name to start (port $port)..."
    for i in $(seq 1 $max_wait); do
        if check_port $port; then
            echo "✅ $service_name is running"
            return 0
        fi
        sleep 1
    done
    
    echo "❌ $service_name startup timeout"
    return 1
}

wait_for_lightroom_plugin() {
    local max_wait=60

    echo "🔍 Checking Lightroom plugin socket..."
    for i in $(seq 1 $max_wait); do
        if check_port "$LIGHTROOM_PLUGIN_PORT"; then
            echo "✅ Lightroom plugin socket is running (port $LIGHTROOM_PLUGIN_PORT)"
            return 0
        fi
        if [ "$i" -eq 1 ]; then
            open -a "Adobe Lightroom Classic" >/dev/null 2>&1 || true
        fi
        sleep 1
    done

    echo "❌ Lightroom plugin socket startup timeout (port $LIGHTROOM_PLUGIN_PORT)"
    return 1
}

# 1. Check and start test_lightroom_api.py
if ! wait_for_lightroom_plugin; then
    echo "📋 Please check:"
    echo "  1. Is Lightroom Classic fully open?"
    echo "  2. Is the XMPlayer Lightroom plugin installed and enabled?"
    echo "  3. Is plugin port $LIGHTROOM_PLUGIN_PORT already blocked?"
    exit 1
fi

echo "🔍 Checking Lightroom API service..."
if check_port $LIGHTROOM_API_PORT; then
    echo "✅ Lightroom API service is already running (port $LIGHTROOM_API_PORT)"
else
    echo "🚀 Starting Lightroom API service..."
    "$PYTHON_BIN" agent_to_lightroom/lrc_api_server.py --port "$LIGHTROOM_API_PORT" > lightroom_api.log 2>&1 &
    LIGHTROOM_API_PID=$!
    echo "Lightroom API PID: $LIGHTROOM_API_PID"
    
    if ! wait_for_service $LIGHTROOM_API_PORT "Lightroom API"; then
        echo "❌ Lightroom API service startup failed"
        echo "📋 Please check:"
        echo "  1. Is Lightroom running?"
        echo "  2. Is port $LIGHTROOM_API_PORT already in use?"
        echo "  3. Check logs: tail lightroom_api.log"
        exit 1
    fi
fi

# 2. Test connection to Linux servers
echo "🔍 Testing connection to Linux servers..."
SERVERS_AVAILABLE=0
TOTAL_SERVERS=0

IFS=',' read -ra SERVER_ARRAY <<< "$LINUX_SERVERS"
for server in "${SERVER_ARRAY[@]}"; do
    TOTAL_SERVERS=$((TOTAL_SERVERS + 1))
    IFS=':' read -ra SERVER_PARTS <<< "$server"
    SERVER_IP="${SERVER_PARTS[0]}"
    SERVER_PORT="${SERVER_PARTS[1]}"
    
    echo "  Testing server: $SERVER_IP:$SERVER_PORT"
    if nc -z "$SERVER_IP" "$SERVER_PORT" 2>/dev/null; then
        echo "  ✅ $SERVER_IP:$SERVER_PORT connection successful"
        SERVERS_AVAILABLE=$((SERVERS_AVAILABLE + 1))
    else
        echo "  ⚠️ $SERVER_IP:$SERVER_PORT connection failed"
    fi
done

if [ $SERVERS_AVAILABLE -eq 0 ]; then
    echo "⚠️ All Linux servers are currently unreachable"
    echo "📋 Possible reasons:"
    echo "  1. Linux server IP addresses are incorrect"
    echo "  2. reverse_server.py is not running on Linux"
    echo "  3. Network connection or firewall issues"
    echo "💡 Client will continue to start and automatically attempt to reconnect..."
fi

if [ $SERVERS_AVAILABLE -gt 0 ]; then
    echo "✅ $SERVERS_AVAILABLE/$TOTAL_SERVERS Linux servers connected successfully"
fi

# 3. Start reverse_client.py
echo "🚀 Starting reverse_client to connect to Linux servers..."
echo "Press Ctrl+C to stop the client"
echo "================================================"

# Create startup function with intelligent reconnection support
start_reverse_client() {
    local restart_count=0
    local max_restarts=0  # 0 means unlimited retries
    local base_delay=5     # Base restart delay
    
    while true; do
        restart_count=$((restart_count + 1))
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 🔄 Starting reverse_client (attempt ${restart_count})..."
        
        # Start client with parameters from command line or defaults
        "$PYTHON_BIN" lr_task_client.py \
            --servers "$LINUX_SERVERS" \
            --local-port "$LIGHTROOM_API_PORT" \
            --client-id "$CLIENT_ID" \
            --poll-interval "$POLL_INTERVAL" \
            --connection-retry-delay "$CONNECTION_RETRY_DELAY" \
            --max-consecutive-failures "$MAX_CONSECUTIVE_FAILURES" \
            --health-check-interval "$HEALTH_CHECK_INTERVAL" \
            --max-empty-polls "$MAX_EMPTY_POLLS" \
            --long-poll-wait "$LONG_POLL_WAIT" \
            --local-cache-limit "$LOCAL_CACHE_LIMIT" \
            --workdir-base "$WORKDIR_BASE" \
            --http-timeout-total 300.0 \
            --connector-limit 5 \
            --base-processing-timeout 10.0 \
            --max-timeout-mask 180.0 \
            --max-timeout-complex 120.0 \
            --processing-extra-buffer 15.0
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - ✅ reverse_client exited normally"
            break
        elif [ $EXIT_CODE -eq 130 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - 🛑 User interrupted (Ctrl+C)"
            break
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') - ⚠️ reverse_client exited abnormally (code: $EXIT_CODE)"
            
            # Check if maximum restart count is reached (if limit is set)
            if [ $max_restarts -gt 0 ] && [ $restart_count -ge $max_restarts ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') - ❌ Maximum restart count ($max_restarts) reached, stopping retries"
                break
            fi
            
            # Use fixed retry interval
            local delay=$base_delay  # Use base delay fixedly
            
            echo "$(date '+%Y-%m-%d %H:%M:%S') - ⏳ Auto-reconnecting in ${delay} seconds..."
            sleep $delay
            
            # Print a log message every 30 restarts without resetting the counter
            if [ $((restart_count % 30)) -eq 0 ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') - 🔄 Attempted restart $restart_count times"
            fi
        fi
    done
}

# Setup cleanup function
cleanup() {
    echo ""
    echo "🧹 Cleaning up processes..."
    
    # Clean up Lightroom API process (if we started it)
    if [ ! -z "$LIGHTROOM_API_PID" ]; then
        if kill -0 $LIGHTROOM_API_PID 2>/dev/null; then
            echo "🧹 Stopping Lightroom API service (PID: $LIGHTROOM_API_PID)..."
            kill $LIGHTROOM_API_PID
        fi
    fi
    
    echo "👋 Mac client stopped"
    exit 0
}

# Setup signal handlers
trap cleanup SIGINT SIGTERM

# Start reverse_client (with auto-reconnect)
start_reverse_client

# Cleanup on normal exit
cleanup
