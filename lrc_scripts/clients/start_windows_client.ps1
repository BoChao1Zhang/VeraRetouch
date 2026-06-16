<#
.SYNOPSIS
    Windows launcher for the JarvisArt Lightroom client.

.DESCRIPTION
    PowerShell port of start_mac_client.sh. It:
      1. Parses the same flags/defaults as the Mac script (plus --long-poll-wait).
      2. Resolves a Python 3.9+ interpreter (prefers "py -3", falls back to "python").
      3. Verifies the required Python files exist under the API project path.
      4. Starts the local Lightroom bridge (agent_to_lightroom/lrc_api_server.py)
         on the requested port if it is not already listening.
      5. Waits for the bridge to come up, then probes each configured Linux server
         for reachability (raw TcpClient, ~1s timeout).
      6. Runs lr_task_client.py in an auto-restart loop until it exits cleanly
         (exit code 0) or the user presses Ctrl+C.
      7. On exit, stops the bridge process if (and only if) this script started it.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File start_windows_client.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File start_windows_client.ps1 --servers 100.65.247.100:8081
#>

# Stop on unhandled errors so failures surface immediately rather than silently.
$ErrorActionPreference = 'Stop'

# On PowerShell 7.3+ $PSNativeCommandUseErrorActionPreference defaults to $true, which
# (combined with the Stop preference above) turns a non-zero exit from a native command
# (python) into a TERMINATING error. The Python version probe and the poller loop both
# intentionally rely on reading $LASTEXITCODE after a non-zero exit, so disable that
# behavior here. No-op on Windows PowerShell 5.1 (the variable does not exist there).
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

# ===================== Default Configuration =====================
# These values are used if not provided as command line arguments.

# Multi-server configuration - supports connecting to multiple Linux servers.
# The default is the Linux server's Tailscale IP; client/server communication is
# expected to stay inside the tailnet.
# Format: "IP:PORT,IP:PORT,IP:PORT"
$DefaultLinuxServers          = '100.65.247.100:8081'
$DefaultLightroomApiPort      = '7777'
$DefaultApiLightroomPath      = './'

# Client configuration. Default client id: win_<COMPUTERNAME>_<unix-seconds>
$DefaultClientId              = "win_$($env:COMPUTERNAME)_$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
$DefaultPollInterval          = '1.0'
$DefaultConnectionRetryDelay  = '3.0'
$DefaultMaxConsecutiveFailures = '5'
$DefaultHealthCheckInterval   = '30.0'
$DefaultMaxEmptyPolls         = '50'
$DefaultLongPollWait          = '20.0'
$DefaultLocalCacheLimit       = '100'
$DefaultWorkdirBase           = Join-Path $env:LOCALAPPDATA 'LightroomTaskClient\workdir'
$DefaultLightroomRecoveryCommand = ''
$DefaultRequireTailscale      = $true

# ===================== Command Line Arguments =====================
# We parse $args manually (rather than param()) so the flag names match the Mac
# script exactly (e.g. "--servers"), and so passing them through .bat is trivial.

function Show-Usage {
    Write-Host "Usage: start_windows_client.ps1 [options]"
    Write-Host "Options:"
    Write-Host "  --servers SERVERS       Linux servers (format: IP:PORT,IP:PORT)"
    Write-Host "  --api-port PORT         Local Lightroom API port"
    Write-Host "  --api-path PATH         API_Lightroom project path"
    Write-Host "  --client-id ID          Client ID"
    Write-Host "  --poll-interval SEC     Polling interval in seconds"
    Write-Host "  --retry-delay SEC       Connection retry delay in seconds"
    Write-Host "  --max-failures NUM      Maximum consecutive failures"
    Write-Host "  --health-interval SEC   Health check interval in seconds"
    Write-Host "  --max-empty-polls NUM   Consecutive empty polls threshold"
    Write-Host "  --long-poll-wait SEC    Long-poll wait hint sent to the server (seconds)"
    Write-Host "  --local-cache-limit NUM Number of local lightroom_task_* dirs to keep"
    Write-Host "  --workdir-base PATH     Directory for local lightroom_task_* work dirs"
    Write-Host "  --lightroom-recovery-command CMD Command run after bridge export timeout"
    Write-Host "  --skip-tailscale-check  Skip local Tailscale CLI/IP validation"
    Write-Host "  -h, --help              Show this help message"
}

# Initialize from defaults; CLI args override below.
$LinuxServers           = $DefaultLinuxServers
$LightroomApiPort       = $DefaultLightroomApiPort
$ApiLightroomPath       = $DefaultApiLightroomPath
$ClientId               = $DefaultClientId
$PollInterval           = $DefaultPollInterval
$ConnectionRetryDelay   = $DefaultConnectionRetryDelay
$MaxConsecutiveFailures = $DefaultMaxConsecutiveFailures
$HealthCheckInterval    = $DefaultHealthCheckInterval
$MaxEmptyPolls          = $DefaultMaxEmptyPolls
$LongPollWait           = $DefaultLongPollWait
$LocalCacheLimit        = $DefaultLocalCacheLimit
$WorkdirBase            = $DefaultWorkdirBase
$LightroomRecoveryCommand = $DefaultLightroomRecoveryCommand
$RequireTailscale       = $DefaultRequireTailscale

# Walk the argument list two at a time (flag + value), except for help.
$i = 0
while ($i -lt $args.Count) {
    $arg = [string]$args[$i]
    switch ($arg) {
        '--servers'         { $LinuxServers = [string]$args[$i + 1]; $i += 2; continue }
        '--api-port'        { $LightroomApiPort = [string]$args[$i + 1]; $i += 2; continue }
        '--api-path'        { $ApiLightroomPath = [string]$args[$i + 1]; $i += 2; continue }
        '--client-id'       { $ClientId = [string]$args[$i + 1]; $i += 2; continue }
        '--poll-interval'   { $PollInterval = [string]$args[$i + 1]; $i += 2; continue }
        '--retry-delay'     { $ConnectionRetryDelay = [string]$args[$i + 1]; $i += 2; continue }
        '--max-failures'    { $MaxConsecutiveFailures = [string]$args[$i + 1]; $i += 2; continue }
        '--health-interval' { $HealthCheckInterval = [string]$args[$i + 1]; $i += 2; continue }
        '--max-empty-polls' { $MaxEmptyPolls = [string]$args[$i + 1]; $i += 2; continue }
        '--long-poll-wait'  { $LongPollWait = [string]$args[$i + 1]; $i += 2; continue }
        '--local-cache-limit' { $LocalCacheLimit = [string]$args[$i + 1]; $i += 2; continue }
        '--workdir-base'    { $WorkdirBase = [string]$args[$i + 1]; $i += 2; continue }
        '--lightroom-recovery-command' { $LightroomRecoveryCommand = [string]$args[$i + 1]; $i += 2; continue }
        '--skip-tailscale-check' { $RequireTailscale = $false; $i += 1; continue }
        '-h'                { Show-Usage; exit 0 }
        '--help'            { Show-Usage; exit 0 }
        default {
            Write-Host "Unknown option: $arg"
            Write-Host "Use --help for usage information"
            exit 1
        }
    }
}
# ================================================

Write-Host "[Win] === Windows Lightroom Client Startup ==="
Write-Host "Connection target: $LinuxServers"
Write-Host "Local API port: $LightroomApiPort"
Write-Host "Client ID: $ClientId"
Write-Host "Workdir base: $WorkdirBase"
Write-Host "Tailscale required: $RequireTailscale"
Write-Host "================================================"

# ===================== Tailscale validation =====================
function Test-TailscaleReady {
    if (-not (Get-Command 'tailscale' -ErrorAction SilentlyContinue)) {
        Write-Host "[X] Tailscale CLI not found on PATH."
        Write-Host "[i] Install Tailscale for Windows, log in, then restart this launcher."
        return $false
    }

    $ipOutput = & tailscale ip -4 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($ipOutput)) {
        Write-Host "[X] Tailscale is not ready or this machine has no Tailscale IPv4 address."
        Write-Host "[i] Open Tailscale, sign in, and confirm this Windows machine is connected."
        return $false
    }

    Write-Host "[OK] Tailscale IPv4: $($ipOutput.Trim())"
    return $true
}

if ($RequireTailscale -and -not (Test-TailscaleReady)) {
    exit 1
}

# ===================== Resolve Python interpreter =====================
# Prefer the package-local virtual environment when present; fall back to the
# Windows launcher "py -3", then plain "python". We store the resolved command
# as an argument list so callers can splat/invoke it uniformly.
function Resolve-Python {
    $localVenvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $localVenvPython -PathType Leaf) {
        return @{ Exe = $localVenvPython; PreArgs = @() }
    }

    # Try "py -3" first (the recommended Python launcher on Windows).
    if (Get-Command 'py' -ErrorAction SilentlyContinue) {
        try {
            & py -3 -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @{ Exe = 'py'; PreArgs = @('-3') }
            }
        } catch {
            # Fall through to "python".
        }
    }
    # Fall back to plain "python".
    if (Get-Command 'python' -ErrorAction SilentlyContinue) {
        return @{ Exe = 'python'; PreArgs = @() }
    }
    return $null
}

$pyInfo = Resolve-Python
if ($null -eq $pyInfo) {
    Write-Host "[X] Error: Python not found. Install Python 3.9+ and ensure 'py' or 'python' is on PATH."
    Write-Host "[i] Recommended setup from this package directory:"
    Write-Host "    py -3 -m venv .venv"
    Write-Host "    .\.venv\Scripts\python.exe -m pip install -r requirements_windows.txt"
    exit 1
}
$PyExe  = $pyInfo.Exe
$PyArgs = $pyInfo.PreArgs

# Verify the interpreter is >= 3.9.
& $PyExe @PyArgs -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[X] Error: Python 3.9 or newer is required."
    exit 1
}
Write-Host "[OK] Using Python: $PyExe $($PyArgs -join ' ')"

# Verify required Python packages before starting long-running processes.
& $PyExe @PyArgs -c "import aiohttp, requests" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[X] Missing Python dependencies."
    Write-Host "[i] Run these commands from this package directory:"
    Write-Host "    py -3 -m venv .venv"
    Write-Host "    .\.venv\Scripts\python.exe -m pip install -r requirements_windows.txt"
    exit 1
}
Write-Host "[OK] Python dependencies are installed"

# ===================== Project path & required files =====================
if (-not (Test-Path -LiteralPath $ApiLightroomPath -PathType Container)) {
    Write-Host "[X] Error: API_Lightroom project path does not exist: $ApiLightroomPath"
    Write-Host "[i] Please pass --api-path or adjust the script default."
    exit 1
}

# cd into the project path so relative script paths resolve like the Mac script.
Set-Location -LiteralPath $ApiLightroomPath

if (-not (Test-Path -LiteralPath 'lr_task_client.py' -PathType Leaf)) {
    Write-Host "[X] Error: lr_task_client.py file not found"
    Write-Host "[i] Ensure you are in the correct API_Lightroom directory: $ApiLightroomPath"
    exit 1
}

if (-not (Test-Path -LiteralPath 'agent_to_lightroom/lrc_api_server.py' -PathType Leaf)) {
    Write-Host "[X] Error: agent_to_lightroom/lrc_api_server.py file not found"
    Write-Host "[i] Ensure you are in the correct API_Lightroom directory: $ApiLightroomPath"
    exit 1
}

# ===================== Helpers =====================

# Returns $true if a local TCP port is being listened on.
function Test-PortInUse {
    param([int]$Port)

    # Preferred: Get-NetTCPConnection (available on modern Windows).
    if (Get-Command 'Get-NetTCPConnection' -ErrorAction SilentlyContinue) {
        $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        return ($null -ne $conn)
    }

    # Fallback: parse netstat output for a LISTENING socket on this port.
    $pattern = ":$Port\s"
    $lines = netstat -ano -p tcp 2>$null | Select-String -Pattern $pattern
    foreach ($line in $lines) {
        if ($line -match 'LISTENING') {
            return $true
        }
    }
    return $false
}

# Waits up to $MaxWait seconds for a service to start listening on $Port.
function Wait-ForService {
    param(
        [int]$Port,
        [string]$ServiceName,
        [int]$MaxWait = 30
    )

    Write-Host "[..] Waiting for $ServiceName to start (port $Port)..."
    for ($s = 1; $s -le $MaxWait; $s++) {
        if (Test-PortInUse -Port $Port) {
            Write-Host "[OK] $ServiceName is running"
            return $true
        }
        Start-Sleep -Seconds 1
    }

    Write-Host "[X] $ServiceName startup timeout"
    return $false
}

# Raw TCP reachability probe with ~1s timeout (avoids the slow Test-NetConnection).
function Test-ServerReachable {
    param(
        [string]$ServerIp,
        [int]$ServerPort,
        [int]$TimeoutMs = 1000
    )

    $client = $null
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $async = $client.BeginConnect($ServerIp, $ServerPort, $null, $null)
        $ok = $async.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
        if ($ok -and $client.Connected) {
            $client.EndConnect($async)
            return $true
        }
        return $false
    } catch {
        return $false
    } finally {
        if ($null -ne $client) {
            $client.Close()
        }
    }
}

# Stops the bridge we started, including any child process. When Python is the
# "py" launcher, $BridgePid is the launcher and the real python.exe (which holds
# the port) is its child, so we must kill the tree, not just the launcher.
function Stop-BridgeTree {
    param([int]$BridgeId)
    if (-not $BridgeId) { return }
    try {
        Get-CimInstance Win32_Process -Filter "ParentProcessId=$BridgeId" -ErrorAction SilentlyContinue |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    } catch {
        # Get-CimInstance may be unavailable on minimal hosts; fall through.
    }
    Stop-Process -Id $BridgeId -Force -ErrorAction SilentlyContinue
}

# ===================== 1. Check and start the Lightroom bridge =====================
$BridgePid = $null  # Only set when WE start the bridge (so cleanup is scoped).

if (-not [string]::IsNullOrWhiteSpace($LightroomRecoveryCommand)) {
    $env:LIGHTROOM_BRIDGE_RECOVERY_COMMAND = $LightroomRecoveryCommand
    Write-Host "[OK] Lightroom recovery command enabled"
}

Write-Host "[..] Checking Lightroom API service..."
if (Test-PortInUse -Port ([int]$LightroomApiPort)) {
    Write-Host "[OK] Lightroom API service is already running (port $LightroomApiPort)"
} else {
    Write-Host "[..] Starting Lightroom API service..."

    # Build the full argument list: <PreArgs> agent_to_lightroom/lrc_api_server.py --port <port>
    $bridgeArgs = @()
    $bridgeArgs += $PyArgs
    $bridgeArgs += 'agent_to_lightroom/lrc_api_server.py'
    $bridgeArgs += '--port'
    $bridgeArgs += [string]$LightroomApiPort

    $proc = Start-Process -FilePath $PyExe `
        -ArgumentList $bridgeArgs `
        -RedirectStandardOutput 'lightroom_api.log' `
        -RedirectStandardError 'lightroom_api.err' `
        -PassThru -NoNewWindow
    $BridgePid = $proc.Id
    Write-Host "Lightroom API PID: $BridgePid"

    if (-not (Wait-ForService -Port ([int]$LightroomApiPort) -ServiceName 'Lightroom API')) {
        Write-Host "[X] Lightroom API service startup failed"
        Write-Host "[i] Please check:"
        Write-Host "  1. Is Lightroom running?"
        Write-Host "  2. Is port $LightroomApiPort already in use?"
        Write-Host "  3. Check logs: lightroom_api.log / lightroom_api.err"
        # Stop the bridge we just started before bailing out.
        Stop-BridgeTree -BridgeId $BridgePid
        exit 1
    }
}

# ===================== 2. Test connection to Linux servers =====================
Write-Host "[..] Testing connection to Linux servers..."
$ServersAvailable = 0
$TotalServers = 0

foreach ($server in ($LinuxServers -split ',')) {
    $server = $server.Trim()
    if ([string]::IsNullOrWhiteSpace($server)) { continue }

    $TotalServers++
    $parts = $server -split ':'
    $serverIp = $parts[0]
    $serverPort = [int]$parts[1]

    Write-Host "  Testing server: $serverIp`:$serverPort"
    if (Test-ServerReachable -ServerIp $serverIp -ServerPort $serverPort -TimeoutMs 1000) {
        Write-Host "  [OK] $serverIp`:$serverPort connection successful"
        $ServersAvailable++
    } else {
        Write-Host "  [!] $serverIp`:$serverPort connection failed"
    }
}

if ($ServersAvailable -eq 0) {
    Write-Host "[!] All Linux servers are currently unreachable"
    Write-Host "[i] Possible reasons:"
    Write-Host "  1. Linux server IP addresses are incorrect"
    Write-Host "  2. The task server is not running on Linux"
    Write-Host "  3. Network connection or firewall issues"
    Write-Host "[i] Client will continue to start and automatically attempt to reconnect..."
} else {
    Write-Host "[OK] $ServersAvailable/$TotalServers Linux servers connected successfully"
}

# ===================== 3. Start the poller (with auto-restart) =====================
Write-Host "[..] Starting lr_task_client to connect to Linux servers..."
Write-Host "Press Ctrl+C to stop the client"
Write-Host "================================================"

# Ctrl+C handling: set a flag (and suppress immediate termination) so the restart
# loop can stop cleanly instead of relaunching the client the user just killed.
$script:Interrupted = $false
try {
    [Console]::add_CancelKeyPress({
        param($sender, $e)
        $script:Interrupted = $true
        $e.Cancel = $true  # let the script run its cleanup rather than aborting
    })
} catch {
    # Non-interactive host without a console; rely on exit-code detection below.
}

try {
    $restartCount = 0
    while ($true) {
        if ($script:Interrupted) {
            Write-Host "[stop] Interrupt received, stopping client"
            break
        }
        $restartCount++
        $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        Write-Host "$ts - [..] Starting lr_task_client (attempt $restartCount)..."

        # Invoke the poller with the same CLI flags lr_task_client.py accepts.
        & $PyExe @PyArgs lr_task_client.py `
            --servers $LinuxServers `
            --local-port $LightroomApiPort `
            --client-id $ClientId `
            --poll-interval $PollInterval `
            --connection-retry-delay $ConnectionRetryDelay `
            --max-consecutive-failures $MaxConsecutiveFailures `
            --health-check-interval $HealthCheckInterval `
            --max-empty-polls $MaxEmptyPolls `
            --long-poll-wait $LongPollWait `
            --local-cache-limit $LocalCacheLimit `
            --workdir-base $WorkdirBase `
            --http-timeout-total 300.0 `
            --connector-limit 5 `
            --base-processing-timeout 10.0 `
            --max-timeout-mask 180.0 `
            --max-timeout-complex 120.0 `
            --processing-extra-buffer 15.0

        $code = $LASTEXITCODE
        $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

        # Clean stop on: normal exit (0), a user interrupt flag, or a Windows/Unix
        # interrupt exit code (Ctrl+C: STATUS_CONTROL_C_EXIT 0xC000013A = -1073741510
        # / 3221225786 unsigned; Unix SIGINT 130). Do NOT relaunch in these cases.
        if ($code -eq 0 -or $script:Interrupted -or
            $code -eq 130 -or $code -eq -1073741510 -or $code -eq 3221225786) {
            Write-Host "$ts - [stop] lr_task_client stopped"
            break
        }

        Write-Host "$ts - [!] lr_task_client exited abnormally (code: $code)"
        Write-Host "$ts - [..] Auto-reconnecting in 5 seconds..."
        Start-Sleep -Seconds 5

        # Periodic heartbeat without resetting the counter.
        if (($restartCount % 30) -eq 0) {
            $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
            Write-Host "$ts - [..] Attempted restart $restartCount times"
        }
    }
} finally {
    # ===================== Cleanup =====================
    # Stop the bridge (and its child interpreter) only if THIS script started it.
    Write-Host ""
    Write-Host "[..] Cleaning up processes..."
    if ($null -ne $BridgePid) {
        Write-Host "[..] Stopping Lightroom API service (PID: $BridgePid)..."
        Stop-BridgeTree -BridgeId $BridgePid
    }
    Write-Host "[bye] Windows client stopped"
}
