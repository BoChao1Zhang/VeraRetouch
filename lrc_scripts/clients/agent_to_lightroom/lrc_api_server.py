import socket
import os
import time
import threading
import argparse
import tempfile
import subprocess
import re
import shutil
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse

# 等待真实导出文件落盘的轮询参数（可通过环境变量覆盖）
BRIDGE_POLL_INTERVAL = float(os.getenv('LIGHTROOM_BRIDGE_POLL_INTERVAL', '0.15'))
BRIDGE_WAIT_TIMEOUT = float(os.getenv('LIGHTROOM_BRIDGE_WAIT_TIMEOUT', '120.0'))
BRIDGE_RECOVERY_COMMAND = os.getenv('LIGHTROOM_BRIDGE_RECOVERY_COMMAND')
DEFAULT_BRIDGE_HEALTH_FILE = (
    Path(tempfile.gettempdir()) / 'lightroom_bridge_health.txt'
    if os.name == 'nt'
    else Path('/tmp/lightroom_bridge_health.txt')
)
BRIDGE_HEALTH_FILE = Path(os.getenv(
    'LIGHTROOM_BRIDGE_HEALTH_FILE',
    str(DEFAULT_BRIDGE_HEALTH_FILE)
))
BRIDGE_HEALTH_FILE_MAX_AGE = float(os.getenv('LIGHTROOM_BRIDGE_HEALTH_FILE_MAX_AGE', '300.0'))

class LightroomBridgeError(Exception):
    """Structured bridge-side failure that can be propagated to callers."""

    def __init__(self, code, message, *, retryable=False, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}


class LightroomAPI:
    def __init__(self, host="127.0.0.1", port=7878):
        self.host = host
        self.port = port
        self.socket = None
        self.response_thread = None
        self.connected = False
        self.last_output_path = None  # 存储最后处理的图片输出路径
        self.last_error = None  # 存储插件返回的最近一次错误信息
        self.last_health = None
        self._lr_lock = threading.Lock()  # 串行化 send+wait 区段，保证单一 Lightroom socket
    
    def connect(self):
        """建立与服务器的连接"""
        if self.connected:
            return True
            
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(None)  # 禁用超时
            # 设置 TCP keepalive
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # 在 Linux 系统上设置更多的 keepalive 参数
            if hasattr(socket, 'TCP_KEEPIDLE'):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
            if hasattr(socket, 'TCP_KEEPINTVL'):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60)
            if hasattr(socket, 'TCP_KEEPCNT'):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
            
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # 启动响应处理线程
            self.response_thread = threading.Thread(target=self._handle_responses)
            self.response_thread.daemon = True
            self.response_thread.start()
            
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
            return False
    
    def _handle_responses(self):
        """处理服务器响应"""
        while self.connected:
            try:
                response = self.socket.recv(4096).decode().strip()
                if not response:
                    print("Empty response from server")
                    break
                
                print(f"Received response: {response}")
                status, *message = response.split('|')
                message = '|'.join(message) if message else ''
                
                if status == "success":
                    print(f"\nPhoto processed successfully!")
                    if message:
                        self.last_output_path = message  # 保存输出路径
                        print(f"Output saved to: {message}")
                elif status == "error":
                    self.last_output_path = None
                    self.last_error = message  # 记录错误，供 process_photo 快速失败
                    print(f"\nError: {message}")
                elif status == "health":
                    self.last_output_path = None
                    self.last_health = message
                    print(f"Lightroom health: {message}")
                elif status == "ok":
                    # LrSocket sends an "ok" connection acknowledgement before
                    # command responses. It is not a task result.
                    pass
                elif status == "pong":
                    print("Server is alive (received pong)")
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error in response handler: {e}")
                self.connected = False
                break
    
    def send_request(self, message):
        """发送请求到服务器"""
        if not self.connected and not self.connect():
            return None
            
        try:
            self.socket.sendall((message + "\n").encode())
            return True
        except Exception as e:
            print(f"Error sending request: {e}")
            self.connected = False
            return None
    
    def sanitize_export_basename(self, basename):
        """Return a single safe filename stem for Lightroom export."""
        if not basename:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(basename)).strip("._-")
        if not safe:
            return None
        return safe[:180]

    def normalize_output_path(self, output_path, expected_output_path):
        """Move/copy Lightroom's actual export to the unique expected path."""
        if not output_path:
            return output_path

        source = Path(output_path)
        expected = Path(expected_output_path)
        try:
            if source.resolve() == expected.resolve():
                return str(source)
        except OSError:
            if str(source) == str(expected):
                return str(source)

        try:
            if expected.exists() and expected.stat().st_size > 0:
                return str(expected)
        except OSError:
            pass

        if not source.exists():
            return str(source)

        expected.parent.mkdir(parents=True, exist_ok=True)
        try:
            source.replace(expected)
            print(f"Normalized Lightroom output filename: {source} -> {expected}")
            return str(expected)
        except OSError as move_error:
            try:
                shutil.copy2(source, expected)
                print(f"Copied Lightroom output to unique filename after move failed: {move_error}")
                return str(expected)
            except OSError as copy_error:
                print(f"Failed to normalize Lightroom output filename: move={move_error}; copy={copy_error}")
                return str(source)

    def prepare_render_source(self, photo_path_obj, expected_output_dir, export_stem, source_stem):
        """Give Lightroom a unique source basename so export names stay unique."""
        if export_stem == source_stem:
            return photo_path_obj

        source_ext = photo_path_obj.suffix or ".jpg"
        alias_dir = expected_output_dir / "_source"
        alias_dir.mkdir(parents=True, exist_ok=True)
        alias_path = alias_dir / f"{export_stem}{source_ext}"

        if alias_path.exists() and alias_path.stat().st_size > 0:
            return alias_path.absolute()

        try:
            os.link(photo_path_obj, alias_path)
            print(f"Created unique Lightroom source hardlink: {alias_path}")
        except OSError:
            shutil.copy2(photo_path_obj, alias_path)
            print(f"Created unique Lightroom source copy: {alias_path}")

        return alias_path.absolute()

    def process_photo(self, photo_path, xmp_path, output_dir=None, timeout=None, task_id=None, output_basename=None):
        """处理照片并阻塞直至真实导出文件落盘"""
        # 确保文件存在（在加锁之前校验，使 test_connection 等带伪路径的探测能快速失败而不抢锁）
        if not os.path.exists(photo_path):
            raise FileNotFoundError(f"Photo file not found: {photo_path}")
        if not os.path.exists(xmp_path):
            raise FileNotFoundError(f"XMP preset file not found: {xmp_path}")

        # Lightroom actual export location. Newer clients pass a per-render
        # directory so main/baseline/probe renders cannot overwrite each other.
        photo_path_obj = Path(photo_path).absolute()
        if output_dir:
            expected_output_dir = Path(output_dir).expanduser().absolute()
        else:
            expected_output_dir = photo_path_obj.parent / "processed"
        stem = photo_path_obj.stem
        export_stem = self.sanitize_export_basename(output_basename) or stem
        expected_output_path = expected_output_dir / f"{export_stem}.jpg"
        status_path = expected_output_dir / ".lightroom_render_status"
        render_photo_path_obj = self.prepare_render_source(
            photo_path_obj,
            expected_output_dir,
            export_stem,
            stem,
        )

        # 计算等待超时：优先使用调用方给出的预算；环境变量只作为无 payload 时的默认值。
        # 上层 aiohttp 超时会比该预算更长，保证 bridge 先返回结构化失败并释放 Lightroom lock。
        wait_timeout = BRIDGE_WAIT_TIMEOUT
        if timeout is not None:
            try:
                wait_timeout = float(timeout)
            except (TypeError, ValueError):
                wait_timeout = BRIDGE_WAIT_TIMEOUT
        wait_timeout = max(1.0, wait_timeout)

        # 串行化“发送 + 等待”区段：同一时刻只有一个导出占用唯一的 Lightroom socket，
        # 使 last_error/last_output_path 在 ThreadingHTTPServer 下语义明确。
        with self._lr_lock:
            # 清空之前的输出路径与错误标记
            self.last_output_path = None
            self.last_error = None

            # Send an explicit output directory and filename stem to the plugin
            # so every main/baseline/probe render has a distinct target path.
            message = (
                f"process|{str(render_photo_path_obj)}|{str(Path(xmp_path).absolute())}|"
                f"{str(expected_output_dir)}|{task_id or ''}|{export_stem}"
            )

            request_started_at = time.time()
            known_output_mtimes = {}
            for existing_output in expected_output_dir.glob("*"):
                try:
                    if existing_output.is_file():
                        known_output_mtimes[str(existing_output)] = existing_output.stat().st_mtime
                except OSError:
                    continue

            result = self.send_request(message)
            if not result:
                return result, self.last_output_path

            import glob
            # 优先尝试唯一导出文件名；保留源文件名候选用于旧插件兼容。
            preferred_candidates = [
                expected_output_dir / f"{export_stem}.jpg",
                expected_output_dir / f"{export_stem}.jpeg",
                expected_output_dir / f"{export_stem}.tif",
                expected_output_dir / f"{export_stem}.tiff",
            ]
            if export_stem != stem:
                preferred_candidates.extend([
                    expected_output_dir / f"{stem}.jpg",
                    expected_output_dir / f"{stem}.jpeg",
                    expected_output_dir / f"{stem}.tif",
                    expected_output_dir / f"{stem}.tiff",
                ])
            possible_patterns = [
                str(expected_output_dir / "*.jpg"),
                str(expected_output_dir / "*.jpeg"),
                str(expected_output_dir / "*.tif"),
                str(expected_output_dir / "*.tiff"),
            ]

            deadline = time.time() + wait_timeout
            while True:
                # 插件直接返回了路径（兼容未来 BR-5）：立即采用并结束等待
                if self.last_output_path:
                    normalized_path = self.normalize_output_path(self.last_output_path, expected_output_path)
                    self.last_output_path = normalized_path
                    return result, normalized_path
                # 插件返回错误：尽快中止等待并抛出，使 do_POST 返回 500
                if self.last_error:
                    raise LightroomBridgeError(
                        "plugin_error",
                        f"Lightroom plugin error: {self.last_error}",
                        retryable=False,
                        details={"plugin_error": self.last_error},
                    )
                plugin_status = self.read_render_status(status_path)
                if plugin_status.get("status") == "error":
                    code = plugin_status.get("error_code") or "plugin_error"
                    message_text = plugin_status.get("message") or f"Lightroom plugin error: {code}"
                    retryable = str(plugin_status.get("retryable", "")).lower() == "true"
                    raise LightroomBridgeError(
                        code,
                        message_text,
                        retryable=retryable,
                        details={
                            "plugin_status": plugin_status,
                            "status_path": str(status_path),
                        },
                    )

                # 首选：与原文件同名的导出文件，要求体积 > 0
                for candidate in preferred_candidates:
                    try:
                        stat = candidate.stat()
                        old_mtime = known_output_mtimes.get(str(candidate))
                        is_fresh = old_mtime is None or stat.st_mtime > old_mtime or stat.st_mtime >= request_started_at
                        if candidate.exists() and stat.st_size > 0 and is_fresh:
                            self.last_output_path = self.normalize_output_path(candidate, expected_output_path)
                            print(f"Found expected output file: {self.last_output_path}")
                            return result, self.last_output_path
                    except OSError:
                        continue

                # 次选：目录下最新 mtime 的非空图片文件
                matches = []
                for pattern in possible_patterns:
                    matches.extend(glob.glob(pattern))
                fresh_matches = []
                for match in matches:
                    try:
                        match_path = Path(match)
                        stat = match_path.stat()
                        if not match_path.is_file() or stat.st_size <= 0:
                            continue

                        old_mtime = known_output_mtimes.get(str(match_path))
                        if old_mtime is None or stat.st_mtime > old_mtime or stat.st_mtime >= request_started_at:
                            fresh_matches.append(match)
                    except OSError:
                        continue
                if fresh_matches:
                    found_path = max(fresh_matches, key=lambda x: Path(x).stat().st_mtime)
                    self.last_output_path = self.normalize_output_path(found_path, expected_output_path)
                    print(f"Found output file by latest mtime: {self.last_output_path}")
                    return result, self.last_output_path

                # 超时：返回 None 使 do_POST 返回 500
                if time.time() >= deadline:
                    print(f"Timed out waiting for output file after {wait_timeout}s: {expected_output_path}")
                    self.recover_from_stall("export_timeout")
                    raise LightroomBridgeError(
                        "export_timeout",
                        f"Timed out waiting for output file after {wait_timeout:.1f}s",
                        retryable=True,
                        details={
                            "expected_output_dir": str(expected_output_dir),
                            "expected_stem": export_stem,
                            "source_stem": stem,
                            "source_photo_path": str(photo_path_obj),
                            "render_photo_path": str(render_photo_path_obj),
                            "wait_timeout": wait_timeout,
                            "recovery_command_configured": bool(BRIDGE_RECOVERY_COMMAND),
                            "plugin_status": self.read_render_status(status_path),
                        },
                    )

                time.sleep(min(BRIDGE_POLL_INTERVAL, max(0.0, deadline - time.time())))

    def read_render_status(self, status_path):
        """Read the plugin's per-render status file if present."""
        try:
            path = Path(status_path)
            if not path.exists():
                return {}
            raw = path.read_text(encoding="utf-8", errors="replace").strip()
            fields = {}
            for item in raw.split(";"):
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                fields[key.strip()] = value.strip()
            return fields
        except Exception as e:
            return {"status_read_error": str(e)}

    def recover_from_stall(self, reason):
        """Run an optional operator-provided recovery hook after Lightroom stalls."""
        if not BRIDGE_RECOVERY_COMMAND:
            return
        try:
            subprocess.Popen(
                BRIDGE_RECOVERY_COMMAND,
                shell=True,
                env={**os.environ, "LIGHTROOM_STALL_REASON": reason},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Started Lightroom recovery command for {reason}")
        except Exception as e:
            print(f"Failed to start Lightroom recovery command: {e}")

    def _parse_health_fields(self, health_text):
        parsed = {}
        for part in (health_text or "").split(";"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            parsed[key] = value
        return parsed

    def _read_health_file(self):
        try:
            if not BRIDGE_HEALTH_FILE.exists():
                return None
            stat = BRIDGE_HEALTH_FILE.stat()
            age = time.time() - stat.st_mtime
            if age > BRIDGE_HEALTH_FILE_MAX_AGE:
                return None
            parsed = self._parse_health_fields(BRIDGE_HEALTH_FILE.read_text().strip())
            parsed["health_file_age_seconds"] = f"{age:.1f}"
            return parsed
        except Exception as e:
            print(f"Failed to read Lightroom health file: {e}")
            return None

    def health(self):
        """Return best-effort plugin/bridge health for client registration."""
        with self._lr_lock:
            plugin_health = self._read_health_file()
            if plugin_health:
                return {
                    "connected": self.connected,
                    "status": "ready",
                    "plugin": plugin_health,
                }

            if not self.connected:
                self.close()
            self.last_output_path = None
            self.last_error = None
            self.last_health = None
            if not self.send_request("health"):
                return {
                    "connected": False,
                    "status": "bridge_disconnected",
                    "error": "Failed to send health request",
                }

            deadline = time.time() + 2.0
            while time.time() < deadline:
                if self.last_health is not None:
                    parsed = self._parse_health_fields(self.last_health)
                    return {
                        "connected": True,
                        "status": "ready",
                        "plugin": parsed,
                    }
                if self.last_error:
                    return {
                        "connected": True,
                        "status": "plugin_error",
                        "error": self.last_error,
                    }
                time.sleep(0.05)

            payload = {
                "connected": self.connected,
                "status": "ready" if self.connected else "bridge_disconnected",
            }
            return payload
    
    def close(self):
        """关闭连接"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None

def test_server_connection(api):
    """测试服务器连接"""
    if not api.connect():
        return False
        
    # 发送 ping 请求
    result = api.send_request("ping")
    if result:
        print("Server is running")
        return True
    
    print("Could not connect to server")
    return False

class PhotoProcessHandler(BaseHTTPRequestHandler):
    def send_json(self, status_code, payload):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_GET(self):
        if self.path not in ("/", "/health"):
            self.send_json(404, {"status": "error", "error": "not_found"})
            return

        payload = self.server.lightroom_api.health()
        self.send_json(200, payload)

    def do_POST(self):
        # Parse request defensively: a missing/invalid Content-Length or a
        # non-JSON body must yield a real HTTP 400 (not a dropped connection),
        # so callers that treat "any HTTP response" as "bridge alive" keep working.
        try:
            content_length = int(self.headers.get('Content-Length') or 0)
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            self.send_json(400, {
                "status": "error",
                "error_code": "bad_request",
                "error": f"Bad request: {e}",
                "retryable": False,
            })
            return

        photo_path = data.get('photo_path')
        xmp_path = data.get('xmp_path')
        
        if not photo_path or not xmp_path:
            self.send_json(400, {
                "status": "error",
                "error_code": "bad_request",
                "error": "Missing photo_path or xmp_path",
                "retryable": False,
            })
            return
            
        try:
            task_id = data.get('task_id', f"task_{int(time.time())}")
            output_dir = data.get('output_dir')
            output_basename = data.get('output_basename')
            timeout = data.get('timeout')

            result, output_path = self.server.lightroom_api.process_photo(
                photo_path,
                xmp_path,
                output_dir,
                timeout=timeout,
                task_id=task_id,
                output_basename=output_basename,
            )
            if result and output_path:
                response_data = {
                    "status": "success",
                    "output_path": output_path,
                    "task_id": task_id
                }

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode())
            else:
                self.send_json(500, {
                    "status": "error",
                    "error_code": "missing_output",
                    "error": "Failed to process photo",
                    "retryable": True,
                    "task_id": task_id,
                })
        except LightroomBridgeError as e:
            status_code = 504 if e.code == "export_timeout" else 500
            self.send_json(status_code, {
                "status": "error",
                "error_code": e.code,
                "error": e.message,
                "retryable": e.retryable,
                "task_id": data.get('task_id'),
                "details": e.details,
            })
        except FileNotFoundError as e:
            self.send_json(400, {
                "status": "error",
                "error_code": "input_missing",
                "error": str(e),
                "retryable": False,
                "task_id": data.get('task_id'),
            })
        except Exception as e:
            self.send_json(500, {
                "status": "error",
                "error_code": "bridge_exception",
                "error": str(e),
                "retryable": True,
                "task_id": data.get('task_id'),
            })

def run_http_server(api, port=7777):
    """运行HTTP服务器"""
    server = ThreadingHTTPServer(('127.0.0.1', port), PhotoProcessHandler)
    server.lightroom_api = api  # 将API实例附加到服务器
    print(f"Starting HTTP server on port {port}")
    server.serve_forever()

def main():
    parser = argparse.ArgumentParser(description="Lightroom bridge HTTP server")
    parser.add_argument('--port', type=int, default=7777,
                        help='HTTP server port (default: 7777)')
    args = parser.parse_args()

    # 创建API客户端
    api = LightroomAPI()

    try:
        # Probe the Lightroom plugin once, but keep the HTTP bridge running even
        # if Lightroom is still starting. Per-request code reconnects lazily, and
        # /health can still expose the plugin's file heartbeat.
        if not test_server_connection(api):
            print("Warning: Could not connect to the Lightroom plugin server; will retry on requests")

        # 启动HTTP服务器
        http_thread = threading.Thread(target=run_http_server, args=(api, args.port))
        http_thread.daemon = True
        http_thread.start()

        print("\nHTTP server is running. You can now send POST requests to process photos.")
        print(f"Example POST request to http://127.0.0.1:{args.port}:")
        print('''
        {
            "photo_path": "path/to/photo.dng",
            "xmp_path": "path/to/preset.xmp"
        }
        ''')
        
        # 保持主程序运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        api.close()

if __name__ == "__main__":
    main()
