"""
Reverse Connection Server - Linux server provides task queue
Clients actively connects to fetch tasks
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import time
import os
from typing import Deque, Dict, List, Optional, Any
from uuid import uuid4
from pathlib import Path
import logging
from enum import Enum
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lightroom Reverse Connection Service", version="1.0.0")

# Task status enumeration - simple state machine eliminates all special cases
class TaskStatus(str, Enum):
    PENDING = "pending"      # Waiting to be read
    READING = "reading"      # Has been read, waiting for processing
    PROCESSING = "processing" # Processing in progress
    COMPLETED = "completed"   # Processing completed
    FAILED = "failed"        # Processing failed

# Data models
class Task(BaseModel):
    task_id: str
    photo_path: str
    xmp_path: str
    created_at: float
    status: TaskStatus = TaskStatus.PENDING
    client_id: Optional[str] = None
    read_at: Optional[float] = None        # Read timestamp
    read_timeout: float = 10.0             # Read timeout in seconds
    result: Optional[Dict[str, Any]] = None
    file_package_path: Optional[str] = None
    requires_download: bool = False
    attempts: int = 0
    max_attempts: int = 3
    next_attempt_at: float = 0.0
    error_history: List[Dict[str, Any]] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)

class TaskResult(BaseModel):
    task_id: str
    client_id: str
    success: bool
    elapsed_time: float
    error: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None

class ClientInfo(BaseModel):
    client_id: str
    client_type: str = "lightroom_bridge"
    capabilities: List[str] = ["photo_processing"]
    status: str = "ready"
    last_seen: float
    local_port: int = 7777
    health: Dict[str, Any] = Field(default_factory=dict)
    catalog_count: Optional[int] = None
    recent_render_p95: Optional[float] = None
    disk_free_bytes: Optional[int] = None

# Backpressure thresholds. Defaults are permissive except the statuses that
# explicitly mean "do not assign work".
MIN_CLIENT_DISK_FREE_BYTES = int(os.getenv("LIGHTROOM_MIN_CLIENT_DISK_FREE_BYTES", str(5 * 1024 * 1024 * 1024)))
MAX_CLIENT_CATALOG_COUNT = int(os.getenv("LIGHTROOM_MAX_CLIENT_CATALOG_COUNT", "120"))

# Global storage - message center core data structures
tasks: Dict[str, Task] = {}
clients: Dict[str, ClientInfo] = {}
completed_tasks: Dict[str, Task] = {}
pending_task_ids: Deque[str] = deque()
completed_task_ids: Deque[str] = deque()
recent_completed_task_ids: Deque[str] = deque(maxlen=10)
active_status_counts: Dict[TaskStatus, int] = {status: 0 for status in TaskStatus}
completed_status_counts: Dict[TaskStatus, int] = {
    TaskStatus.COMPLETED: 0,
    TaskStatus.FAILED: 0,
}

# Task status lock/condition - ensures atomic operations and wakes long-polls
task_lock = asyncio.Lock()
task_condition = asyncio.Condition(task_lock)

# File transfer storage - use environment variables to configure paths
file_packages: Dict[str, Dict[str, str]] = {}  # task_id -> {"photo_path": ..., "xmp_path": ...}
upload_dir = Path(os.getenv('LIGHTROOM_UPLOAD_DIR', './lr_caches/uploads'))
upload_dir.mkdir(parents=True, exist_ok=True)

# Processing result storage - separate folder for each task
results_dir = Path(os.getenv('LIGHTROOM_RESULTS_DIR', './lr_caches/lightroom_results'))
results_dir.mkdir(parents=True, exist_ok=True)
pending_uploads: Dict[str, Dict[str, Any]] = {}


def _task_masks_dir(task_id: str) -> Path:
    return results_dir / task_id / "masks"


def _list_durable_masks(task_id: str) -> List[Dict[str, Any]]:
    masks_dir = _task_masks_dir(task_id)
    if not masks_dir.exists():
        return []
    masks = []
    for path in sorted(masks_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".pgm"}:
            masks.append({
                "mask_id": path.stem,
                "path": str(path),
                "filename": path.name,
                "file_size": path.stat().st_size,
            })
    return masks


def _percentile(values: List[float], ratio: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * ratio)))
    return ordered[index]


def _result_data(task: Task) -> Dict[str, Any]:
    if not task.result:
        return {}
    data = task.result.get("result_data")
    return data if isinstance(data, dict) else {}


def _error_code(task: Task) -> str:
    data = _result_data(task)
    code = data.get("error_code")
    if code:
        return str(code)
    error = task.result.get("error") if task.result else None
    if error:
        return "error"
    return "unknown"


def _completed_metrics() -> Dict[str, Any]:
    completed = list(completed_tasks.values())
    latencies = [
        float(task.result["elapsed_time"])
        for task in completed
        if task.result and isinstance(task.result.get("elapsed_time"), (int, float))
    ]
    successes = [task for task in completed if task.status == TaskStatus.COMPLETED]
    failures = [task for task in completed if task.status == TaskStatus.FAILED]
    failure_counts: Dict[str, int] = {}
    for task in failures:
        code = _error_code(task)
        failure_counts[code] = failure_counts.get(code, 0) + 1

    total = len(completed)
    return {
        "total_terminal_tasks": total,
        "successful_tasks": len(successes),
        "failed_tasks": len(failures),
        "success_rate": (len(successes) / total) if total else None,
        "render_latency_p50": _percentile(latencies, 0.50),
        "render_latency_p95": _percentile(latencies, 0.95),
        "failure_counts": failure_counts,
    }

# Retry configuration
MAX_FILE_WAIT_RETRIES = int(os.getenv('LIGHTROOM_MAX_RETRIES', '5'))
FILE_WAIT_TIMEOUT = float(os.getenv('LIGHTROOM_FILE_WAIT_TIMEOUT', '180.0'))
RETRY_DELAY = float(os.getenv('LIGHTROOM_RETRY_DELAY', '2.0'))
BACKOFF_FACTOR = float(os.getenv('LIGHTROOM_BACKOFF_FACTOR', '1.5'))

# Long-poll configuration
LONG_POLL_MAX = float(os.getenv('LIGHTROOM_LONG_POLL_MAX', '25.0'))

# Cleanup loop configuration
CLEANUP_INTERVAL = float(os.getenv('LIGHTROOM_CLEANUP_INTERVAL', '10.0'))
COMPLETED_TTL = float(os.getenv('LIGHTROOM_COMPLETED_TTL', '3600.0'))
LIGHTROOM_TASK_MAX_ATTEMPTS = int(os.getenv("LIGHTROOM_TASK_MAX_ATTEMPTS", "3"))
LIGHTROOM_TASK_RETRY_BASE_DELAY = float(os.getenv("LIGHTROOM_TASK_RETRY_BASE_DELAY", "5.0"))


def _active_count(status: TaskStatus) -> int:
    return active_status_counts.get(status, 0)


def _add_active_task(task: Task) -> None:
    tasks[task.task_id] = task
    active_status_counts[task.status] += 1
    if task.status == TaskStatus.PENDING:
        pending_task_ids.append(task.task_id)


def _set_active_task_status(task: Task, status: TaskStatus) -> None:
    if task.status == status:
        return

    active_status_counts[task.status] -= 1
    task.status = status
    active_status_counts[task.status] += 1


def _pop_next_pending_task() -> Optional[Task]:
    now = time.time()
    for _ in range(len(pending_task_ids)):
        task_id = pending_task_ids.popleft()
        task = tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING and task.next_attempt_at <= now:
            return task
        if task and task.status == TaskStatus.PENDING:
            pending_task_ids.append(task_id)
    return None


def _seconds_until_next_pending_task(now: Optional[float] = None) -> Optional[float]:
    now = time.time() if now is None else now
    waits = [
        max(0.0, task.next_attempt_at - now)
        for task_id in pending_task_ids
        if (task := tasks.get(task_id)) is not None and task.status == TaskStatus.PENDING
    ]
    return min(waits) if waits else None


def _completed_count(status: Optional[TaskStatus] = None) -> int:
    if status is None:
        return len(completed_tasks)
    return completed_status_counts.get(status, 0)


def _store_completed_task(task: Task) -> None:
    completed_tasks[task.task_id] = task
    completed_task_ids.append(task.task_id)
    recent_completed_task_ids.append(task.task_id)
    if task.status in completed_status_counts:
        completed_status_counts[task.status] += 1


def _evict_completed_task(task_id: str) -> None:
    task = completed_tasks.pop(task_id, None)
    if task and task.status in completed_status_counts:
        completed_status_counts[task.status] -= 1
    file_packages.pop(task_id, None)
    pending_uploads.pop(task_id, None)


def _serialize_task_status(task_id: str, task: Task) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "status": task.status,
        "created_at": task.created_at,
        "client_id": task.client_id,
        "result": task.result,
        "options": task.options,
    }


def _client_can_accept_task(client: ClientInfo) -> tuple[bool, Optional[str]]:
    if client.status in {"busy", "blocked", "draining", "resetting"}:
        return False, f"client status={client.status}"
    if client.disk_free_bytes is not None and client.disk_free_bytes < MIN_CLIENT_DISK_FREE_BYTES:
        return False, f"low disk free={client.disk_free_bytes}"
    if client.catalog_count is not None and client.catalog_count > MAX_CLIENT_CATALOG_COUNT:
        return False, f"catalog count={client.catalog_count}"

    bridge_health = client.health.get("bridge") if isinstance(client.health, dict) else None
    if isinstance(bridge_health, dict) and bridge_health.get("status") not in (None, "ready"):
        return False, f"bridge status={bridge_health.get('status')}"

    return True, None

async def wait_for_file_with_retries(file_path: str, timeout: float = None) -> bool:
    """
    File waiting function with retry mechanism
    Handles Mac Lightroom transmission delay issues
    """
    if timeout is None:
        timeout = FILE_WAIT_TIMEOUT
    
    start_time = time.time()
    retry_count = 0
    current_delay = RETRY_DELAY
    
    while time.time() - start_time < timeout and retry_count < MAX_FILE_WAIT_RETRIES:
        if Path(file_path).exists():
            # Additional wait after file exists to ensure write completion
            await asyncio.sleep(0.5)
            if Path(file_path).exists() and Path(file_path).stat().st_size > 0:
                logger.info(f"File ready: {file_path} (retry {retry_count} times)")
                return True
        
        logger.debug(f"Waiting for file {file_path}... (retry {retry_count}/{MAX_FILE_WAIT_RETRIES})")
        await asyncio.sleep(current_delay)
        
        # Exponential backoff
        current_delay = min(current_delay * BACKOFF_FACTOR, 30.0)  # Maximum 30 seconds
        retry_count += 1
    
    logger.error(f"File wait timeout: {file_path} (elapsed {time.time() - start_time:.1f}s, retry {retry_count} times)")
    return False

@app.get("/")
async def root():
    return {
        "message": "Lightroom reverse connection service is running (message center mode)",
        "version": "2.0.0",
        "active_clients": len(clients),
        "task_stats": {
            "pending": _active_count(TaskStatus.PENDING),
            "reading": _active_count(TaskStatus.READING),
            "processing": _active_count(TaskStatus.PROCESSING),
            "completed": _completed_count(TaskStatus.COMPLETED),
            "failed": _completed_count(TaskStatus.FAILED),
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "stats": {
            "active_clients": len(clients),
            "pending_tasks": _active_count(TaskStatus.PENDING),
            "reading_tasks": _active_count(TaskStatus.READING),
            "processing_tasks": _active_count(TaskStatus.PROCESSING),
            "completed_tasks": _completed_count(TaskStatus.COMPLETED),
            "failed_tasks": _completed_count(TaskStatus.FAILED),
        }
    }

@app.post("/api/register_client")
async def register_client(client: ClientInfo):
    """Register Mac client"""
    client.last_seen = time.time()
    async with task_lock:
        clients[client.client_id] = client
    logger.info(f"Client registered: {client.client_id}")
    return {"message": "Registration successful", "client_id": client.client_id}

@app.get("/api/get_task/{client_id}")
async def get_task(client_id: str, wait: float = 0.0):
    """Mac client fetches task - atomic FIFO operation, no race conditions

    Optional long-poll: when ``wait`` (seconds) > 0 the server waits on a
    condition variable until a task appears or the capped deadline passes,
    returning ``{}`` on timeout. Default ``wait=0.0`` reproduces the original
    immediate short-poll behavior.
    """
    # Add debug logging
    logger.debug(f"Received task request, client ID: {client_id}")
    logger.debug(f"Currently registered clients: {list(clients.keys())}")

    wait_seconds = min(max(wait, 0.0), LONG_POLL_MAX)
    deadline = time.monotonic() + wait_seconds

    async with task_condition:
        if client_id not in clients:
            # Return more explicit error instead of default 404
            logger.warning(f"Client {client_id} not registered")
            raise HTTPException(status_code=403, detail="Client not registered, please register first")

        while True:
            # Update client last active time
            client = clients[client_id]
            client.last_seen = time.time()

            can_accept, reason = _client_can_accept_task(client)
            if not can_accept:
                logger.info(f"Client {client_id} not eligible for new task: {reason}")
                if wait_seconds <= 0:
                    return {}

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {}

                try:
                    await asyncio.wait_for(task_condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return {}
                continue

            available_task = _pop_next_pending_task()
            if available_task:
                # Atomic state transition: pending -> reading
                _set_active_task_status(available_task, TaskStatus.READING)
                available_task.client_id = client_id
                available_task.read_at = time.time()
                available_task.attempts += 1

                logger.info(f"Task {available_task.task_id} read by client {client_id}")

                return {
                    "task_id": available_task.task_id,
                    "photo_path": available_task.photo_path,
                    "xmp_path": available_task.xmp_path,
                    "created_at": available_task.created_at,
                    "requires_download": available_task.requires_download,
                    "file_package_path": available_task.file_package_path,
                    "read_timeout": available_task.read_timeout,
                    "attempt": available_task.attempts,
                    "max_attempts": available_task.max_attempts,
                    "options": available_task.options,
                }

            if wait_seconds <= 0:
                return {}  # No available tasks

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {}

            retry_wait = _seconds_until_next_pending_task()
            wait_timeout = remaining if retry_wait is None else min(remaining, retry_wait)

            try:
                await asyncio.wait_for(task_condition.wait(), timeout=wait_timeout)
            except asyncio.TimeoutError:
                if retry_wait is not None and retry_wait <= remaining:
                    continue
                return {}

class StartProcessingRequest(BaseModel):
    client_id: str

@app.post("/api/start_processing/{task_id}")
async def start_processing(task_id: str, request: StartProcessingRequest):
    """Client confirms start processing task - state transition reading -> processing"""
    async with task_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="Task does not exist")
        
        task = tasks[task_id]
        if task.client_id != request.client_id:
            raise HTTPException(status_code=403, detail="Task does not belong to this client")
        
        if task.status != TaskStatus.READING:
            raise HTTPException(status_code=400, detail=f"Task status error: {task.status}")
        
        # Atomic state transition: reading -> processing
        _set_active_task_status(task, TaskStatus.PROCESSING)
        logger.info(f"Task {task_id} started processing (client: {request.client_id})")
        
        return {"message": "Processing started", "task_id": task_id}

@app.post("/api/report_result")
async def report_result(result: TaskResult):
    """Mac client reports task result - atomic state transition"""
    async with task_condition:
        task_id = result.task_id
        
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="Task does not exist")
        
        task = tasks[task_id]
        if task.client_id != result.client_id:
            raise HTTPException(status_code=403, detail="Task does not belong to this client")
        
        result_payload = {
            "success": result.success,
            "elapsed_time": result.elapsed_time,
            "error": result.error,
            "result_data": result.result_data,
            "completed_at": time.time(),
            "attempt": task.attempts,
            "max_attempts": task.max_attempts,
        }

        if result.success:
            _set_active_task_status(task, TaskStatus.COMPLETED)
            task.result = result_payload
            if task_id in pending_uploads:
                task.result.update(pending_uploads.pop(task_id))
            durable_masks = _list_durable_masks(task_id)
            if durable_masks:
                existing_masks = task.result.get("masks") if isinstance(task.result, dict) else None
                if isinstance(existing_masks, list):
                    by_id = {mask.get("mask_id"): mask for mask in existing_masks if isinstance(mask, dict)}
                    for mask in durable_masks:
                        by_id[mask.get("mask_id")] = mask
                    task.result["masks"] = list(by_id.values())
                else:
                    task.result["masks"] = durable_masks

            _store_completed_task(task)
            del tasks[task_id]
            active_status_counts[task.status] -= 1
            file_packages.pop(task_id, None)

            logger.info(f"Task {task_id} ✅Success (elapsed: {result.elapsed_time:.1f}s)")
            task_condition.notify_all()
            return {"message": "Result recorded", "task_id": task_id}

        task.error_history.append(result_payload)
        result_data = result.result_data or {}
        retryable = bool(result_data.get("retryable")) if isinstance(result_data, dict) else False
        if retryable and task.attempts < task.max_attempts:
            delay = LIGHTROOM_TASK_RETRY_BASE_DELAY * (2 ** max(0, task.attempts - 1))
            task.next_attempt_at = time.time() + delay
            task.client_id = None
            task.read_at = None
            task.result = result_payload
            _set_active_task_status(task, TaskStatus.PENDING)
            pending_task_ids.append(task_id)
            logger.warning(
                f"Task {task_id} transient failure; retry {task.attempts}/{task.max_attempts} "
                f"after {delay:.1f}s: {result.error}"
            )
            task_condition.notify_all()
            return {
                "message": "Result recorded; task scheduled for retry",
                "task_id": task_id,
                "retry": True,
                "attempt": task.attempts,
                "max_attempts": task.max_attempts,
                "next_attempt_at": task.next_attempt_at,
            }
        
        # Atomic state transition: processing -> failed
        _set_active_task_status(task, TaskStatus.FAILED)
        task.result = result_payload
        if isinstance(result_data, dict):
            task.result["retryable"] = retryable

        _store_completed_task(task)
        del tasks[task_id]
        active_status_counts[task.status] -= 1

        # Drop the source-file index for this task - no longer needed once the
        # client has finished processing (prevents file_packages memory leak).
        file_packages.pop(task_id, None)

        logger.info(f"Task {task_id} ❌Failed (elapsed: {result.elapsed_time:.1f}s)")
        task_condition.notify_all()

        return {"message": "Result recorded", "task_id": task_id}

@app.post("/api/upload_result")
async def upload_result(task_id: str, processed_image: UploadFile = File(...)):
    """Receive processed result image uploaded by Mac client.

    New clients upload before reporting completion so the training side sees a
    completed task only after the result file metadata is available. Old clients
    that still report before upload remain supported.
    """

    async with task_lock:
        task_exists = task_id in tasks or task_id in completed_tasks

    if not task_exists:
        raise HTTPException(status_code=404, detail="Task does not exist or not completed")
    
    # Create separate folder for each task
    task_result_dir = results_dir / task_id
    task_result_dir.mkdir(exist_ok=True)
    
    try:
        # Save processed image
        processed_filename = f"processed.jpg"
        processed_path = task_result_dir / processed_filename
        
        # Write to temporary file, then atomically move
        temp_path = processed_path.with_suffix('.tmp')
        
        with open(temp_path, "wb") as f:
            while True:
                chunk = await processed_image.read(1024 * 1024)
                if not chunk:
                    break
                await asyncio.to_thread(f.write, chunk)

        # Atomically move to final location
        temp_path.rename(processed_path)

        # Verify the saved file is present and non-empty (no sleep needed:
        # the upload body is fully received before the atomic rename above).
        if not (processed_path.exists() and processed_path.stat().st_size > 0):
            raise Exception("Saved file missing/empty")
        
        upload_metadata = {
            "processed_image_path": str(processed_path),
            "upload_timestamp": time.time(),
            "file_size": processed_path.stat().st_size
        }

        async with task_lock:
            if task_id in completed_tasks:
                task = completed_tasks[task_id]
                if not task.result:
                    task.result = {}
                task.result.update(upload_metadata)
            elif task_id in tasks:
                pending_uploads[task_id] = upload_metadata
            else:
                raise HTTPException(status_code=404, detail="Task disappeared before upload metadata update")
        
        logger.info(f"Task {task_id} processing result saved: {processed_path} ({processed_path.stat().st_size} bytes)")
        
        return {
            "message": "Processing result uploaded successfully", 
            "task_id": task_id,
            "saved_path": str(processed_path),
            "result_folder": str(task_result_dir),
            "file_size": processed_path.stat().st_size
        }
        
    except Exception as e:
        # Clean up temporary file
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        
        logger.error(f"Failed to save processing result {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@app.post("/api/upload_mask_result")
async def upload_mask_result(task_id: str, mask_id: str, mask_image: UploadFile = File(...)):
    """Receive one exported mask image from a Lightroom client."""

    safe_mask_id = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in mask_id).strip("_")
    if not safe_mask_id:
        raise HTTPException(status_code=400, detail="mask_id is required")
    suffix = Path(mask_image.filename or "").suffix.lower()
    if suffix not in {".png", ".pgm"}:
        suffix = ".png"

    async with task_lock:
        task_exists = task_id in tasks or task_id in completed_tasks

    if not task_exists:
        raise HTTPException(status_code=404, detail="Task does not exist or not completed")

    masks_dir = _task_masks_dir(task_id)
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_path = masks_dir / f"{safe_mask_id}{suffix}"
    temp_path = mask_path.with_suffix(".tmp")

    try:
        with open(temp_path, "wb") as f:
            while True:
                chunk = await mask_image.read(1024 * 1024)
                if not chunk:
                    break
                await asyncio.to_thread(f.write, chunk)

        temp_path.rename(mask_path)
        if not (mask_path.exists() and mask_path.stat().st_size > 0):
            raise Exception("Saved mask missing/empty")

        mask_metadata = {
            "mask_id": safe_mask_id,
            "path": str(mask_path),
            "filename": mask_path.name,
            "upload_timestamp": time.time(),
            "file_size": mask_path.stat().st_size,
        }

        async with task_lock:
            if task_id in completed_tasks:
                task = completed_tasks[task_id]
                if not task.result:
                    task.result = {}
                masks = task.result.setdefault("masks", [])
                if isinstance(masks, list):
                    masks[:] = [m for m in masks if not (isinstance(m, dict) and m.get("mask_id") == safe_mask_id)]
                    masks.append(mask_metadata)
            elif task_id in tasks:
                pending = pending_uploads.setdefault(task_id, {})
                masks = pending.setdefault("masks", [])
                if isinstance(masks, list):
                    masks[:] = [m for m in masks if not (isinstance(m, dict) and m.get("mask_id") == safe_mask_id)]
                    masks.append(mask_metadata)
            else:
                raise HTTPException(status_code=404, detail="Task disappeared before mask metadata update")

        logger.info(f"Task {task_id} mask saved: {mask_path} ({mask_path.stat().st_size} bytes)")
        return {
            "message": "Mask uploaded successfully",
            "task_id": task_id,
            "mask_id": safe_mask_id,
            "saved_path": str(mask_path),
            "file_size": mask_path.stat().st_size,
        }
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Failed to save mask {task_id}/{safe_mask_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")

@app.post("/api/submit_task_with_files")
async def submit_task_with_files(photo_path: str, xmp_path: str, export_masks: bool = False):
    """Submit task requiring file transfer - efficient direct transfer"""
    task_id = str(uuid4())
    
    # Check if files exist
    if not Path(photo_path).exists():
        raise HTTPException(status_code=404, detail=f"Photo file does not exist: {photo_path}")
    if not Path(xmp_path).exists():
        raise HTTPException(status_code=404, detail=f"XMP file does not exist: {xmp_path}")
    
    # Directly store source file paths, no packaging needed
    task = Task(
        task_id=task_id,
        photo_path=f"/tmp/lightroom_task_{task_id}/before.jpg",  # Mac local path
        xmp_path=f"/tmp/lightroom_task_{task_id}/config.lua",   # Mac local path
        created_at=time.time(),
        requires_download=True,
        file_package_path=None,  # No packaging used
        max_attempts=LIGHTROOM_TASK_MAX_ATTEMPTS,
        options={"export_masks": export_masks},
    )
    
    async with task_condition:
        _add_active_task(task)

        # Store source file paths for download use
        file_packages[task_id] = {
            "photo_path": photo_path,
            "xmp_path": xmp_path
        }
        task_condition.notify()
    
    logger.info(f"Created efficient transfer task: {task_id}")
    return {"message": "Task submitted", "task_id": task_id}

@app.get("/api/download_file/{task_id}/{file_type}")
async def download_file(task_id: str, file_type: str):
    """Download task files - efficient direct transfer"""
    if task_id not in file_packages:
        raise HTTPException(status_code=404, detail="Task files do not exist")
    
    file_paths = file_packages[task_id]
    
    if file_type == "photo":
        file_path = file_paths["photo_path"]
        filename = "before.jpg"
        media_type = "image/jpeg"
    elif file_type == "xmp":
        file_path = file_paths["xmp_path"] 
        filename = "config.lua"
        media_type = "text/plain"
    else:
        raise HTTPException(status_code=400, detail="Invalid file type, supported: photo, xmp")
    
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"File does not exist: {file_path}")

    def iterfile(path: str):
        with open(path, mode="rb") as file_like:
            while True:
                chunk = file_like.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return StreamingResponse(iterfile(file_path), media_type=media_type, headers=headers)

# Efficient transfer, no packaging function needed

@app.post("/api/submit_task")
async def submit_task(photo_path: str, xmp_path: str, export_masks: bool = False):
    """Submit new task"""
    task_id = str(uuid4())
    task = Task(
        task_id=task_id,
        photo_path=photo_path,
        xmp_path=xmp_path,
        created_at=time.time(),
        max_attempts=LIGHTROOM_TASK_MAX_ATTEMPTS,
        options={"export_masks": export_masks},
    )
    
    async with task_condition:
        _add_active_task(task)
        queue_position = _active_count(TaskStatus.PENDING)
        task_condition.notify()
    
    logger.info(f"New task submitted: {task_id}")
    logger.info(f"  Photo: {Path(photo_path).name}")
    logger.info(f"  XMP: {Path(xmp_path).name}")
    
    return {
        "message": "Task submitted",
        "task_id": task_id,
        "queue_position": queue_position
    }

@app.get("/api/task_status/{task_id}")
async def get_task_status(task_id: str, wait: float = 0.0):
    """Query task status

    Optional long-poll: when ``wait`` (seconds) > 0 the server waits on a
    condition variable until the task reaches a terminal state
    (completed/failed) or the capped deadline passes, at which point it returns
    the same non-terminal JSON shape it would return today. Default ``wait=0.0``
    reproduces the original immediate behavior.
    """
    wait_seconds = min(max(wait, 0.0), LONG_POLL_MAX)
    deadline = time.monotonic() + wait_seconds

    async with task_condition:
        while True:
            # Check in-progress tasks
            if task_id in tasks:
                task = tasks[task_id]
                current = _serialize_task_status(task_id, task)
                terminal = task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            # Check completed tasks
            elif task_id in completed_tasks:
                task = completed_tasks[task_id]
                return _serialize_task_status(task_id, task)
            elif (results_dir / task_id / "processed.jpg").exists():
                durable_masks = _list_durable_masks(task_id)
                result = {
                    "success": True,
                    "processed_image_path": str(results_dir / task_id / "processed.jpg"),
                    "durable_store": True,
                }
                if durable_masks:
                    result["masks"] = durable_masks
                return {
                    "task_id": task_id,
                    "status": TaskStatus.COMPLETED,
                    "result": result,
                }
            else:
                raise HTTPException(status_code=404, detail="Task does not exist")

            if terminal or wait_seconds <= 0:
                return current

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return current

            try:
                await asyncio.wait_for(task_condition.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return current

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    active_clients = [
        {
            "client_id": client_id,
            "last_seen": info.last_seen,
            "status": info.status,
            "capabilities": info.capabilities,
            "catalog_count": info.catalog_count,
            "recent_render_p95": info.recent_render_p95,
            "disk_free_bytes": info.disk_free_bytes,
            "eligible": _client_can_accept_task(info)[0],
            "ineligible_reason": _client_can_accept_task(info)[1],
        }
        for client_id, info in clients.items()
        if time.time() - info.last_seen < 60  # Active within 60 seconds
    ]
    
    return {
        "active_clients": active_clients,
        "queue_stats": {
            "pending": _active_count(TaskStatus.PENDING),
            "reading": _active_count(TaskStatus.READING),
            "processing": _active_count(TaskStatus.PROCESSING),
            "completed_today": len(completed_tasks)
        },
        "render_metrics": _completed_metrics(),
        "recent_tasks": [
            {
                "task_id": task.task_id,
                "status": task.status,
                "created_at": task.created_at,
                "photo": Path(task.photo_path).name,
                "elapsed_time": task.result.get("elapsed_time") if task.result else None,
                "error_code": _error_code(task) if task.status == TaskStatus.FAILED else None,
            }
            for task_id in recent_completed_task_ids
            if (task := completed_tasks.get(task_id)) is not None
        ]
    }

@app.get("/api/clients")
async def list_clients():
    """List all clients"""
    current_time = time.time()
    return {
        "clients": [
            {
                "client_id": client_id,
                "status": "online" if current_time - info.last_seen < 30 else "offline",
                "client_status": info.status,
                "last_seen": info.last_seen,
                "capabilities": info.capabilities,
                "local_port": info.local_port,
                "catalog_count": info.catalog_count,
                "recent_render_p95": info.recent_render_p95,
                "disk_free_bytes": info.disk_free_bytes,
                "health": info.health,
                "eligible": _client_can_accept_task(info)[0],
                "ineligible_reason": _client_can_accept_task(info)[1],
            }
            for client_id, info in clients.items()
        ]
    }

# Background task for cleaning up expired tasks
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_tasks())

async def cleanup_old_tasks():
    """Periodically clean up expired tasks and timeout reads - Linus-style concise logic"""
    while True:
        try:
            current_time = time.time()
            
            async with task_condition:
                # Clean up timed-out READING status tasks - reset to pending for other clients to execute
                reading_timeout_tasks = []
                for task_id, task in tasks.items():
                    if (task.status == TaskStatus.READING and 
                        task.read_at and 
                        current_time - task.read_at > task.read_timeout):
                        reading_timeout_tasks.append(task_id)
                
                for task_id in reading_timeout_tasks:
                    task = tasks[task_id]
                    logger.warning(f"Task {task_id} read timeout, reset to pending (client: {task.client_id})")
                    # Atomic state transition: reading -> pending
                    _set_active_task_status(task, TaskStatus.PENDING)
                    task.client_id = None
                    task.read_at = None
                    pending_task_ids.append(task_id)
                
                # Clean up long-running PROCESSING tasks (30 minutes)
                processing_timeout_tasks = []
                for task_id, task in tasks.items():
                    if (task.status == TaskStatus.PROCESSING and 
                        current_time - task.created_at > 1800):
                        processing_timeout_tasks.append(task_id)
                
                for task_id in processing_timeout_tasks:
                    task = tasks[task_id]
                    logger.warning(f"Task {task_id} processing timeout, reset to pending (client: {task.client_id})")
                    # Atomic state transition: processing -> pending
                    _set_active_task_status(task, TaskStatus.PENDING)
                    task.client_id = None
                    task.read_at = None
                    pending_task_ids.append(task_id)

                if reading_timeout_tasks or processing_timeout_tasks:
                    task_condition.notify_all()
            
            async with task_lock:
                offline_clients = [
                    client_id for client_id, info in clients.items()
                    if current_time - info.last_seen > 600
                ]

                for client_id in offline_clients:
                    logger.info(f"Cleaning up offline client: {client_id}")
                    del clients[client_id]

            # Evict old completed tasks past their TTL to bound memory growth.
            # TTL must exceed manager max_wait (300s) + grace (120s); default
            # 3600s is safe. Also drop any lingering source-file index entries.
            while completed_task_ids:
                tid = completed_task_ids[0]
                task = completed_tasks.get(tid)
                if task is None:
                    completed_task_ids.popleft()
                    continue

                completed_at = task.result.get("completed_at", 0) if task.result else 0
                if current_time - completed_at <= COMPLETED_TTL:
                    break

                completed_task_ids.popleft()
                logger.info(f"Evicting expired completed task: {tid}")
                _evict_completed_task(tid)

            await asyncio.sleep(CLEANUP_INTERVAL)  # Check periodically for more timely timeout handling

        except Exception as e:
            logger.error(f"Cleanup task exception: {e}")
            await asyncio.sleep(60)



@app.get("/api/download_task_result/{task_id}")
async def download_task_result(task_id: str):
    """Download task processing result image"""

    processed_path = None
    if task_id in completed_tasks:
        task = completed_tasks[task_id]
        if task.result and "processed_image_path" in task.result:
            processed_path = task.result["processed_image_path"]

    if processed_path is None:
        durable_path = results_dir / task_id / "processed.jpg"
        if durable_path.exists():
            processed_path = str(durable_path)
        else:
            raise HTTPException(status_code=404, detail="Task result does not exist")
    
    # Check if file exists
    if not Path(processed_path).exists():
        raise HTTPException(status_code=404, detail=f"Result file does not exist: {processed_path}")
    
    # Stream file
    def iterfile(path: str):
        with open(path, mode="rb") as file_like:
            while True:
                chunk = file_like.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

    # Get file extension to determine media type
    file_ext = Path(processed_path).suffix.lower()
    media_type = "image/jpeg" if file_ext in ['.jpg', '.jpeg'] else "application/octet-stream"
    
    headers = {
        "Content-Disposition": f"attachment; filename=processed_{task_id}.jpg"
    }
    
    logger.info(f"Downloading task result: {task_id} -> {processed_path}")
    return StreamingResponse(iterfile(processed_path), media_type=media_type, headers=headers)


@app.get("/api/list_task_masks/{task_id}")
async def list_task_masks(task_id: str):
    """List durable mask images for a completed task."""

    masks = []
    if task_id in completed_tasks:
        task = completed_tasks[task_id]
        task_masks = task.result.get("masks") if task.result else None
        if isinstance(task_masks, list):
            masks.extend([mask for mask in task_masks if isinstance(mask, dict)])

    durable_masks = _list_durable_masks(task_id)
    if durable_masks:
        by_id = {mask.get("mask_id"): mask for mask in masks}
        for mask in durable_masks:
            by_id[mask.get("mask_id")] = mask
        masks = list(by_id.values())

    return {"task_id": task_id, "masks": masks}


@app.get("/api/download_task_mask/{task_id}/{mask_id}")
async def download_task_mask(task_id: str, mask_id: str):
    """Download one durable mask image."""

    safe_mask_id = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in mask_id).strip("_")
    if not safe_mask_id:
        raise HTTPException(status_code=400, detail="mask_id is required")

    masks_dir = _task_masks_dir(task_id)
    mask_path = None
    for candidate in [masks_dir / f"{safe_mask_id}.png", masks_dir / f"{safe_mask_id}.pgm"]:
        if candidate.exists() and candidate.is_file():
            mask_path = candidate
            break
    if mask_path is None:
        matches = sorted(masks_dir.glob(f"{safe_mask_id}.*")) if masks_dir.exists() else []
        matches = [path for path in matches if path.suffix.lower() in {".png", ".pgm"} and path.is_file()]
        mask_path = matches[0] if matches else None
    if mask_path is None or not (mask_path.exists() and mask_path.is_file()):
        raise HTTPException(status_code=404, detail="Task mask does not exist")

    def iterfile(path: Path):
        with open(path, mode="rb") as file_like:
            while True:
                chunk = file_like.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

    headers = {
        "Content-Disposition": f"attachment; filename={safe_mask_id}_{task_id}{mask_path.suffix}"
    }
    media_type = "image/png" if mask_path.suffix.lower() == ".png" else "image/x-portable-graymap"
    return StreamingResponse(iterfile(mask_path), media_type=media_type, headers=headers)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Lightroom Reverse Connection Server')
    parser.add_argument('--host', default='0.0.0.0', help='Listen address')
    parser.add_argument('--port', type=int, default=8081, help='Listen port')
    
    args = parser.parse_args()
    
    print("🚀 Starting Lightroom Reverse Connection Server")
    print("=" * 40)
    print(f"Listening: {args.host}:{args.port}")
    print("API Documentation: http://localhost:8081/docs")
    print("=" * 40)
    
    uvicorn.run(app, host=args.host, port=args.port)
