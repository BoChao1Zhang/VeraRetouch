#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reverse connection client - Runs on Mac, actively connects to Linux servers to fetch tasks
Adapted to message center state machine: pending -> reading -> processing -> completed/failed
"""

from __future__ import annotations

import asyncio
import aiohttp
import json
import argparse
import time
import logging
import os
import requests
import shutil
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaskExportError(Exception):
    """Raised when an export_masks task cannot produce the requested masks."""

    def __init__(self, message: str, *, retryable: bool = False, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.retryable = retryable
        self.details = details or {}


class LightroomReverseClient:
    """Mac reverse connection client - Supports multi-server polling - Adapted to message center state machine"""
    
    def __init__(self, servers: str = None, server_ip: str = None, server_port: int = 8080, 
                 local_port: int = 7777, client_id: str = None, poll_interval: float = 2.0,
                 http_timeout_total: float = 300.0,
                 connector_limit: int = 10,
                 read_timeout_default: float = 10.0,
                 base_processing_timeout: float = 10.0,
                 mask_increment_seconds: float = 2.0,
                 complex_increment_seconds: float = 3.0,
                 processing_extra_buffer: float = 10.0,
                 max_timeout_mask: float = 120.0,
                 max_timeout_complex: float = 60.0,
                 file_check_interval: float = 0.1,
                 test_timeout: float = 5.0,
                 max_consecutive_failures: int = 5,
                 connection_retry_delay: float = 5.0,
                 health_check_interval: float = 30.0,
                 max_empty_polls: int = 50,
                 long_poll_wait: float = 20.0,
                 local_cache_limit: int = 100,
                 workdir_base: Optional[str] = None):
        # Parse server configuration
        if servers:
            # Multi-server mode: "ip1:port1,ip2:port2"
            self.servers = []
            for server in servers.split(','):
                ip, port = server.strip().split(':')
                self.servers.append({
                    'ip': ip,
                    'port': int(port),
                    'url': f"http://{ip}:{port}",
                    'available': True,
                    'last_error': None
                })
        elif server_ip:
            # Single server mode (backward compatible)
            self.servers = [{
                'ip': server_ip,
                'port': server_port,
                'url': f"http://{server_ip}:{server_port}",
                'available': True,
                'last_error': None
            }]
        else:
            raise ValueError("Must provide either servers or server_ip parameter")
        
        # Use the IPv4 loopback literal to match the bridge bind (127.0.0.1). On
        # Windows "localhost" can resolve to ::1 first, where the bridge does not
        # listen, causing connection failures / Happy-Eyeballs fallback delay.
        self.local_port = local_port
        self.local_url = f"http://127.0.0.1:{local_port}"
        self.client_id = client_id or f"mac_{int(time.time())}"
        self.poll_interval = poll_interval
        self.session = None
        self.running = False
        self.current_server_index = 0  # Polling server index
        
        # Connection recovery configuration
        self.max_consecutive_failures = max_consecutive_failures  # Maximum consecutive failures
        self.connection_retry_delay = connection_retry_delay  # Connection retry delay
        self.health_check_interval = health_check_interval  # Health check interval
        self.last_health_check = 0  # Last health check time
        
        # Polling statistics
        self.task_counts = {}  # Record task count for each server
        self.last_poll_time = {}  # Record last poll time for each server
        self.consecutive_empty_polls = 0  # Consecutive empty poll count
        self.max_empty_polls = max_empty_polls  # Consecutive empty poll threshold, print log after exceeding
        self.base_poll_interval = poll_interval  # Save base polling interval
        self.current_poll_interval = poll_interval  # Current polling interval, always keep fixed value
        
        # Unified timeout/delay configuration
        self.http_timeout_total = http_timeout_total
        self.connector_limit = connector_limit
        self.read_timeout_default = read_timeout_default
        self.base_processing_timeout = base_processing_timeout
        self.mask_increment_seconds = mask_increment_seconds
        self.complex_increment_seconds = complex_increment_seconds
        self.processing_extra_buffer = processing_extra_buffer
        self.max_timeout_mask = max_timeout_mask
        self.max_timeout_complex = max_timeout_complex
        self.file_check_interval = file_check_interval
        self.test_timeout = test_timeout
        self.long_poll_wait = long_poll_wait  # Long-poll wait seconds sent to server via ?wait=
        self.local_cache_limit = max(0, local_cache_limit)
        self.workdir_base = Path(workdir_base).expanduser() if workdir_base else None
        self.render_latencies = deque(maxlen=50)
        self.client_status = "ready"
        self.last_bridge_health: Dict[str, Any] = {}

        # Initialize server statistics
        for server in self.servers:
            self.task_counts[server['url']] = 0
            self.last_poll_time[server['url']] = 0
        
    async def start(self):
        """Start the client"""
        # Create HTTP session with more relaxed connection configuration
        connector = aiohttp.TCPConnector(
            limit=self.connector_limit,
            limit_per_host=5,  # Maximum 5 connections per host
            keepalive_timeout=30,  # Keep-alive timeout
            enable_cleanup_closed=True  # Enable cleanup of closed connections
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.http_timeout_total,
            connect=10,  # Connection timeout
            sock_read=30  # Read timeout
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        self.running = True
        
        print(f"🚀 Starting Lightroom reverse client (multi-server message center mode)")
        print(f"Server list: {', '.join([s['url'] for s in self.servers])}")
        print(f"Local Lightroom: {self.local_url}")
        print(f"Client ID: {self.client_id}")
        print("=" * 50)
        
        # Loop to attempt connection until successful
        max_retry_attempts = 0  # Use default value
        retry_count = 0
        connection_success = False
        
        while not connection_success:
            # Test connection
            print(f"Attempting to connect to servers (attempt {retry_count+1})...")
            if await self.test_connections():
                # Register client after successful connection
                if await self.register():
                    connection_success = True
                    print("✅ Server connection and registration successful")
                else:
                    print("❌ Registration failed, will retry connection...")
                    retry_count += 1
            else:
                print("❌ Connection test failed, will retry after delay...")
                retry_count += 1
            
            # Check if maximum retry attempts reached
            if max_retry_attempts > 0 and retry_count >= max_retry_attempts:
                print(f"❌ Maximum retry attempts ({max_retry_attempts}) reached, exiting")
                return
            
            # If connection failed, wait for fixed time before retry
            if not connection_success:
                # Use fixed retry interval
                print(f"⏳ Waiting {self.connection_retry_delay} seconds before retrying connection...")
                await asyncio.sleep(self.connection_retry_delay)
        
        print("✅ Startup complete, starting to poll for tasks...")
        print("Press Ctrl+C to stop the client\n")
        
        # Start polling for tasks
        await self.poll_loop()
    
    async def test_connections(self) -> bool:
        """Test all server connections"""
        available_servers = 0
        
        # Test all server connections
        for i, server in enumerate(self.servers):
            try:
                async with self.session.get(f"{server['url']}/api/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        print(f"✅ Server {server['url']} connection OK (version: {health_data.get('version', 'unknown')})")
                        server['available'] = True
                        server['last_error'] = None
                        available_servers += 1
                    else:
                        print(f"❌ Server {server['url']} abnormal response: {response.status}")
                        server['available'] = False
                        server['last_error'] = f"HTTP {response.status}"
            except Exception as e:
                print(f"❌ Server {server['url']} connection failed: {e}")
                server['available'] = False
                server['last_error'] = str(e)
        
        if available_servers == 0:
            print("❌ All servers are unreachable")
            return False
        
        print(f"✅ {available_servers}/{len(self.servers)} servers connected successfully")
        
        # Test local Lightroom - use POST request for testing
        try:
            test_payload = {
                "photo_path": "test_connection", 
                "xmp_path": "test_connection"
            }
            async with self.session.post(self.local_url, json=test_payload) as response:
                # Lightroom service responds even with invalid data, we're just testing connectivity
                if response.status in [200, 400, 500]:  # Any HTTP response indicates service is available
                    print("✅ Local Lightroom service is OK")
                else:
                    print(f"❌ Local Lightroom abnormal response: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Local Lightroom connection failed: {e}")
            print("Please ensure test_lightroom_api.py is running")
            return False
        
        return True
    
    async def register_single_server(self, server: Dict) -> bool:
        """Register to a single server"""
        if not server['available']:
            return False

        health = await self.get_local_health()
        reported_status = health.get("status", self.client_status)
            
        registration_data = {
            "client_id": self.client_id,
            "client_type": "lightroom_bridge",
            "capabilities": ["photo_processing"],
            "status": reported_status,
            "last_seen": time.time(),
            "local_port": self.local_port,
            "health": health,
            "catalog_count": health.get("catalog_count"),
            "recent_render_p95": health.get("recent_render_p95"),
            "disk_free_bytes": health.get("disk_free_bytes"),
        }
        
        try:
            async with self.session.post(
                f"{server['url']}/api/register_client",
                json=registration_data
            ) as response:
                if response.status == 200:
                    print(f"✅ Successfully registered to server {server['url']}")
                    return True
                else:
                    print(f"❌ Registration failed {server['url']}: {response.status}")
                    server['available'] = False
                    return False
        except Exception as e:
            print(f"❌ Registration exception {server['url']}: {e}")
            server['available'] = False
            return False
    
    async def register(self) -> bool:
        """Register to all available servers"""
        successful_registrations = 0
        
        for server in self.servers:
            if await self.register_single_server(server):
                successful_registrations += 1
        
        if successful_registrations == 0:
            print("❌ All server registrations failed")
            return False
        
        print(f"✅ Successfully registered to {successful_registrations} servers")
        return True

    async def get_local_health(self) -> Dict[str, Any]:
        """Collect local bridge/cache health for server-side routing decisions."""
        health: Dict[str, Any] = {
            "status": self.client_status,
            "local_cache_limit": self.local_cache_limit,
            "recent_render_p95": self.recent_latency_p95(),
        }

        try:
            usage = shutil.disk_usage(self.get_workdir_base())
            health["disk_free_bytes"] = usage.free
        except Exception as e:
            health["disk_error"] = str(e)

        try:
            task_dirs = [
                path for path in self.get_workdir_base().glob("lightroom_task_*")
                if path.is_dir()
            ]
            health["local_cache_count"] = len(task_dirs)
        except Exception as e:
            health["local_cache_error"] = str(e)

        try:
            health_timeout = aiohttp.ClientTimeout(total=min(self.test_timeout, 5.0))
            async with self.session.get(f"{self.local_url}/health", timeout=health_timeout) as response:
                bridge_health = await response.json()
                health["bridge"] = bridge_health
                plugin = bridge_health.get("plugin") if isinstance(bridge_health, dict) else None
                if isinstance(plugin, dict):
                    catalog_size = plugin.get("catalog_lru_size")
                    catalog_limit = plugin.get("catalog_lru_limit")
                    if catalog_size is not None:
                        health["catalog_count"] = int(catalog_size)
                    if catalog_limit is not None:
                        health["catalog_limit"] = int(catalog_limit)
                if isinstance(bridge_health, dict) and bridge_health.get("status") != "ready":
                    health["status"] = "blocked"
        except Exception as e:
            health["bridge"] = {
                "connected": False,
                "status": "bridge_unreachable",
                "error": str(e),
            }
            health["status"] = "blocked"

        self.last_bridge_health = health
        return health

    def recent_latency_p95(self) -> Optional[float]:
        if not self.render_latencies:
            return None
        values = sorted(self.render_latencies)
        index = max(0, min(len(values) - 1, int(round((len(values) - 1) * 0.95))))
        return values[index]

    def adaptive_processing_timeout(self, complexity_timeout: float) -> float:
        """Use recent p95 latency as a floor so slower Lightroom states get more budget."""
        p95 = self.recent_latency_p95()
        if p95 is None:
            return complexity_timeout
        adaptive_floor = p95 * 1.5
        max_budget = max(self.max_timeout_mask, self.max_timeout_complex)
        return min(max(complexity_timeout, adaptive_floor), max_budget)
    
    async def poll_loop(self):
        """Main polling loop - Fair polling of all available servers (improved version)"""
        consecutive_failures = 0
        last_task_time = time.time()
        reconnect_attempts = 0
        
        while self.running:
            try:
                current_time = time.time()
                task_found = False
                
                # Periodic health check and statistics display
                if current_time - self.last_health_check > self.health_check_interval:
                    print(f"👩‍⚕️ Periodic health check... (Time since last task: {current_time - last_task_time:.1f}s)")
                    await self.health_check_all_servers()
                    self.last_health_check = current_time
                    
                    # Display polling statistics
                    await self.show_polling_statistics()
                
                # Get all available servers
                available_servers = [s for s in self.servers if s['available']]
                
                if not available_servers:
                    # Use fixed time interval for reconnection
                    reconnect_attempts += 1
                    
                    print(f"⚠️ No available servers, retrying connection after {self.connection_retry_delay:.1f} seconds... (Reconnect attempt {reconnect_attempts})")
                    await asyncio.sleep(self.connection_retry_delay)
                    
                    # Try to reconnect all servers
                    connection_success = await self.test_connections()
                    
                    # If reconnection successful, try to re-register
                    if connection_success:
                        register_success = await self.register()
                        if register_success:
                            print("✅ Reconnection and registration successful, continuing to poll for tasks")
                            reconnect_attempts = 0  # Reset reconnect count
                    
                    continue
                
                # Implement polling schedule: start from current index, check each server in turn
                servers_to_check = []
                for i in range(len(available_servers)):
                    # Calculate actual server index (round-robin)
                    actual_index = (self.current_server_index + i) % len(available_servers)
                    servers_to_check.append(available_servers[actual_index])
                
                # Split the long-poll budget across servers so a multi-server client does
                # not block the full long_poll_wait on one empty server before checking the
                # next. With a single server this is the full long_poll_wait (optimal).
                effective_wait = self.long_poll_wait / max(1, len(servers_to_check))

                # Poll all available servers, but use polling order
                for server in servers_to_check:
                    try:
                        # Long-poll: server holds the request up to effective_wait seconds.
                        # Read timeout must exceed effective_wait. Old servers ignore ?wait
                        # and return instantly (safe; the empty-poll sleep still applies).
                        poll_timeout = aiohttp.ClientTimeout(total=effective_wait + 10.0)
                        async with self.session.get(
                            f"{server['url']}/api/get_task/{self.client_id}?wait={effective_wait}",
                            timeout=poll_timeout
                        ) as response:
                            if response.status == 200:
                                task = await response.json()
                                
                                if task and task.get('task_id'):
                                    consecutive_failures = 0
                                    last_task_time = current_time
                                    task['source_server'] = server  # Record task source server
                                    print(f"🎆 Got task from {server['url']}: {task.get('task_id')}")
                                    
                                    # Update statistics
                                    self.task_counts[server['url']] += 1
                                    self.last_poll_time[server['url']] = current_time
                                    self.consecutive_empty_polls = 0  # Reset empty poll count
                                    
                                    # Intelligently adjust polling interval: restore base interval when task found
                                    self.current_poll_interval = self.base_poll_interval
                                    
                                    # Update server index, start from next server next time
                                    server_index_in_available = available_servers.index(server)
                                    self.current_server_index = (server_index_in_available + 1) % len(available_servers)
                                    
                                    await self.process_task(task)
                                    task_found = True
                                    break  # Stop this round of polling after finding a task
                            elif response.status == 404:
                                # No task, normal situation, continue checking next server
                                pass
                            else:
                                print(f"⚠️ Server {server['url']} failed to get task: {response.status}")
                                await self.mark_server_unavailable(server, f"HTTP {response.status}")
                    
                    except asyncio.TimeoutError:
                        print(f"⚠️ Server {server['url']} polling timeout")
                        await self.mark_server_unavailable(server, "Timeout")
                        consecutive_failures += 1
                    except Exception as e:
                        print(f"⚠️ Server {server['url']} polling exception: {e}")
                        await self.mark_server_unavailable(server, str(e))
                        consecutive_failures += 1
                
                # If no task found, update polling index and statistics
                if not task_found and available_servers:
                    self.current_server_index = (self.current_server_index + 1) % len(available_servers)
                    self.consecutive_empty_polls += 1
                    
                    # Use fixed polling interval, no longer adjust based on empty poll count
                    self.current_poll_interval = self.base_poll_interval
                    
                    # If consecutive empty polls exceed threshold, print log but don't adjust interval
                    if self.consecutive_empty_polls > self.max_empty_polls and self.consecutive_empty_polls % 10 == 0:
                        print(f"🔄 {self.consecutive_empty_polls} consecutive empty polls, keeping fixed interval {self.current_poll_interval:.1f}s")
                    
                    print(f"🔄 No task this round, next poll starts from server index {self.current_server_index} ({available_servers[self.current_server_index]['url']})")
                
                # If too many consecutive failures, try to reconnect all servers
                if consecutive_failures >= self.max_consecutive_failures:
                    print(f"🔄 {consecutive_failures} consecutive failures, attempting to reconnect all servers...")
                    await asyncio.sleep(self.connection_retry_delay)
                    connection_success = await self.test_connections()

                    # If reconnection successful, try to re-register
                    if connection_success:
                        register_success = await self.register()
                        if register_success:
                            print("✅ Reconnection and registration successful, continuing to poll for tasks")

                    consecutive_failures = 0

                # Only sleep when no task was processed this round. After handling a task,
                # loop immediately to keep latency low (long-poll already paces empty rounds).
                if task_found:
                    await asyncio.sleep(0)
                    continue
                else:
                    await asyncio.sleep(self.current_poll_interval)

            except Exception as e:
                print(f"⚠️ Polling loop exception: {e}")
                consecutive_failures += 1
                await asyncio.sleep(self.connection_retry_delay)
    
    async def process_task(self, task: Dict[str, Any]) -> bool:
        """Process task - Complete message center state machine flow (supports multi-server)"""
        task_id = task.get('task_id')
        photo_path = task.get('photo_path')
        xmp_path = task.get('xmp_path')
        requires_download = task.get('requires_download', False)
        read_timeout = task.get('read_timeout', self.read_timeout_default)
        options = task.get('options') or {}
        export_masks = bool(options.get('export_masks')) if isinstance(options, dict) else False
        source_server = task.get('source_server')  # Task source server
        task_dir: Optional[Path] = None
        
        print(f"\n📸 Received task {task_id} (source: {source_server['url']})")
        print(f"  Photo: {Path(photo_path).name}")
        print(f"  XMP: {Path(xmp_path).name}")
        print(f"  Read timeout: {read_timeout}s")
        if export_masks:
            print("  Mask export: enabled")
        
        start_time = time.time()
        
        try:
            self.client_status = "busy"
            # 1. Confirm start processing to source server - state transition reading -> processing
            print(f"  🔄 Confirming start processing...")
            async with self.session.post(
                f"{source_server['url']}/api/start_processing/{task_id}",
                json={"client_id": self.client_id}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"  ❌ Confirm processing failed: {error_text}")
                    return False
            
            # 2. If files need to be downloaded, download from source server to local
            if requires_download:
                print(f"  📥 Downloading task files...")
                task_dir = await self.download_task_files(task_id, source_server)
                photo_path = str(task_dir / "before.jpg")
                xmp_path = str(task_dir / "config.lua")
            else:
                task_dir = self.get_task_dir_from_path(photo_path)
            
            # 3. Check if files exist
            if not Path(photo_path).exists():
                raise FileNotFoundError(f"Photo file does not exist: {photo_path}")
            
            if not Path(xmp_path).exists():
                raise FileNotFoundError(f"XMP file does not exist: {xmp_path}")

            mask_features = self.inspect_mask_features(xmp_path)
            
            # 4. Check lua configuration complexity, set appropriate processing timeout
            complexity_timeout = await self.calculate_processing_timeout(xmp_path)
            processing_timeout = self.adaptive_processing_timeout(complexity_timeout)
            
            # 5. Send to local Lightroom for processing.
            #    The bridge now BLOCKS until the exported file lands (holding its
            #    Lightroom lock), so we hand it our wait budget and make our own
            #    aiohttp read timeout a bit LONGER, guaranteeing the bridge gives up
            #    (and releases its lock) before the client abandons the request.
            bridge_budget = processing_timeout + self.processing_extra_buffer
            payload = {
                "photo_path": photo_path,
                "xmp_path": xmp_path,
                "task_id": task_id,
                "timeout": bridge_budget,
                **self.render_output_target(task_dir, f"{task_id}_main"),
            }

            print(f"  🔄 Sending to Lightroom for processing (estimated time: {processing_timeout}s)...")

            # Use HTTP client with dynamic timeout (slightly above the bridge budget).
            timeout = aiohttp.ClientTimeout(total=bridge_budget + 5.0)
            async with self.session.post(
                self.local_url,
                json=payload,
                timeout=timeout
            ) as response:
                elapsed = time.time() - start_time
                success = response.status == 200
                processed_image_path = None
                
                if success:
                    try:
                        result_data = await response.json()
                        processed_image_path = result_data.get('output_path')
                        logger.debug(f"Lightroom returned output_path: {processed_image_path}; full response: {result_data}")
                        print(f"  ✅ Processing successful ({elapsed:.1f}s)")
                    except Exception as json_error:
                        result_data = {"message": "Processing successful"}
                        processed_image_path = None
                        print(f"  ⚠️ JSON parsing failed: {json_error}")
                    result_data["mask_features"] = mask_features
                    
                    error = None
                else:
                    result_data = await self.parse_error_response(response)
                    result_data["mask_features"] = mask_features
                    error = result_data.get("error") or result_data.get("message") or str(result_data)
                    print(f"  ❌ Processing failed: {error}")
                
                # 5. If processing successful but file doesn't exist, wait for file save to complete
                final_image_path = processed_image_path
                if success and processed_image_path and not Path(processed_image_path).exists():
                    print("  ⏳ Waiting for file save to complete...")
                    file_ready, found_path = await self.wait_for_output_file(processed_image_path, 20)
                    if file_ready and found_path:
                        final_image_path = found_path
                        print(f"  ✅ Found output file: {final_image_path}")
                    else:
                        print(f"  ⚠️ Wait timeout, file still doesn't exist: {processed_image_path}")

                # 6. Upload the result before publishing completed status. The
                #    server also accepts the legacy report->upload order, but
                #    upload->report makes "completed" mean the file is ready.
                if success and final_image_path and Path(final_image_path).exists():
                    print("  📤 Uploading processing result to source server...")
                    upload_success = await self.upload_processed_image(task_id, final_image_path, source_server)
                    if upload_success:
                        print("  ✅ Result upload successful")
                        if result_data is None:
                            result_data = {}
                        if export_masks:
                            mask_bridge_budget = max(
                                bridge_budget,
                                self.max_timeout_mask + self.processing_extra_buffer,
                            )
                            mask_result = await self.export_and_upload_masks(
                                task_id=task_id,
                                photo_path=photo_path,
                                xmp_path=xmp_path,
                                bridge_budget=mask_bridge_budget,
                                source_server=source_server,
                            )
                            result_data.update(mask_result)
                    else:
                        print("  ⚠️ Result upload failed")
                        success = False
                        error = "Result upload failed"
                        if result_data is None:
                            result_data = {}
                        result_data["output_path"] = final_image_path
                elif success and not processed_image_path:
                    print("  ⚠️ Processing successful but output_path is None")
                    success = False
                    error = "Processing succeeded but output_path is None"
                elif success and processed_image_path and not Path(processed_image_path).exists():
                    print(f"  ⚠️ Processing successful but output file doesn't exist: {processed_image_path}")
                    success = False
                    error = f"Processing succeeded but output file does not exist: {processed_image_path}"

                # 7. Report final task state to source server.
                await self.report_result(
                    task_id=task_id,
                    success=success,
                    elapsed_time=time.time() - start_time,
                    error=error,
                    result_data=result_data,
                    source_server=source_server
                )

                return success
        
        except MaskExportError as e:
            elapsed = time.time() - start_time
            print(f"  ❌ Mask export exception: {e}")
            result_data = {
                "status": "error",
                "error_code": "mask_export_failed",
                "error": str(e),
                "retryable": e.retryable,
                "details": e.details,
            }

            await self.report_result(
                task_id=task_id,
                success=False,
                elapsed_time=elapsed,
                error=str(e),
                result_data=result_data,
                source_server=source_server
            )

            return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ❌ Processing exception: {e}")
            
            # Also report result to source server on exception to ensure state machine transitions correctly
            await self.report_result(
                task_id=task_id,
                success=False,
                elapsed_time=elapsed,
                error=str(e),
                result_data=None,
                source_server=source_server
            )
            
            return False
        finally:
            elapsed = time.time() - start_time
            if elapsed > 0:
                self.render_latencies.append(elapsed)
            self.client_status = "ready"
            if task_dir:
                await self.cleanup_local_cache(task_dir)
    
    async def report_result(self, task_id: str, success: bool, elapsed_time: float,
                          error: Optional[str] = None, result_data: Optional[Dict] = None,
                          source_server: Optional[Dict] = None):
        """Report task result to source server - Trigger state transition"""
        if not source_server:
            print("  ⚠️ Missing source server information, cannot report result")
            return
            
        try:
            result_payload = {
                "task_id": task_id,
                "client_id": self.client_id,
                "success": success,
                "elapsed_time": elapsed_time,
                "error": error,
                "result_data": result_data
            }
            
            async with self.session.post(
                f"{source_server['url']}/api/report_result",
                json=result_payload
            ) as response:
                if response.status == 200:
                    status = "✅ Success" if success else "❌ Failed"
                    print(f"  📡 Result reported to {source_server['url']} ({status})")
                else:
                    print(f"  ⚠️ Report failed: {response.status}")
        
        except Exception as e:
            print(f"  ⚠️ Report exception: {e}")
    
    async def upload_processed_image(self, task_id: str, image_path: str, source_server: Dict) -> bool:
        """Upload processed image to source server"""
        try:
            image_file = Path(image_path)
            if not image_file.exists():
                print(f"  ❌ Image file does not exist: {image_path}")
                return False
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            with image_file.open("rb") as image_stream:
                data.add_field('processed_image', image_stream,
                             filename=image_file.name,
                             content_type='image/jpeg')

                async with self.session.post(
                    f"{source_server['url']}/api/upload_result",
                    params={'task_id': task_id},
                    data=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"    💾 Saved to {source_server['url']}: {result.get('saved_path')}")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"    ❌ Upload failed: {response.status} - {error_text}")
                        return False
                    
        except Exception as e:
            print(f"    ❌ Upload exception: {e}")
            return False

    def _split_lua_top_level_table(self, table_body: str) -> List[str]:
        """Split Lua table body items while respecting nested braces/strings."""
        items: List[str] = []
        start = 0
        depth = 0
        in_string: Optional[str] = None
        escape = False
        i = 0
        while i < len(table_body):
            ch = table_body[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_string:
                    in_string = None
            else:
                if ch in ("'", '"'):
                    in_string = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth = max(0, depth - 1)
                elif ch == "," and depth == 0:
                    item = table_body[start:i].strip()
                    if item:
                        items.append(item)
                    start = i + 1
            i += 1

        tail = table_body[start:].strip()
        if tail:
            items.append(tail)
        return items

    def _find_lua_assignment_value(self, content: str, key: str) -> Optional[Tuple[int, int, str]]:
        """Return value span for a top-level-ish `key = { ... }` assignment."""
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(key)}\s*=\s*{{")
        match = pattern.search(content)
        if not match:
            return None

        open_brace = content.find("{", match.start())
        depth = 0
        in_string: Optional[str] = None
        escape = False
        for index in range(open_brace, len(content)):
            ch = content[index]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_string:
                    in_string = None
                continue

            if ch in ("'", '"'):
                in_string = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return match.start(), index + 1, content[open_brace:index + 1]

        return None

    def _remove_lua_assignment(self, content: str, key: str) -> str:
        span = self._find_lua_assignment_value(content, key)
        if not span:
            return content

        start, end, _value = span
        while end < len(content) and content[end] in " \t\r\n,":
            if content[end] == ",":
                end += 1
                break
            end += 1
        return content[:start] + content[end:]

    def _extract_mask_groups(self, lua_content: str) -> List[str]:
        span = self._find_lua_assignment_value(lua_content, "MaskGroupBasedCorrections")
        if not span:
            return []

        table_value = span[2].strip()
        if not (table_value.startswith("{") and table_value.endswith("}")):
            return []
        return self._split_lua_top_level_table(table_value[1:-1])

    def inspect_mask_features(self, xmp_path: str) -> Dict[str, Any]:
        """G3 marker: expose mask count/types and likely AI/Sensei usage."""
        try:
            lua_content = Path(xmp_path).read_text(encoding="utf-8")
        except Exception as e:
            return {
                "has_masks": False,
                "mask_count": 0,
                "mask_types": [],
                "has_ai_masks": False,
                "inspect_error": str(e),
            }

        mask_groups = self._extract_mask_groups(lua_content)
        mask_types = sorted(set(re.findall(r'What\s*=\s*["\']([^"\']*Mask/[^"\']+)["\']', lua_content)))
        ai_tokens = [
            "Mask/Subject",
            "Mask/Sky",
            "Mask/People",
            "Mask/Person",
            "Mask/Object",
            "Mask/Background",
            "Mask/Adaptive",
            "Mask/Sensei",
            "Mask/AI",
            "Denoise",
            "SelectSubject",
            "SelectSky",
        ]
        has_ai_masks = any(token.lower() in lua_content.lower() for token in ai_tokens)
        return {
            "has_masks": bool(mask_groups or "MaskGroupBasedCorrections" in lua_content),
            "mask_count": len(mask_groups),
            "mask_types": mask_types,
            "has_ai_masks": has_ai_masks,
        }

    def _write_mask_probe_lua_files(self, xmp_path: str, task_dir: Path) -> Tuple[Optional[Path], List[Dict[str, str]]]:
        """Create baseline and per-mask probe Lua configs for G2 diff export."""
        source = Path(xmp_path)
        lua_content = source.read_text(encoding="utf-8")
        mask_groups = self._extract_mask_groups(lua_content)
        if not mask_groups:
            return None, []

        mask_dir = task_dir / "mask_export"
        mask_dir.mkdir(parents=True, exist_ok=True)

        baseline_lua = self._remove_lua_assignment(lua_content, "MaskGroupBasedCorrections")
        baseline_path = mask_dir / "baseline_no_masks.lua"
        baseline_path.write_text(baseline_lua, encoding="utf-8")

        probes: List[Dict[str, str]] = []
        for index, group in enumerate(mask_groups, 1):
            probe_content = self._build_single_mask_probe_lua(lua_content, group)
            mask_id = f"mask_{index:03d}"
            probe_path = mask_dir / f"{mask_id}_probe.lua"
            probe_path.write_text(probe_content, encoding="utf-8")
            probes.append({
                "mask_id": mask_id,
                "xmp_path": str(probe_path),
            })

        return baseline_path, probes

    def _build_single_mask_probe_lua(self, original_lua: str, mask_group: str) -> str:
        """Replace all mask groups with one high-contrast probe group."""
        span = self._find_lua_assignment_value(original_lua, "MaskGroupBasedCorrections")
        if not span:
            return original_lua

        probe_group = self._force_mask_group_visible(mask_group)
        replacement = "MaskGroupBasedCorrections = {\n" + probe_group + "\n}"
        return original_lua[:span[0]] + replacement + original_lua[span[1]:]

    def _force_mask_group_visible(self, mask_group: str) -> str:
        """Make a mask group render as a strong visible probe while preserving masks."""
        remove_keys = [
            "Exposure2012",
            "Contrast2012",
            "Highlights2012",
            "Shadows2012",
            "Whites2012",
            "Blacks2012",
            "Clarity2012",
            "Texture",
            "Dehaze",
            "Saturation",
            "Sharpness",
            "LuminanceNoiseReduction",
            "ColorNoiseReduction",
            "Temperature",
            "Tint",
            "LocalExposure",
            "LocalHue",
            "LocalSaturation",
            "LocalContrast",
            "LocalClarity",
            "LocalSharpness",
            "LocalBrightness",
            "LocalToningHue",
            "LocalToningSaturation",
            "LocalExposure2012",
            "LocalContrast2012",
            "LocalHighlights2012",
            "LocalShadows2012",
            "LocalWhites2012",
            "LocalBlacks2012",
            "LocalClarity2012",
            "LocalDehaze",
            "LocalLuminanceNoise",
            "LocalMoire",
            "LocalDefringe",
            "LocalTemperature",
            "LocalTint",
            "LocalTexture",
            "LocalGrain",
            "LocalCurveRefineSaturation",
        ]
        group = mask_group
        for key in remove_keys:
            group = re.sub(
                rf"(^[ \t]*){re.escape(key)}\s*=\s*[-+]?\d+(?:\.\d+)?\s*,?\s*$",
                "",
                group,
                flags=re.MULTILINE,
            )

        insert_at = group.find("{")
        if insert_at == -1:
            return mask_group
        insert_at += 1
        probe_settings = (
            "\n    CorrectionAmount = 1,"
            "\n    CorrectionActive = true,"
            "\n    LocalExposure2012 = 5.0,"
            "\n    LocalContrast2012 = 1.0,"
            "\n    LocalHighlights2012 = 1.0,"
            "\n    LocalWhites2012 = 1.0,"
            "\n    LocalClarity2012 = 1.0,"
            "\n    LocalSaturation = -100,"
        )
        return group[:insert_at] + probe_settings + group[insert_at:]

    async def render_lightroom_variant(
        self,
        photo_path: str,
        xmp_path: str,
        task_id: str,
        bridge_budget: float,
        output_target: Dict[str, str],
    ) -> str:
        """Render a Lightroom variant through the local bridge and return output path."""
        payload = {
            "photo_path": photo_path,
            "xmp_path": xmp_path,
            "task_id": task_id,
            "timeout": bridge_budget,
            **output_target,
        }
        timeout = aiohttp.ClientTimeout(total=bridge_budget + 5.0)
        async with self.session.post(self.local_url, json=payload, timeout=timeout) as response:
            if response.status != 200:
                result_data = await self.parse_error_response(response)
                error = result_data.get("error") or result_data.get("message") or str(result_data)
                raise MaskExportError(
                    f"Mask probe render failed: {error}",
                    retryable=bool(result_data.get("retryable", response.status >= 500)),
                    details=result_data,
                )

            result_data = await response.json()
            output_path = result_data.get("output_path")
            if not output_path:
                raise MaskExportError("Mask probe render succeeded without output_path")

            final_path = output_path
            if not Path(final_path).exists():
                file_ready, found_path = await self.wait_for_output_file(final_path, 20)
                if file_ready and found_path:
                    final_path = found_path

            if not Path(final_path).exists():
                raise MaskExportError(
                    f"Mask probe output file does not exist: {output_path}",
                    retryable=True,
                    details={"output_path": output_path},
                )
            return final_path

    def create_mask_from_difference(self, baseline_path: str, probe_path: str, mask_path: Path) -> Dict[str, Any]:
        """Create a grayscale PNG mask from two Lightroom renders."""
        try:
            from PIL import Image, ImageChops, ImageOps
        except Exception as e:
            raise MaskExportError(
                f"Pillow is required for mask diff export: {e}",
                retryable=False,
                details={"dependency": "Pillow"},
            ) from e

        with Image.open(baseline_path) as base_img, Image.open(probe_path) as probe_img:
            base = base_img.convert("RGB")
            probe = probe_img.convert("RGB")
            if probe.size != base.size:
                probe = probe.resize(base.size)

            diff = ImageChops.difference(base, probe)
            gray = ImageOps.grayscale(diff)
            extrema = gray.getextrema()
            max_value = extrema[1] if extrema else 0
            if max_value > 0:
                gray = gray.point(lambda value: int(round(value * 255.0 / max_value)))

            mask_path.parent.mkdir(parents=True, exist_ok=True)
            gray.save(mask_path)
            return {
                "width": gray.size[0],
                "height": gray.size[1],
                "max_diff": max_value,
            }

    async def snapshot_render_output(self, output_path: str, snapshot_path: Path) -> str:
        """Copy a mutable Lightroom output into a stable per-probe file."""
        source = Path(output_path)
        if not source.exists():
            raise MaskExportError(
                f"Cannot snapshot missing render output: {output_path}",
                retryable=True,
                details={"output_path": output_path},
            )
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, source, snapshot_path)
        return str(snapshot_path)

    async def upload_mask_image(self, task_id: str, mask_id: str, mask_path: Path, source_server: Dict) -> bool:
        """Upload one generated mask image to source server."""
        try:
            if not mask_path.exists():
                print(f"    ❌ Mask file does not exist: {mask_path}")
                return False

            content_type = "image/png" if mask_path.suffix.lower() == ".png" else "image/x-portable-graymap"
            data = aiohttp.FormData()
            with mask_path.open("rb") as mask_stream:
                data.add_field(
                    "mask_image",
                    mask_stream,
                    filename=mask_path.name,
                    content_type=content_type,
                )
                async with self.session.post(
                    f"{source_server['url']}/api/upload_mask_result",
                    params={"task_id": task_id, "mask_id": mask_id},
                    data=data,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"    💾 Mask saved to {source_server['url']}: {result.get('saved_path')}")
                        return True
                    error_text = await response.text()
                    print(f"    ❌ Mask upload failed: {response.status} - {error_text}")
                    return False
        except Exception as e:
            print(f"    ❌ Mask upload exception: {e}")
            return False

    async def export_and_upload_masks(
        self,
        task_id: str,
        photo_path: str,
        xmp_path: str,
        bridge_budget: float,
        source_server: Dict,
    ) -> Dict[str, Any]:
        """Export requested mask rasters via Lightroom probe renders and upload them."""
        task_dir = self.get_task_dir_from_path(photo_path) or Path(photo_path).parent
        baseline_lua, probes = self._write_mask_probe_lua_files(xmp_path, task_dir)
        if not probes or baseline_lua is None:
            print("  🎭 export_masks requested but no MaskGroupBasedCorrections found")
            return {
                "mask_export_status": "skipped_no_masks",
                "mask_export_artifacts": [],
            }

        print(f"  🎭 Exporting {len(probes)} mask(s) via Lightroom diff probes")
        baseline_render = await self.render_lightroom_variant(
            photo_path=photo_path,
            xmp_path=str(baseline_lua),
            task_id=f"{task_id}_mask_baseline",
            bridge_budget=bridge_budget,
            output_target=self.render_output_target(task_dir, f"{task_id}_mask_baseline"),
        )
        stable_dir = task_dir / "mask_export" / "renders"
        global_baseline = await self.snapshot_render_output(
            baseline_render,
            stable_dir / "baseline_no_masks.jpg",
        )

        masks: List[Dict[str, Any]] = []
        generated_dir = task_dir / "mask_export" / "generated"
        for probe in probes:
            mask_id = probe["mask_id"]
            print(f"    🔄 Rendering probe for {mask_id}")
            probe_render = await self.render_lightroom_variant(
                photo_path=photo_path,
                xmp_path=probe["xmp_path"],
                task_id=f"{task_id}_{mask_id}",
                bridge_budget=bridge_budget,
                output_target=self.render_output_target(task_dir, f"{task_id}_{mask_id}"),
            )
            probe_output = await self.snapshot_render_output(
                probe_render,
                stable_dir / f"{mask_id}_probe.jpg",
            )
            mask_path = generated_dir / f"{mask_id}.png"
            stats = self.create_mask_from_difference(global_baseline, probe_output, mask_path)
            upload_ok = await self.upload_mask_image(task_id, mask_id, mask_path, source_server)
            if not upload_ok:
                raise MaskExportError(f"Mask upload failed for {mask_id}", retryable=True)

            masks.append({
                "mask_id": mask_id,
                "filename": mask_path.name,
                "local_path": str(mask_path),
                "method": "lightroom_diff_probe",
                **stats,
            })

        return {
            "mask_export_status": "completed",
            "mask_export_method": "lightroom_diff_probe",
            "mask_count": len(masks),
            "mask_export_artifacts": masks,
        }

    async def parse_error_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Parse bridge errors without losing structured failure details."""
        try:
            data = await response.json()
            if isinstance(data, dict):
                data.setdefault("http_status", response.status)
                return data
        except Exception:
            pass

        text = await response.text()
        return {
            "status": "error",
            "error_code": f"http_{response.status}",
            "error": text,
            "retryable": response.status >= 500,
            "http_status": response.status,
        }

    def get_workdir_base(self) -> Path:
        if self.workdir_base:
            return self.workdir_base
        workdir_base = os.getenv("LIGHTROOM_CLIENT_WORKDIR")
        if workdir_base:
            return Path(workdir_base).expanduser()
        return Path.home() / "Documents" / "Local_workspace" / "projects"

    def get_task_dir_from_path(self, photo_path: Optional[str]) -> Optional[Path]:
        if not photo_path:
            return None
        path = Path(photo_path)
        for candidate in [path.parent, *path.parents]:
            if candidate.name.startswith("lightroom_task_"):
                return candidate
        return None

    def render_output_dir(self, task_dir: Optional[Path], variant_id: str) -> Path:
        base_dir = task_dir or self.get_workdir_base()
        safe_variant = re.sub(r"[^A-Za-z0-9_.-]+", "_", variant_id).strip("._")
        if not safe_variant:
            safe_variant = f"render_{int(time.time())}"
        output_dir = base_dir / "render_outputs" / f"{safe_variant}_{time.time_ns()}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def render_output_target(self, task_dir: Optional[Path], variant_id: str) -> Dict[str, str]:
        safe_variant = re.sub(r"[^A-Za-z0-9_.-]+", "_", variant_id).strip("._-")
        if not safe_variant:
            safe_variant = f"render_{int(time.time())}"
        suffix = time.time_ns()
        output_dir = self.render_output_dir(task_dir, f"{safe_variant}_{suffix}")
        return {
            "output_dir": str(output_dir),
            "output_basename": f"{safe_variant}_{suffix}",
        }

    async def cleanup_local_cache(self, current_task_dir: Path):
        """Keep only the most recent task directories on the client."""
        await asyncio.to_thread(self._cleanup_local_cache_sync, current_task_dir)

    def _cleanup_local_cache_sync(self, current_task_dir: Path):
        try:
            current_task_dir.mkdir(parents=True, exist_ok=True)
            now = time.time()
            os.utime(current_task_dir, (now, now))

            base_dir = self.get_workdir_base()
            if not base_dir.exists():
                return

            task_dirs = [
                path for path in base_dir.glob("lightroom_task_*")
                if path.is_dir()
            ]
            task_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

            for old_dir in task_dirs[self.local_cache_limit:]:
                try:
                    shutil.rmtree(old_dir)
                    print(f"  🧹 Removed old local cache: {old_dir}")
                except Exception as e:
                    print(f"  ⚠️ Failed to remove old local cache {old_dir}: {e}")
        except Exception as e:
            print(f"  ⚠️ Local cache cleanup failed: {e}")
    
    async def download_task_files(self, task_id: str, source_server: Dict):
        """Efficiently download task files from source server in parallel"""
        try:
            # Create unique task directory, use timestamp and random number to ensure uniqueness
            import uuid
            unique_id = f"{task_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            # Cross-platform path (Windows-safe). Optionally honor LIGHTROOM_CLIENT_WORKDIR base.
            base_dir = self.get_workdir_base()
            task_dir = base_dir / f"lightroom_task_{unique_id}"
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Create processed directory for storing processed images
            processed_dir = task_dir / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Download photo and xmp files from source server in parallel
            download_tasks = [
                self.download_single_file(task_id, "photo", task_dir / "before.jpg", source_server),
                self.download_single_file(task_id, "xmp", task_dir / "config.lua", source_server)
            ]
            
            await asyncio.gather(*download_tasks)
            print(f"    ✅ Files downloaded from {source_server['url']}: {task_dir}")
            
            # Return task directory path
            return task_dir
            
        except Exception as e:
            raise Exception(f"File download failed: {e}")
    
    async def download_single_file(self, task_id: str, file_type: str, local_path: Path, source_server: Dict):
        """Download a single file from source server"""
        async with self.session.get(
            f"{source_server['url']}/api/download_file/{task_id}/{file_type}"
        ) as response:
            if response.status != 200:
                raise Exception(f"{file_type} file download failed: {response.status}")
            
            # Stream write to file
            with open(local_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(1024 * 1024):
                    await asyncio.to_thread(f.write, chunk)
    
    async def calculate_processing_timeout(self, xmp_path: str) -> float:
        """Calculate processing timeout based on lua configuration complexity"""
        try:
            # Read lua configuration file content
            with open(xmp_path, 'r', encoding='utf-8') as f:
                lua_content = f.read()
            
            base_timeout = self.base_processing_timeout  # Base timeout
            
            # Check if contains complex mask processing
            if 'MaskGroupBasedCorrections' in lua_content:
                print("  🎭 Detected mask processing, extending wait time")
                
                # Count actual mask count in CorrectionMasks
                actual_mask_count = len(re.findall(r'What\s*=\s*"Mask/', lua_content))
                actual_mask_count = max(1, actual_mask_count)
                
                # Add processing time for each actual mask (configurable)
                mask_timeout = base_timeout + (actual_mask_count * self.mask_increment_seconds)
                
                print(f"  📊 Actual mask count: {actual_mask_count}, timeout set: {mask_timeout}s")
                return min(max(mask_timeout, self.max_timeout_mask), self.max_timeout_mask)
            
            # Check other potentially time-consuming operations
            complex_operations = [
                'LocalizedCorrections',  # Local adjustments
                'CircularGradientBasedCorrections',  # Radial filter
                'GradientBasedCorrections',  # Gradient filter
                'RetouchAreas',  # Spot removal
            ]
            
            complex_count = sum(1 for op in complex_operations if op in lua_content)
            if complex_count > 0:
                complex_timeout = base_timeout + (complex_count * self.complex_increment_seconds)  # Add configurable seconds for each complex operation
                print(f"  ⚙️ Complex operation count: {complex_count}, timeout set: {complex_timeout}s")
                return min(complex_timeout, self.max_timeout_complex)
            
            # Simple adjustments, use base timeout
            print(f"  🚀 Simple adjustments, timeout set: {base_timeout}s")
            return base_timeout
            
        except Exception as e:
            print(f"  ⚠️ Configuration parsing failed, using default timeout: {e}")
            return 30.0  # Default 30 seconds
    
    async def wait_for_output_file(self, file_path: str, max_wait_seconds: float) -> tuple[bool, str]:
        """Wait for output file generation, adapted to Lightroom asynchronous processing
        
        Returns:
            tuple[bool, str]: (Whether file is ready, file path)
        """
        start_time = time.time()
        check_interval = self.file_check_interval  # Configurable check interval
        
        while time.time() - start_time < max_wait_seconds:
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                if file_size > 0:  # Ensure file is not empty
                    print(f"    ✅ File ready: {file_size:,} bytes")
                    return True, file_path
            
            await asyncio.sleep(check_interval)
        
        return False, file_path
    
    async def mark_server_unavailable(self, server: Dict, error: str):
        """Mark server as unavailable"""
        server['available'] = False
        server['last_error'] = error
        server['last_failure_time'] = time.time()
        print(f"🔴 Server {server['url']} marked as unavailable: {error}")
    
    async def health_check_all_servers(self):
        """Perform health check on all servers"""
        print("👩‍⚕️ Performing health check on all servers...")
        
        for server in self.servers:
            try:
                # Use short timeout for health check
                health_timeout = aiohttp.ClientTimeout(total=5.0)
                async with self.session.get(
                    f"{server['url']}/api/health", 
                    timeout=health_timeout
                ) as response:
                    if response.status == 200:
                        if not server['available']:
                            print(f"🟢 Server {server['url']} has recovered")
                        server['available'] = True
                        server['last_error'] = None
                        # Refresh server-side client health/capacity on every
                        # health pass, not just after recovery.
                        await self.register_single_server(server)
                    else:
                        await self.mark_server_unavailable(server, f"Health check failed: {response.status}")
            except Exception as e:
                await self.mark_server_unavailable(server, f"Health check error: {e}")
        
        available_count = sum(1 for s in self.servers if s['available'])
        print(f"📊 Health check complete: {available_count}/{len(self.servers)} servers available")
    
    async def show_polling_statistics(self):
        """Display polling statistics"""
        print("\n📊 === Polling Statistics ===")
        print(f"🔄 Current polling interval: {self.current_poll_interval:.1f}s")
        print(f"📈 Consecutive empty polls: {self.consecutive_empty_polls}")
        print(f"🎯 Task acquisition statistics:")
        
        total_tasks = sum(self.task_counts.values())
        for server in self.servers:
            task_count = self.task_counts.get(server['url'], 0)
            last_poll = self.last_poll_time.get(server['url'], 0)
            time_since_last = time.time() - last_poll if last_poll > 0 else float('inf')
            
            status_icon = "✅" if server['available'] else "❌"
            percentage = (task_count / total_tasks * 100) if total_tasks > 0 else 0
            
            print(f"  {status_icon} {server['url']}: {task_count} tasks ({percentage:.1f}%), time since last poll: {time_since_last:.0f}s")
        
        if total_tasks > 0:
            print(f"📊 Total tasks: {total_tasks}")
        print("=" * 30)
    
    async def get_server_status_summary(self) -> str:
        """Get server status summary"""
        available = [s for s in self.servers if s['available']]
        unavailable = [s for s in self.servers if not s['available']]
        
        status = f"📊 Server status: {len(available)}/{len(self.servers)} available"
        
        if unavailable:
            status += "\n❌ Unavailable servers:"
            for server in unavailable:
                last_error = server.get('last_error', 'Unknown error')
                status += f"\n  - {server['url']}: {last_error}"
        
        return status

    async def stop(self):
        """Stop the client"""
        print("\n🛑 Stopping client...")
        self.running = False
        
        if self.session:
            try:
                # Give some time to complete ongoing requests
                await asyncio.sleep(1.0)
                await self.session.close()
                print("✅ HTTP session closed")
            except Exception as e:
                print(f"⚠️ Error closing HTTP session: {e}")
        
        # Display final status
        status_summary = await self.get_server_status_summary()
        print(status_summary)
        print("\n👋 Client stopped")


# Synchronous test functions
def test_server_connection(server_ip: str, server_port: int = 8080, timeout: float = 5.0) -> bool:
    """Test server connection"""
    try:
        response = requests.get(f"http://{server_ip}:{server_port}/api/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def test_local_lightroom(port: int = 7777, timeout: float = 5.0) -> bool:
    """Test local Lightroom"""
    try:
        response = requests.get(f"http://127.0.0.1:{port}", timeout=timeout)
        return response.status_code == 200
    except:
        return False


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Lightroom reverse client (runs on Mac) - Multi-server message center mode')
    parser.add_argument('--servers', help='Multiple server addresses (format: ip1:port1,ip2:port2)')
    parser.add_argument('--server-ip', default='28.48.6.32', help='Linux server IP address (single server mode)')
    parser.add_argument('--server-port', type=int, default=8081, help='Server port (single server mode)')
    parser.add_argument('--local-port', type=int, default=7777, help='Local Lightroom port')
    parser.add_argument('--client-id', help='Client ID (optional)')
    parser.add_argument('--poll-interval', type=float, default=2.0, help='Polling interval (seconds)')
    # Unified delay/timeout parameters
    parser.add_argument('--http-timeout-total', type=float, default=300.0, help='HTTP session total timeout (seconds)')
    parser.add_argument('--connector-limit', type=int, default=10, help='HTTP connection pool concurrency limit')
    parser.add_argument('--read-timeout-default', type=float, default=10.0, help='Read phase default timeout (seconds)')
    parser.add_argument('--base-processing-timeout', type=float, default=10.0, help='Base processing timeout (seconds); also the floor of the blocking bridge wait budget')
    parser.add_argument('--mask-increment-seconds', type=float, default=5.0, help='Timeout increment per mask (seconds)')
    parser.add_argument('--complex-increment-seconds', type=float, default=3.0, help='Timeout increment per complex operation (seconds)')
    parser.add_argument('--processing-extra-buffer', type=float, default=10.0, help='Processing request extra buffer (seconds)')
    parser.add_argument('--max-timeout-mask', type=float, default=120.0, help='Maximum timeout for mask scenarios (seconds)')
    parser.add_argument('--max-timeout-complex', type=float, default=60.0, help='Maximum timeout for complex scenarios (seconds)')
    parser.add_argument('--file-check-interval', type=float, default=0.1, help='Output file check interval (seconds)')
    parser.add_argument('--test-timeout', type=float, default=5.0, help='Test connection timeout (seconds)')
    parser.add_argument('--test', action='store_true', help='Test connection only')
    # Connection and retry related parameters
    parser.add_argument('--max-consecutive-failures', type=int, default=5, help='Maximum consecutive failures')
    parser.add_argument('--connection-retry-delay', type=float, default=5.0, help='Connection retry delay (seconds)')
    parser.add_argument('--health-check-interval', type=float, default=30.0, help='Health check interval (seconds)')
    parser.add_argument('--max-empty-polls', type=int, default=50, help='Consecutive empty poll threshold, print log after exceeding')
    parser.add_argument('--long-poll-wait', type=float, default=20.0, help='Long-poll wait seconds sent to server via ?wait= (0 = short poll)')
    parser.add_argument('--local-cache-limit', type=int, default=100, help='Number of local lightroom_task_* directories to keep')
    parser.add_argument('--workdir-base', type=str, default=None, help='Directory used for downloaded lightroom_task_* working folders')
    parser.add_argument('--max-retry-attempts', type=int, default=0, help='Maximum retry attempts, 0 means unlimited retries')
    
    args = parser.parse_args()
    
    if args.test:
        print("🔍 Connection test mode")
        print("=" * 30)
        
        print("Testing server connection...")
        if test_server_connection(args.server_ip, args.server_port, timeout=args.test_timeout):
            print("✅ Server connection OK")
        else:
            print("❌ Server connection failed")
        
        print("Testing local Lightroom...")
        if test_local_lightroom(args.local_port, timeout=args.test_timeout):
            print("✅ Local Lightroom OK")
        else:
            print("❌ Local Lightroom connection failed")
        
        return
    
    client = LightroomReverseClient(
        servers=args.servers,
        server_ip=args.server_ip if not args.servers else None,
        server_port=args.server_port,
        local_port=args.local_port,
        client_id=args.client_id,
        poll_interval=args.poll_interval,
        http_timeout_total=args.http_timeout_total,
        connector_limit=args.connector_limit,
        read_timeout_default=args.read_timeout_default,
        base_processing_timeout=args.base_processing_timeout,
        mask_increment_seconds=args.mask_increment_seconds,
        complex_increment_seconds=args.complex_increment_seconds,
        processing_extra_buffer=args.processing_extra_buffer,
        max_timeout_mask=args.max_timeout_mask,
        max_timeout_complex=args.max_timeout_complex,
        file_check_interval=args.file_check_interval,
        test_timeout=args.test_timeout,
        max_consecutive_failures=args.max_consecutive_failures,
        connection_retry_delay=args.connection_retry_delay,
        health_check_interval=args.health_check_interval,
        max_empty_polls=args.max_empty_polls,
        long_poll_wait=args.long_poll_wait,
        local_cache_limit=args.local_cache_limit,
        workdir_base=args.workdir_base
    )
    
    try:
        await client.start()
    except KeyboardInterrupt:
        print("\n⏹️ Received stop signal")
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
