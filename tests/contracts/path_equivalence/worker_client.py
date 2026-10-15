"""Line-oriented worker requests with a fresh queue after worker failure."""
import json
import os
import queue
import signal
import subprocess
import threading

import pytest


class WorkerClient:
    def __init__(self, command, *, cwd, env, errors, timeout=120):
        self.command = command
        self.cwd = cwd
        self.env = env
        self.errors = errors
        self.timeout = timeout
        self.cache = {}
        self.process = None

    def _start(self):
        process = subprocess.Popen(
            self.command, cwd=self.cwd, env=self.env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.errors, text=True, bufsize=1,
            start_new_session=True)
        pending = queue.Queue()

        def read():
            for line in process.stdout:
                pending.put(line)
            pending.put(None)

        self.process = process
        self.pending = pending
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def __call__(self, cell):
        if cell.id not in self.cache:
            if self.process is None:
                self._start()
            try:
                self.process.stdin.write(json.dumps({'cell': cell.id}) + '\n')
                self.process.stdin.flush()
            except OSError:
                self.close()
                pytest.fail(f'S0 worker exited: {self.errors.name}')
            try:
                response = self.pending.get(timeout=self.timeout)
            except queue.Empty:
                self.close()
                pytest.fail(f'S0 worker timed out: {cell.id}')
            if response is None:
                self.close()
                pytest.fail(f'S0 worker exited: {self.errors.name}')
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                self.close()
                raise
            assert 'worker_error' not in result, result.get('worker_error')
            if result['cell'] != cell.id:
                self.close()
                pytest.fail(f'S0 worker cell mismatch: expected {cell.id}, got {result["cell"]}')
            self.cache[cell.id] = result
        return self.cache[cell.id]

    def close(self):
        process = self.process
        if process is None:
            return
        try:
            process.poll()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait(timeout=5)
            try:
                process.stdin.close()
            except OSError:
                pass
            self.reader.join(timeout=5)
            process.stdout.close()
        finally:
            self.process = None
