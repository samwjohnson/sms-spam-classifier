#!/usr/bin/env bash
set -euo pipefail
cd /workspace/sms_mlflow
if [[ -f server.pid ]] && kill -0 "$(<server.pid)" 2>/dev/null; then
  echo "Server already running with PID $(<server.pid)"
  exit 0
fi
nohup .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 > server.log 2>&1 < /dev/null &
echo $! > server.pid
echo "Started PID $(<server.pid): http://127.0.0.1:8000 (see server.log)"
