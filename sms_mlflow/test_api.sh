#!/usr/bin/env bash
set -euo pipefail
set -x
curl -sS http://127.0.0.1:8000/health
printf '\n'
curl -sS http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"message":"WINNER!! You have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."}'
printf '\n'
curl -sS http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"message":"Hey, are we still meeting for lunch at noon tomorrow?"}'
printf '\n'
curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"message":""}'
