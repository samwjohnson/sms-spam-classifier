# SMS spam — MLflow subset demo

Agreed design: CPU DistilBERT, UCI SMS Spam Collection, stratified 80/20 split
(random seed 42), then stratified 1,000-training / 300-test subsets, max length
64, one epoch, batch 16, AdamW learning rate 5e-5. This is a quick MLflow
integration demo, **not a reproduction** of the full Colab spam F1 0.973.
No deployment or FastAPI code is included.

Run from this folder:

```sh
source .venv/bin/activate
python train_mlflow.py 2>&1 | tee run1.log
mlflow ui --backend-store-uri ./mlruns
```

Data: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
The script requires 5,574 physical tab-separated message records.
Exact package versions: versions.json. Persisted verification: run_summary.json.
HF model downloads and pip caches stay under this folder's .cache.
The model is logged using MLflow's Transformers flavor (PyTorch fallback if
needed), plus a separate tokenizer artifact; training updates the entire model.

## Local FastAPI inference (no deployment)

Reads the existing model without retraining or updating its MLflow run.
Default run: `2b09015f87d94c65870560c98e8346a3`. Override using
`MODEL_RUN_ID=<run-id> ./run_server.sh`. Startup validates 0=ham, 1=spam.
Inference uses the training maximum length of 64 tokens (long messages truncate).

```sh
cd /workspace/sms_mlflow
./run_server.sh
# Logs: server.log; PID: server.pid; bound only to local loopback.
# Stop the server (the PID file can remain; startup checks liveness):
kill "$(cat server.pid)"
```

```sh
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"message":"WINNER!! You have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."}'
curl -sS http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"message":"Hey, are we still meeting for lunch at noon tomorrow?"}'
```

Exact smoke-test commands/responses: `api_test_results.log`; repeat using
`bash test_api.sh`. Both sample labels were correct. Empty input returned HTTP
422; whitespace-only input is also rejected. Dependency versions are recorded
separately in `api_versions.json`; original training versions remain unchanged.
