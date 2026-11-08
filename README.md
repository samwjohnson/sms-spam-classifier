# SMS Spam Classifier: DistilBERT, MLflow, LLM-as-Judge, FastAPI

An end-to-end spam detection project on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,574 English SMS messages labeled ham or spam).

## 1. DistilBERT fine-tune (original Colab run)
- `distilbert-base-uncased` fine-tuned in Google Colab on a T4 GPU.
- Stratified 80/20 split (`random_state=42`): 4,459 train, 1,115 test (966 ham, 149 spam).
- Test accuracy **0.993**, spam precision 0.980, recall 0.966, F1 **0.973**.
- Baseline TF-IDF + logistic regression: accuracy 0.978, spam F1 0.912.

## 2. MLflow tracking (`sms_mlflow/`)
`train_mlflow.py` re-runs a small CPU demo fine-tune and logs it to MLflow (local file store in `sms_mlflow/mlruns`).

First run `2b09015f87d94c65870560c98e8346a3` (a quick subset demo, not a reproduction of the Colab result):
- Same stratified 80/20 split, then 1,000 train / 300 test subsets (seed 42); max length 64, 1 epoch, batch 16, lr 5e-5, ~51 s on CPU.
- Accuracy 0.980, spam precision 0.972, recall 0.875, F1 0.921.
- Logged: parameters, metrics, tokenizer, package versions, and the model (`mlflow.transformers` flavor).

> The model weights file (`model.safetensors`, ~268 MB) is **not committed** because it exceeds GitHub's 100 MB file limit. The rest of the run (params, metrics, MLmodel, config, tokenizer) is included. Rerun `train_mlflow.py` to regenerate the weights.

```sh
cd sms_mlflow
python -m venv .venv && . .venv/bin/activate
pip install mlflow transformers torch scikit-learn pandas fastapi uvicorn
python train_mlflow.py
mlflow ui --backend-store-uri ./mlruns
```

## 3. LLM-as-judge evaluation (`sms_judge_300/`)
300 messages from the same dataset (40 spam / 260 ham, stratified, seed 42, excluding the 30-message pilot), judged by Claude Opus 4.6 and Claude Haiku 5.5 with a fixed judge prompt and an 80-token reply cap.

| Judge | Correct | Precision | Recall | Spam F1 | Notes |
|---|---|---|---|---|---|
| Claude Opus 4.6 (`claude-opus-4-6`, temperature 0) | 297/300 | 0.974 | 0.950 | 0.962 | 38 of 40 spam caught, 1 false alarm; 78,090 in / 9,438 out tokens, ~$0.63 at list price |
| Claude Haiku 5.5 (`claude-haiku-5-5`, temperature omitted, effort low) | 299/300 | 1.000 | 0.975 | 0.987 | 39 of 40 spam caught, 0 false alarms; 103,781 in / 14,351 out tokens, ~$0.018 at list price ($0.10/$0.50 per MTok) |

Haiku 5.5 settings: it rejects temperature 0, so temperature is omitted and effort is set to low; same 300 messages, system prompt, label regex and 80-token cap as Opus 4.6. No replies were truncated (all 300 `end_turn`, longest 72 tokens, no thinking tokens). Its one miss was row 5542. Files: `haiku300.py`, `haiku55_300.jsonl`, `results_haiku55_300.json`, `haiku55_run.log`, `haiku55_probe.jsonl` (5-message truncation probe).

The API key is read from the `ANTHROPIC_API_KEY` environment variable and is never stored in this repo.

## 4. FastAPI serving (`sms_mlflow/app.py`)
Loads the model from the MLflow run (`MODEL_RUN_ID` env var, default the run above) and serves:
- `GET /health`: status and run ID
- `POST /predict`: body `{"message": "..."}`, returns `label` (spam or ham) and `confidence`

```sh
cd sms_mlflow && ./run_server.sh   # http://127.0.0.1:8000
curl -s http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d '{"message":"WINNER!! You have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."}'
# {"label":"spam","confidence":0.9907453060150146,"run_id":"2b09015f87d94c65870560c98e8346a3"}
```
Local only so far; not deployed.

## Dataset credit
Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84
