"""Local inference only; reads an existing MLflow model without retraining."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parent/'.cache'/'huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import mlflow
import mlflow.transformers
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

RUN_ID=os.environ.get('MODEL_RUN_ID','2b09015f87d94c65870560c98e8346a3')

@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(4)
    mlflow.set_tracking_uri('file:/workspace/sms_mlflow/mlruns')
    pipeline=mlflow.transformers.load_model(f'runs:/{RUN_ID}/model')
    labels={int(k):str(v).lower() for k,v in pipeline.model.config.id2label.items()}
    # Training script used 0=ham and 1=spam; reject other unknown mappings.
    for idx,label in list(labels.items()):
        if label == f'label_{idx}': labels[idx]={0:'ham',1:'spam'}[idx]
    if labels != {0:'ham',1:'spam'}:
        raise RuntimeError(f'Unexpected model label mapping: {labels}')
    pipeline.model.config.id2label=labels
    pipeline.model.config.label2id={v:k for k,v in labels.items()}
    app.state.pipeline=pipeline
    yield
    app.state.pipeline=None

app=FastAPI(title='SMS spam classifier — local demo',lifespan=lifespan)

class PredictionRequest(BaseModel):
    message: str = Field(min_length=1)

    @field_validator('message')
    @classmethod
    def nonblank(cls,value):
        if not value.strip(): raise ValueError('message must not be empty or whitespace')
        return value

@app.get('/health')
def health():
    return {'status':'ok','run_id':RUN_ID,'model_loaded':app.state.pipeline is not None}

@app.post('/predict')
def predict(request: PredictionRequest):
    result=app.state.pipeline(request.message,truncation=True,max_length=64)[0]
    return {'label':result['label'],'confidence':float(result['score']),'run_id':RUN_ID}
