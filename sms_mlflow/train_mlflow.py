"""Small CPU-only SMS DistilBERT MLflow demonstration, not the full benchmark."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'false'
import json, random, time, importlib.metadata
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.pytorch

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(4)
rows=[]
with (ROOT/'data'/'SMSSpamCollection').open(encoding='utf-8') as f:
    for line in f:
        label,text=line.rstrip('\r\n').split('\t',1)
        assert label in ('ham','spam')
        rows.append((text,int(label=='spam')))
print('Dataset rows:',len(rows),flush=True)
# Count physical message records, without CSV quoting or blank-line inference.
if len(rows) != 5574:
    raise ValueError(f'Expected 5574 message records; got {len(rows)}')
train,test=train_test_split(rows,test_size=.2,random_state=SEED,stratify=[r[1] for r in rows])
train,_=train_test_split(train,train_size=1000,random_state=SEED,stratify=[r[1] for r in train])
test,_=train_test_split(test,train_size=300,random_state=SEED,stratify=[r[1] for r in test])
name='distilbert-base-uncased'
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForSequenceClassification.from_pretrained(name,num_labels=2,id2label={0:'ham',1:'spam'},label2id={'ham':0,'spam':1})
class SMSDataset(Dataset):
    def __init__(self, rows):
        self.enc=tok([r[0] for r in rows],padding='max_length',truncation=True,max_length=64,return_tensors='pt')
        self.y=torch.tensor([r[1] for r in rows])
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return {**{k:v[i] for k,v in self.enc.items()},'labels':self.y[i]}
tr=DataLoader(SMSDataset(train),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(SEED))
te=DataLoader(SMSDataset(test),batch_size=16)
versions={m:importlib.metadata.version(m) for m in ['torch','transformers','mlflow','scikit-learn','pandas','numpy']}
(ROOT/'versions.json').write_text(json.dumps(versions,indent=2)+'\n')
mlflow.set_tracking_uri((ROOT/'mlruns').as_uri())
mlflow.set_experiment('sms-spam-distilbert')
with mlflow.start_run() as run:
    params=dict(model_name=name,train_subset=len(train),test_subset=len(test),dataset_rows=len(rows),max_len=64,epochs=1,batch_size=16,lr=5e-5,seed=SEED,split='stratified 80/20; seed 42; stratified subsets seed 42',device='cpu',torch_threads=4,**{k+'_version':v for k,v in versions.items()})
    mlflow.log_params(params)
    optim=torch.optim.AdamW(model.parameters(),lr=5e-5)
    start=time.perf_counter(); total=0.; count=0
    model.train()
    for step,batch in enumerate(tr):
        optim.zero_grad(set_to_none=True)
        loss=model(**batch).loss
        loss.backward(); optim.step()
        total+=loss.item()*len(batch['labels']); count+=len(batch['labels'])
        if step%10==0: print(f'step {step}/{len(tr)} loss {loss.item():.4f}',flush=True)
    train_seconds=time.perf_counter()-start
    model.eval(); pred=[]; actual=[]
    with torch.no_grad():
        for batch in te:
            pred.extend(model(**batch).logits.argmax(-1).tolist()); actual.extend(batch['labels'].tolist())
    p,r,f,_=precision_recall_fscore_support(actual,pred,average='binary',zero_division=0)
    metrics=dict(test_accuracy=accuracy_score(actual,pred),spam_precision=p,spam_recall=r,spam_f1=f,train_loss=total/count,train_time_seconds=train_seconds)
    mlflow.log_metrics(metrics)
    flavor='mlflow.transformers'
    try:
        import mlflow.transformers
        mlflow.transformers.log_model({'model':model,'tokenizer':tok},artifact_path='model',task='text-classification',pip_requirements=[f'torch=={versions["torch"]}',f'transformers=={versions["transformers"]}'])
    except Exception as exc:
        print('Transformers artifact logging failed:',repr(exc),flush=True)
        mlflow.set_tag('transformers_log_model_error',str(exc)[:2000])
        flavor='mlflow.pytorch'
        mlflow.pytorch.log_model(model,artifact_path='model',pip_requirements=[f'torch=={versions["torch"]}',f'transformers=={versions["transformers"]}'])
    tok.save_pretrained(ROOT/'tokenizer')
    mlflow.log_artifacts(str(ROOT/'tokenizer'),artifact_path='tokenizer')
    mlflow.log_artifact(str(ROOT/'versions.json'))
    summary=dict(run_id=run.info.run_id,experiment_id=run.info.experiment_id,metrics=metrics,params=params,tracking_uri=mlflow.get_tracking_uri(),mlruns=str(ROOT/'mlruns'),artifact_uri=mlflow.get_artifact_uri(),model_flavor=flavor,model_artifact='model')
# Verify persisted state after the context marks the run FINISHED.
verified=mlflow.get_run(summary['run_id'])
assert verified.info.status=='FINISHED'
assert all(k in verified.data.metrics for k in metrics)
artifacts=mlflow.MlflowClient().list_artifacts(summary['run_id'])
assert 'model' in [a.path for a in artifacts]
summary['verified_status']=verified.info.status
summary['artifacts']=[a.path for a in artifacts]
(ROOT/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
print('Run ID:',summary['run_id']); print('mlruns:',ROOT/'mlruns')
