# Claude LLM-as-judge: 300-message sample (job300.json), fixed system prompt,
# temperature 0, 80-token cap, LABEL/REASON format, regex extraction, resume support.
# Output file set by CLAUDE_OUT (default claude300.jsonl); temperature by CLAUDE_TEMP (default 0).
# Needs ANTHROPIC_API_KEY and CLAUDE_MODEL (the exact model id to use) in the environment.
import json,os,re,time,urllib.request,urllib.error
J=json.load(open('job300.json'));key=os.environ['ANTHROPIC_API_KEY'];MODEL=os.environ.get('CLAUDE_MODEL','claude-haiku-5-5');res=[]
done=set()
OUT=os.environ.get('CLAUDE_OUT','claude300.jsonl')
if os.path.exists(OUT): done={json.loads(l)['row'] for l in open(OUT)}
F=open(OUT,'a')
T0=time.perf_counter()
for m in J['msgs']:
    if m['row'] in done: continue
    b=json.dumps({'model':MODEL,'max_tokens':80,**({} if os.environ.get('CLAUDE_TEMP','0')=='none' else {'temperature':float(os.environ.get('CLAUDE_TEMP','0'))}),'system':J['sys'],'messages':[{'role':'user','content':f'Message: """{m["text"]}"""'}]}).encode()
    t=time.perf_counter()
    rq=urllib.request.Request('https://api.anthropic.com/v1/messages',b,{'x-api-key':key,'anthropic-version':'2023-06-01','content-type':'application/json'})
    for a in range(6):
        try: d=json.load(urllib.request.urlopen(rq,timeout=120));break
        except urllib.error.HTTPError as e:
            if e.code in (429,500,502,503,529) and a<5: time.sleep(2**a*2); continue
            raise
    dt=time.perf_counter()-t
    r=''.join(c.get('text','') for c in d['content'] if c.get('type')=='text').strip()
    lab='spam' if re.search(r'LABEL:\s*spam',r,re.I) else ('ham' if re.search(r'LABEL:\s*ham',r,re.I) else '?')
    res.append({'row':m['row'],'label':m['label'],'pred':lab,'sec':round(dt,3),'raw':r,'model':d.get('model'),'stop':d.get('stop_reason'),'usage':d.get('usage')}); F.write(json.dumps(res[-1])+'\n'); F.flush(); print(len(done)+len(res),m['row'],lab,round(dt,2),flush=True)
print(json.dumps({'total_s':round(time.perf_counter()-T0,3),'results':res}))
