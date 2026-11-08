# Copy of claude300.py adapted for Claude Haiku 5.5: no temperature, output_config.effort=low, max_tokens 80.
# Same job300.json sample, system prompt, label regex, resume. Records stop_reason, thinking blocks, usage.
import json,os,re,sys,time,urllib.request,urllib.error
J=json.load(open('job300.json'));key=os.environ['ANTHROPIC_API_KEY']
MODEL=os.environ.get('CLAUDE_MODEL','claude-haiku-5-5');OUT=os.environ.get('CLAUDE_OUT','haiku55_300.jsonl')
ROWS=os.environ.get('ROWS')  # optional comma list for probe
msgs=J['msgs'] if not ROWS else [m for m in J['msgs'] if str(m['row']) in ROWS.split(',')]
done=set()
if os.path.exists(OUT): done={json.loads(l)['row'] for l in open(OUT)}
F=open(OUT,'a');n=len(done)
for m in msgs:
    if m['row'] in done: continue
    b=json.dumps({'model':MODEL,'max_tokens':80,'output_config':{'effort':'low'},'system':J['sys'],'messages':[{'role':'user','content':f'Message: """{m["text"]}"""'}]}).encode()
    rq=urllib.request.Request('https://api.anthropic.com/v1/messages',b,{'x-api-key':key,'anthropic-version':'2023-06-01','content-type':'application/json'})
    t=time.perf_counter()
    for a in range(6):
        try: d=json.load(urllib.request.urlopen(rq,timeout=120));break
        except urllib.error.HTTPError as e:
            if e.code in (429,500,502,503,529) and a<5: time.sleep(2**a*2); continue
            print('HTTP',e.code,e.read().decode()[:500]);sys.exit(1)
    dt=time.perf_counter()-t
    r=''.join(c.get('text','') for c in d['content'] if c.get('type')=='text').strip()
    th=[c for c in d['content'] if c.get('type') in('thinking','redacted_thinking')]
    lab='spam' if re.search(r'LABEL:\s*spam',r,re.I) else ('ham' if re.search(r'LABEL:\s*ham',r,re.I) else '?')
    rec={'row':m['row'],'label':m['label'],'pred':lab,'sec':round(dt,3),'raw':r,'model':d.get('model'),'stop':d.get('stop_reason'),'thinking_blocks':len(th),'thinking_chars':sum(len(c.get('thinking','')) for c in th),'usage':d.get('usage')}
    F.write(json.dumps(rec)+'\n');F.flush();n+=1;print(n,m['row'],m['label'],lab,rec['stop'],len(th),d.get('usage',{}).get('output_tokens'),round(dt,2),flush=True)
