
import json, sys, threading, glob
import numpy as np
import concurrent.futures as cf
sys.path.insert(0,'${VP_ROOT}/analysis'); sys.path.insert(0,'${VP_ROOT}')
import run_direct_vp_report_smoke as R
import run_state_pipeline_smoke as SP
import run_temporal_llm_smoke_v2 as sm
import os
os.chdir('${VP_ROOT}')
vocab=R.build_vocab_from_schema('analysis/eval_schema_auto_train.json')
rows=[json.loads(l) for l in open('data/temporal_episode_query_test_stratified_1500_clean_supported.jsonl')]
rows=[r for r in rows if vocab.get(r['ns'])]
by={}
for r in rows: by.setdefault(r['ns'],[]).append(r)
N=400; total=len(rows)
take={ns:max(1,round(len(v)*N/total)) for ns,v in by.items()}
rng=np.random.default_rng(0)
sample=[]
for ns,k in take.items():
    idxs=rng.permutation(len(by[ns]))[:k]
    sample+=[by[ns][i] for i in idxs]
sample=sample[:N]
print('sample:',len(sample),flush=True)
corpus={}
want={(str(r['hadm_id']),r['charttime'],r['exam_name']) for r in sample}
for l in open('analysis/cohort_rad_corpus.jsonl'):
    o=json.loads(l); k=(str(o.get('hadm_id')),o.get('charttime'),o.get('exam_name'))
    if k in want: corpus[k]=o.get('text','')
f=[x for x in glob.glob('analysis/direct_vp_report_gpt-5.4-mini*records_n1500.jsonl') if 'v3_full_ehr' in x][0]
e2e={(str(r['hadm_id']),r['charttime'],r['ns']): r for r in (json.loads(l) for l in open(f)) if not r.get('gen_error')}
rcP=R.load_cache('analysis/rendered_report_gpt-5.4-mini_enriched_probC_fullctx_mini.jsonl')
caches={}
for pm in ['gpt-5.4-mini','gpt-5.6-sol']:
    p=f'analysis/direct_vp_report_parse_{pm}_{R.PARSE_PROMPT_VERSION}.jsonl'
    caches[pm]=(R.load_cache(p), p)
cfgs=sm.load_model_config('llm_api/gpt_56_sol.yaml')
locks={pm: threading.Lock() for pm in caches}
def extract(report, rec, gen_tag, pm):
    r2=dict(rec); r2['_gen_model']=gen_tag
    cfg=dict(cfgs[pm]); cfg['max_retry']=4
    cache,path=caches[pm]
    attrs,ok,_=R.generated_attrs_from_report(report, r2, pm, cfg, cache, path, locks[pm], False)
    return R.pred_pos_from_attrs(attrs, rec['ns'], vocab)
def one(r):
    k3=(str(r['hadm_id']),r['charttime'],r['ns'])
    kc=(str(r['hadm_id']),r['charttime'],r['exam_name'])
    gold_txt=corpus.get(kc,''); e=e2e.get(k3)
    roP=rcP.get(SP.skey('gpt-5.4-mini', r, 'enriched_probC_fullctx_mini_render'))
    if not gold_txt or not e or not roP or roP.get('error'): return None
    predP_txt,_,_=R.parse_report_wrapper(roP['text'])
    out={'k':'|'.join(k3)}
    for pm,suffix in [('gpt-5.4-mini','mini'),('gpt-5.6-sol','sol')]:
        out[f'gold_{suffix}']=sorted(extract(gold_txt, r, 'goldaudit_sol' if pm=='gpt-5.6-sol' else 'goldaudit_mini', pm))
        out[f'e2e_{suffix}']=sorted(extract(e['report'], r, 'direct_vp_report_v3_full_ehr_ed_proc_raw_reports_json_confidence_gpt-5.4-mini', pm))
        out[f'probC_{suffix}']=sorted(extract(predP_txt, r, 'enriched_probC_fullctx_mini_gpt-5.4-mini', pm))
    return out
done=[0]
def wrap(r):
    x=one(r); done[0]+=1
    if done[0]%50==0: print(f'[{done[0]}/{len(sample)}]',flush=True)
    return x
with cf.ThreadPoolExecutor(max_workers=24) as ex:
    res=[x for x in ex.map(wrap, sample) if x]
def micro(gk,pk):
    tp=sum(len(set(x[gk])&set(x[pk])) for x in res); fp=sum(len(set(x[pk])-set(x[gk])) for x in res); fn=sum(len(set(x[gk])-set(x[pk])) for x in res)
    P=tp/max(tp+fp,1); Rc=tp/max(tp+fn,1); return round(2*P*Rc/max(P+Rc,1e-9),4)
print(f'=== 抽取器敏感性(n={len(res)})===')
print(f"mini 口径: 端到端={micro('gold_mini','e2e_mini')}  probC={micro('gold_mini','probC_mini')}")
print(f"sol  口径: 端到端={micro('gold_sol','e2e_sol')}  probC={micro('gold_sol','probC_sol')}")
json.dump(res, open('analysis/extractor_sensitivity_400.json','w'))
print('saved')
