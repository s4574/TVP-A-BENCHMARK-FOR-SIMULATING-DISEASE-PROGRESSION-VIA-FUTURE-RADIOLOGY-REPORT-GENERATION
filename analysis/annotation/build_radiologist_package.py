"""Build the radiologist annotation package (zip): two self-contained HTML frontends + README.
Task A: independent re-audit of the SAME 100 sealed cases as the author audit (Q1 coverage /
        Q2 contradiction / Q3 spuriousness) -> inter-rater agreement + independent gold verification.
Task B: omission adjudication -- forgiven FPs (prior-positive, target-silent) from the union of
        {probC fullctx, e2e astra, e2e mini}, plus blinded attention controls (target-absent /
        target-present). Directly calibrates the omission-tolerant reading.
No server needed: answers autosave to localStorage; Export downloads a JSON to send back.
MIMIC text inside -> the zip must NOT go into the public release repo.
"""
import collections, csv, html, json, os, random, re, sys
import numpy as np

ROOT = "/workspace/siyan/virtualPatient"
sys.path.insert(0, f"{ROOT}/analysis"); sys.path.insert(0, ROOT)
os.chdir(ROOT)
import run_direct_vp_report_smoke as R
import score_ml_fourstate as F4

OUTDIR = f"{ROOT}/paper/radiologist_annotation"
os.makedirs(OUTDIR, exist_ok=True)
vocab = R.build_vocab_from_schema("analysis/eval_schema_auto_train.json")
rows = [json.loads(l) for l in open("data/temporal_episode_query_test_stratified_1500_clean_supported.jsonl") if l.strip()]
rows = [r for r in rows if vocab.get(r.get("ns"))]
key2row = {"|".join([str(r["hadm_id"]), r["charttime"], r["ns"]]): r for r in rows}

# corpus join: (hadm, charttime, exam_name) -> text ; also (hadm, charttime, ns) fallback
corpus = {}
corpus_ns = {}
for line in open("analysis/cohort_rad_corpus.jsonl"):
    o = json.loads(line)
    h = str(o.get("hadm_id")); ct = o.get("charttime") or ""
    corpus[(h, ct, o.get("exam_name"))] = (o.get("text", ""), o.get("exam_name", ""))
    corpus_ns[(h, ct, "rad.%s_%s" % (o.get("modality"), o.get("region")))] = (o.get("text", ""), o.get("exam_name", ""))

# ---------- Task A: same 100 cases as author audit (seed 0, proportional ns strata) ----------
by_ns = {}
for r in rows: by_ns.setdefault(r["ns"], []).append(r)
total = sum(len(v) for v in by_ns.values()); N = 100
quota = {ns: len(v) * N / total for ns, v in by_ns.items()}
take = {ns: int(q) for ns, q in quota.items()}
for ns in sorted(quota, key=lambda n: -(quota[n] - take[n]))[:N - sum(take.values())]: take[ns] += 1
rng = np.random.default_rng(0)
sample = []
for ns, k in take.items():
    idx = rng.permutation(len(by_ns[ns]))[:max(k, 1) if k else 0]
    sample += [by_ns[ns][i] for i in idx]
sample = sample[:N]

taskA = []
for i, r in enumerate(sorted(sample, key=lambda x: (x["ns"], str(x["hadm_id"])))):
    key = (str(r["hadm_id"]), r["charttime"], r["exam_name"])
    cid = f"{key[0]}|{key[1]}|{r['ns']}"
    text = corpus.get(key, ("", ""))[0]
    icd = "\n".join(dict.fromkeys(m.strip() for m in re.findall(r"^.*ICD:.*$", r.get("context", ""), re.M)))[:1200]
    gold = []
    gd = {k.split(".", 2)[-1]: (v if isinstance(v, str) else (v or {}).get("presence", "")) for k, v in r["gold"].items()}
    for c in vocab[r["ns"]]:
        if c in gd:
            gold.append({"c": c, "s": gd[c], "loc": ""})
    taskA.append({"id": cid, "i": i + 1, "ns": r["ns"], "exam": r["exam_name"], "t": r["charttime"],
                  "icd": icd, "report": text.strip(), "gold": gold,
                  "schema": vocab[r["ns"]]})
print(f"taskA cases: {len(taskA)} (report found {sum(1 for c in taskA if c['report'])})")

# ---------- Task B: forgiven FPs + blinded controls ----------
IDX4 = F4.load_findings_index()

def prior_status_with_source(row):
    """concept -> (presence, ct, ns) with recency override, before target."""
    target_t = row.get("charttime") or ""
    hadms = {str(row.get("hadm_id"))} | {str(x) for x in (row.get("prior_related_hadms") or [])}
    reports = []
    for h in hadms:
        for ct, ns, attrs in IDX4.get(h, []):
            if ct and ct < target_t: reports.append((ct, ns, attrs, h))
    reports.sort(key=lambda e: e[0])
    st = {}
    for ct, ns, attrs, h in reports:
        for k, v in (attrs or {}).items():
            st[F4.concept_name(k)] = ((v or {}).get("presence", "present"), ct, ns, h)
    return st

def load_pp(path, klist=False):
    out = {}
    if klist:
        for o in json.load(open(path)):
            if o["k"] in key2row: out[o["k"]] = set(o["pp"])
    else:
        for l in open(path):
            o = json.loads(l); k = "|".join([str(o["hadm_id"]), o["charttime"], o["ns"]])
            if k in key2row: out[k] = set(o.get("pred_pos", o.get("pp", [])))
    return out

A = "analysis/"
arms = {
    "probC": load_pp(A + "probC_percase_records.json", klist=True),
    "astra": load_pp(A + "direct_vp_report_gpt-6-astra_direct_vp_report_v3_full_ehr_ed_proc_raw_reports_json_confidence_records_n1500.jsonl"),
    "mini": load_pp(A + "direct_vp_report_gpt-5.4-mini_direct_vp_report_v3_full_ehr_ed_proc_raw_reports_json_confidence_records_n1500.jsonl"),
}

forgiven = {}          # (k, c) -> set(arms)
ctrl_absent, ctrl_present = [], []
seen_ctrl = set()
for k, r in key2row.items():
    st = prior_status_with_source(r)
    gs = {kk.split(".", 2)[-1]: (v if isinstance(v, str) else (v or {}).get("presence", "")) for kk, v in (r.get("gold") or {}).items()}
    gp = R.gold_pos(r, vocab)
    for c, (pres, pct, pns, ph) in st.items():
        if pres not in ("present", "uncertain") or c not in vocab[r["ns"]]:
            continue
        if gs.get(c) == "absent" and (k, c) not in seen_ctrl:
            ctrl_absent.append((k, c)); seen_ctrl.add((k, c))
        elif gs.get(c) == "present" and (k, c) not in seen_ctrl:
            ctrl_present.append((k, c)); seen_ctrl.add((k, c))
    for arm, pp in arms.items():
        for c in pp.get(k, set()) - gp:
            src = st.get(c)
            if src and src[0] in ("present", "uncertain") and gs.get(c) != "absent":
                forgiven.setdefault((k, c), set()).add(arm)

print(f"forgiven unique pairs: {len(forgiven)}; ctrl_absent pool: {len(ctrl_absent)}; ctrl_present pool: {len(ctrl_present)}")

rnd = random.Random(7)
fkeys = sorted(forgiven.keys())
rnd.shuffle(fkeys)
picked_f = fkeys[:100]
rnd.shuffle(ctrl_absent); rnd.shuffle(ctrl_present)
picked = [(k, c, "F") for k, c in picked_f] + [(k, c, "CA") for k, c in ctrl_absent[:10]] + [(k, c, "CP") for k, c in ctrl_present[:10]]
rnd.shuffle(picked)

GRP_CODE = {"F": "g1", "CA": "g2", "CP": "g3"}
taskB, mapping = [], []
from datetime import datetime
def days(a, b):
    try:
        return round((datetime.strptime(a, "%Y-%m-%d %H:%M:%S") - datetime.strptime(b, "%Y-%m-%d %H:%M:%S")).days
                     + ((datetime.strptime(a, "%Y-%m-%d %H:%M:%S") - datetime.strptime(b, "%Y-%m-%d %H:%M:%S")).seconds / 86400), 1)
    except Exception:
        return None

skip = 0
for n, (k, c, typ) in enumerate(picked):
    r = key2row[k]
    st = prior_status_with_source(r)
    src = st.get(c)
    if not src: skip += 1; continue
    pres, pct, pns, ph = src
    ptext, pexam = corpus_ns.get((ph, pct, pns), ("", ""))
    ttext = corpus.get((str(r["hadm_id"]), r["charttime"], r["exam_name"]), ("", ""))[0]
    if not ptext or not ttext: skip += 1; continue
    iid = f"B{len(taskB)+1:03d}"
    taskB.append({"id": iid, "concept": c.replace("_", " "), "concept_raw": c,
                  "gap_days": days(r["charttime"], pct),
                  "prior": {"t": pct, "exam": pexam, "text": ptext.strip(), "state": pres},
                  "target": {"t": r["charttime"], "exam": r["exam_name"], "ns": r["ns"], "text": ttext.strip()},
                  "grp": GRP_CODE[typ]})
    mapping.append({"id": iid, "k": k, "concept": c, "type": typ, "arms": sorted(forgiven.get((k, c), []))})
print(f"taskB items: {len(taskB)} (skipped {skip}); groups:",
      collections.Counter(m["type"] for m in mapping))

json.dump(mapping, open(f"{ROOT}/analysis/radiologist_taskB_mapping.json", "w"), indent=1)  # stays OUT of zip

# ---------- frontends ----------
COMMON_CSS = """
body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;background:#f5f6f8;color:#1c2733}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #dde3ea;padding:10px 18px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:5}
header b{font-size:15px}
.badge{background:#eef3fa;border:1px solid #cfdcef;border-radius:12px;padding:2px 10px;font-size:12px}
button{cursor:pointer;border:1px solid #b9c6d6;background:#fff;border-radius:6px;padding:6px 14px;font-size:13px}
button.primary{background:#2a78d6;border-color:#2a78d6;color:#fff}
main{max-width:1060px;margin:14px auto;padding:0 16px}
.card{background:#fff;border:1px solid #dde3ea;border-radius:10px;padding:16px 18px;margin-bottom:14px}
pre{white-space:pre-wrap;background:#fafbfc;border:1px solid #e6ebf1;border-radius:8px;padding:10px 12px;font-size:13px;line-height:1.45;max-height:420px;overflow:auto}
table{border-collapse:collapse;font-size:13px}
td,th{border:1px solid #dbe2ea;padding:4px 10px}
th{background:#f0f4f8}
.q{margin:10px 0;padding:10px 12px;background:#f8fafc;border-radius:8px}
.q b{display:block;margin-bottom:6px}
label{margin-right:14px;font-size:13.5px}
input[type=text],textarea{width:100%;box-sizing:border-box;border:1px solid #c7d2de;border-radius:6px;padding:6px;font-size:13px;margin-top:4px}
.nav{display:flex;gap:8px;align-items:center;margin:10px 0}
.done{color:#1baf7a;font-weight:600}
.jump{display:flex;flex-wrap:wrap;gap:3px;margin:6px 0}
.jump span{width:22px;height:22px;font-size:10.5px;display:flex;align-items:center;justify-content:center;border-radius:4px;background:#e8edf3;cursor:pointer}
.jump span.a{background:#1baf7a;color:#fff}
.jump span.cur{outline:2px solid #2a78d6}
.meta{color:#5a6b7d;font-size:12.5px;margin-bottom:8px}
.hl{background:#fff3cd;padding:0 3px;border-radius:3px}
.collapsible summary{cursor:pointer;font-weight:600;font-size:13.5px}
@media (max-width:640px){main{padding:0 8px;margin:8px auto}.card{padding:10px 12px}pre{font-size:12px;max-height:300px}header{padding:8px 10px;gap:8px}label{display:block;margin:4px 0}td,th{padding:3px 6px;font-size:12px}}
"""

JS_STORE = """
function E(x){return String(x==null?'':x).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
function store(){return JSON.parse(localStorage.getItem(KEY)||'{}')}
function save(o){localStorage.setItem(KEY,JSON.stringify(o))}
function setAns(id,field,val){const s=store();s[id]=s[id]||{};s[id][field]=val;s[id]._t=new Date().toISOString();save(s);refresh();schedulePush()}
function getAns(id){return store()[id]||{}}
let _pushT=null;
function schedulePush(){clearTimeout(_pushT);_pushT=setTimeout(pushServer,700)}
function annotName(){return (document.getElementById('annot').value||'anonymous').trim()||'anonymous'}
function syncMsg(t){const el=document.getElementById('sync');if(el)el.textContent=t}
async function pushServer(){
 try{
  const r=await fetch('api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:TASK,annotator:annotName(),answers:store()})});
  if(r.ok){const j=await r.json();syncMsg('已保存到服务器 '+new Date().toLocaleTimeString()+' · '+j.n+' 条');return}
  syncMsg('服务器保存失败,已存本机');
 }catch(e){syncMsg(location.protocol==='file:'?'本机模式(结束后请导出)':'离线,已存本机')}
}
async function pullServer(){
 try{
  const r=await fetch('api/load?task='+TASK+'&annotator='+encodeURIComponent(annotName()));
  if(!r.ok)return;const j=await r.json();if(!j.answers)return;
  const s=store();
  for(const k in j.answers){const v=j.answers[k];if(!s[k]||((v._t||'')>(s[k]._t||'')))s[k]=v}
  save(s);
 }catch(e){}
}
function exportJSON(){
  const s=store();const name=document.getElementById('annot').value||'anonymous';
  const out={task:TASK,annotator:name,exported:new Date().toISOString(),n_items:DATA.length,answers:s};
  const b=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download=TASK+'_answers_'+name.replace(/\\s+/g,'_')+'.json';a.click();
}
"""

def esc(s): return html.escape(s or "")

# ---- Task A html ----
a_json = json.dumps(taskA, ensure_ascii=False).replace("</", "<\\/")
taskA_html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>任务A · 抽取标签独立核验 (100例)</title><style>{COMMON_CSS}</style></head><body>
<header><b>任务A · 抽取标签独立核验</b><span class="badge" id="prog"></span><span class="badge" id="sync">…</span>
标注者:<input type="text" id="annot" style="width:130px" placeholder="姓名/代号" onchange="localStorage.setItem(KEY+'_annot',this.value);pullServer().then(()=>{{refresh();render()}})">
<button class="primary" onclick="exportJSON()">导出结果 JSON</button></header>
<main>
<div class="card"><b>怎么做</b>:每例给出真实放射报告原文与基准系统从中抽取的结构化标签(四态;absent=报告明确否定;未列出的概念=报告未提及)。请回答三问。
每题作答即自动保存(本机浏览器),可随时关闭续做;全部/部分完成后点"导出结果 JSON"发回。<b>按顺序做,做到哪算哪;前 50 例为核心样本。</b></div>
<div class="jump" id="jump"></div>
<div class="nav"><button onclick="go(cur-1)">← 上一例</button><span id="pos"></span><button onclick="go(cur+1)">下一例 →</button></div>
<div id="case"></div>
</main>
<script>
const DATA={a_json};const TASK='taskA';const KEY='rad_taskA_v1';let cur=0;
document.getElementById('annot').value=localStorage.getItem(KEY+'_annot')||'';
{JS_STORE}
function isDone(a){{return a.q1&&a.q2&&a.q3}}
function scale(id,f,val,anchors){{
 let h=anchors.map((t,j)=>`<label style="display:block;margin:3px 0"><input type="radio" name="${{id}}_${{f}}" ${{String(val)===String(j+1)?'checked':''}} onchange="setAns('${{id}}','${{f}}',${{j+1}})"> <b>${{j+1}}</b> · ${{t}}</label>`).join('');
 return h}}
function render(){{
 const c=DATA[cur];const a=getAns(c.id);
 const goldRows=c.gold.length?c.gold.map(g=>`<tr><td>${{g.c}}</td><td>${{g.s}}</td><td>${{g.loc}}</td></tr>`).join(''):'<tr><td colspan=3>(该例无 present/absent/uncertain 抽取;全部概念均为 not_mentioned)</td></tr>';
 document.getElementById('case').innerHTML=`
 <div class="card">
 <div class="meta">case ${{c.i}}/100 · <code>${{c.id}}</code> · ${{c.exam}} (${{c.ns}}) @ ${{c.t}}</div>
 <details class="collapsible"><summary>诊断参考(pre-exam ICD 行,仅供背景)</summary><pre>${{E(c.icd)||'(无)'}}</pre></details>
 <b>真实目标报告原文</b><pre>${{E(c.report)||'【未找到报告文本,跳过此例并在备注说明】'}}</pre>
 <b>基准抽取标签(打分依据;not_mentioned 未列出)</b>
 <table><tr><th>concept</th><th>state</th><th>location</th></tr>${{goldRows}}</table>
 <div class="meta" style="margin-top:6px">该检查类型的全部 schema 概念:${{c.schema.join(', ')}}</div>
 <div class="q"><b>Q1 覆盖度:上表标签对报告重点所见的覆盖程度?(1–5)</b>
 ${{scale(c.id,'q1',a.q1,['几乎未覆盖(漏多项主要所见)','覆盖差(漏 1 项主要所见及若干次要点)','部分覆盖(漏多个次要点)','覆盖良好(漏个别次要点)','完全覆盖'])}}
 <input type="text" placeholder="漏掉了哪些(概念名或所见,可留空)" value="${{E(a.q1_missed)}}" onchange="setAns('${{c.id}}','q1_missed',this.value)"></div>
 <div class="q"><b>Q2 矛盾:标签与报告事实的矛盾程度?(1–5)</b>
 ${{scale(c.id,'q2',a.q2,['无任何矛盾','个别措辞歧义,基本无矛盾','轻微矛盾(定位/程度出入)','明确矛盾(次要所见 有↔无 颠倒)','严重矛盾(主要所见 有↔无 颠倒)'])}}
 <input type="text" placeholder="哪些概念矛盾(可留空)" value="${{E(a.q2_which)}}" onchange="setAns('${{c.id}}','q2_which',this.value)"></div>
 <div class="q"><b>Q3 冗余:报告中并不存在/无意义的多余标签程度?(1–5)</b>
 ${{scale(c.id,'q3',a.q3,['无冗余','1 项可疑冗余','1–2 项明确冗余','多项冗余','大量冗余(标签明显不可信)'])}}
 <input type="text" placeholder="哪些概念冗余(可留空)" value="${{E(a.q3_which)}}" onchange="setAns('${{c.id}}','q3_which',this.value)"></div>
 <div class="q"><b>备注(可留空)</b><textarea rows="2" onchange="setAns('${{c.id}}','notes',this.value)">${{E(a.notes)}}</textarea></div>
 </div>`;
 document.getElementById('pos').textContent=`第 ${{cur+1}} / ${{DATA.length}} 例`;
}}
function refresh(){{
 const s=store();const done=DATA.filter(c=>isDone(s[c.id]||{{}})).length;
 document.getElementById('prog').textContent=`已完成 ${{done}} / ${{DATA.length}}`;
 document.getElementById('jump').innerHTML=DATA.map((c,i)=>`<span class="${{isDone(s[c.id]||{{}})?'a':''}} ${{i===cur?'cur':''}}" onclick="go(${{i}})" title="${{c.id}}">${{i+1}}</span>`).join('');
}}
function go(i){{if(i<0||i>=DATA.length)return;cur=i;render();refresh();window.scrollTo(0,0)}}
go(0);if(location.protocol==='file:'){{syncMsg('本机模式(结束后请导出)')}}else{{pullServer().then(()=>{{refresh();render()}})}};
</script></body></html>"""

# ---- Task B html ----
b_json = json.dumps(taskB, ensure_ascii=False).replace("</", "<\\/")
taskB_html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>任务B · 未提及所见的持续性判定 ({len(taskB)}项)</title><style>{COMMON_CSS}</style></head><body>
<header><b>任务B · 未提及所见的持续性判定</b><span class="badge" id="prog"></span><span class="badge" id="sync">…</span>
标注者:<input type="text" id="annot" style="width:130px" placeholder="姓名/代号" onchange="localStorage.setItem(KEY+'_annot',this.value);pullServer().then(()=>{{refresh();render()}})">
<button class="primary" onclick="exportJSON()">导出结果 JSON</button></header>
<main>
<div class="card"><b>怎么做</b>:每项给出同一患者的一份<b>既往报告</b>(其中记录了某一所见)与一份<b>之后的目标报告</b>。
请基于两份报告全文与你的临床经验判断:<b>在目标检查时点,该所见是否大概率仍然存在?</b>
注意:目标报告未提及某所见,可能是已消退,也可能只是本次书写未重复(如慢性/稳定所见)。每项约 30–60 秒;作答自动保存;<b>按顺序做,做到哪算哪;前 60 项为核心样本。</b></div>
<div class="jump" id="jump"></div>
<div class="nav"><button onclick="go(cur-1)">← 上一项</button><span id="pos"></span><button onclick="go(cur+1)">下一项 →</button></div>
<div id="case"></div>
</main>
<script>
const DATA={b_json};const TASK='taskB';const KEY='rad_taskB_v1';let cur=0;
document.getElementById('annot').value=localStorage.getItem(KEY+'_annot')||'';
{JS_STORE}
const ANCHORS=['几乎确定已不存在','倾向已不存在','五五开,无法判断','倾向仍存在','几乎确定仍存在'];
function isDone(a){{return !!a.v||!!a.flagM||!!a.flagN}}
function render(){{
 const c=DATA[cur];const a=getAns(c.id);
 document.getElementById('case').innerHTML=`
 <div class="card">
 <div class="meta">item ${{cur+1}}/${{DATA.length}} · <code>${{c.id}}</code> · 间隔 ${{c.gap_days}} 天</div>
 <div style="font-size:16px;margin:6px 0"><b>判定所见:<span class="hl">${{c.concept}}</span></b>(既往报告标记为 ${{c.prior.state}})</div>
 <b>既往报告</b> <span class="meta">${{c.prior.exam}} @ ${{c.prior.t}}</span><pre>${{E(c.prior.text)}}</pre>
 <b>目标报告</b> <span class="meta">${{c.target.exam}} (${{c.target.ns}}) @ ${{c.target.t}}</span><pre>${{E(c.target.text)}}</pre>
 <div class="q"><b>在目标检查时点,「${{c.concept}}」仍存在的可能性?(1–5)</b>
 ${{ANCHORS.map((t,j)=>`<label style="display:block;margin:4px 0"><input type="radio" name="${{c.id}}_v" ${{String(a.v)===String(j+1)?'checked':''}} onchange="setAns('${{c.id}}','v',${{j+1}})"> <b>${{j+1}}</b> · ${{t}}</label>`).join('')}}</div>
 <div class="q"><b>例外情况(如适用请勾选,勾选后可不打分)</b>
 <label style="display:block"><input type="checkbox" ${{a.flagM?'checked':''}} onchange="setAns('${{c.id}}','flagM',this.checked)"> 目标报告其实已提及该所见(无论肯定或否定)</label>
 <label style="display:block"><input type="checkbox" ${{a.flagN?'checked':''}} onchange="setAns('${{c.id}}','flagN',this.checked)"> 既往报告并不支持该所见(既往标记有误;同义表述算支持,如 opacification ≈ opacity)</label></div>
 <div class="q"><b>备注(可留空)</b><textarea rows="2" onchange="setAns('${{c.id}}','notes',this.value)">${{E(a.notes)}}</textarea></div>
 </div>`;
 document.getElementById('pos').textContent=`第 ${{cur+1}} / ${{DATA.length}} 项`;
}}
function refresh(){{
 const s=store();const done=DATA.filter(c=>isDone(s[c.id]||{{}})).length;
 document.getElementById('prog').textContent=`已完成 ${{done}} / ${{DATA.length}}`;
 document.getElementById('jump').innerHTML=DATA.map((c,i)=>`<span class="${{isDone(s[c.id]||{{}})?'a':''}} ${{i===cur?'cur':''}}" onclick="go(${{i}})" title="${{c.id}}">${{i+1}}</span>`).join('');
}}
function go(i){{if(i<0||i>=DATA.length)return;cur=i;render();refresh();window.scrollTo(0,0)}}
go(0);if(location.protocol==='file:'){{syncMsg('本机模式(结束后请导出)')}}else{{pullServer().then(()=>{{refresh();render()}})}};
</script></body></html>"""

open(f"{OUTDIR}/taskA_核验.html", "w").write(taskA_html)
open(f"{OUTDIR}/taskB_持续性判定.html", "w").write(taskB_html)

README = f"""# 影像科医生标注包(3 名标注员 · 每人预计 1–4 小时,做到哪算哪)

感谢参与!这是一个医学 AI 基准数据集的独立临床核验。**只需一台自己的电脑 + 现代浏览器(Chrome/Edge/Safari),解压后双击 HTML 即可,无需联网、无需安装任何东西。**

## 保密要求(重要)
- 材料为 **MIMIC-IV 去标识临床文本**,受数据使用协议(PhysioNet DUA)约束:请勿转发本包、勿截图外传、勿将报告文本粘贴进任何在线工具(包括 ChatGPT 等 AI 服务)。
- 标注完成后请删除本包,只回传导出的 answers JSON(其中不含报告原文)。

## 三名标注员须知
- **各自独立完成,标注期间请勿相互讨论病例**(结束后欢迎讨论)。
- 打开页面后先在顶部填写自己的**姓名/代号**(每次导出都会带上,三人区分全靠它,请保持一致)。
- 三人做的是**相同的题目、相同的顺序**;都从第 1 题开始按顺序做,做到哪算哪。

## 两个任务(每个任务内按顺序做,做多少算多少)

### 任务 A · 抽取标签独立核验(`taskA_核验.html`,100 例,约 1–2 分钟/例)
基准的"金标准"由算法从真实报告中抽取。请你独立核对:每例给出报告原文 + 抽取的结构化标签,对三个维度各打 1–5 分(每档含义见页面内说明):
- **Q1 覆盖度**:标签对报告重点所见的覆盖程度(1=几乎未覆盖 … 5=完全覆盖)
- **Q2 矛盾**:标签与报告事实的矛盾程度(1=无矛盾 … 5=主要所见有↔无颠倒)
- **Q3 冗余**:报告中并不存在的多余标签程度(1=无冗余 … 5=大量冗余)
判断口径:absent = 报告**明确否定**;未列出的概念 = 报告未提及(不算漏)。只需对照列出的 schema 概念,不要求穷尽报告的一切细节。**前 50 例为核心样本。**

### 任务 B · 未提及所见的持续性判定(`taskB_持续性判定.html`,{len(taskB)} 项,约 30–60 秒/项)
同一患者:一份既往报告记录了某所见,之后的目标报告对它**未置可否或未提及**。请判断在目标检查时点该所见**仍存在的可能性**,打 1–5 分:
- **1** 几乎确定已不存在 · **2** 倾向已不存在 · **3** 五五开/无法判断 · **4** 倾向仍存在 · **5** 几乎确定仍存在
两个例外勾选(如适用,勾选后可不打分):
- **目标报告其实已提及该所见**(无论肯定或否定,说明我们的系统判定有误)
- **既往报告并不支持该所见**(既往标记本身有误;注意同义表述算支持,如 opacification ≈ opacity)
请基于两份报告全文 + 间隔天数 + 你的临床经验判断,不必追求确定,选"最可能"。**前 60 项为核心样本。**(其中混有少量校验项,正常作答即可。)

## 保存与回传
- 每次作答**自动保存在本机浏览器**(localStorage),关闭后重新打开同一文件可继续(请始终用同一台电脑同一浏览器)。
- 完成(或时间用尽)后,确认页面顶部姓名/代号无误,点 **"导出结果 JSON"**,把下载的 `taskA_answers_*.json` / `taskB_answers_*.json` 两个小文件发回即可(不含报告原文)。

## 时间预算参考(每人)
- 只有 1 小时:任务 A 前 30 例 + 任务 B 前 40 项。
- 有 2–3 小时:任务 A 前 50–100 例 + 任务 B 前 60 项。
- 富余:全部完成(A 100 + B {len(taskB)})。

有任何口径疑问,在该例"备注"里写下你的理解即可,不必中断。再次感谢!
"""
open(f"{OUTDIR}/README_请先读我.md", "w").write(README)

import zipfile
zp = f"{ROOT}/paper/radiologist_annotation.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in ["README_请先读我.md", "taskA_核验.html", "taskB_持续性判定.html"]:
        z.write(f"{OUTDIR}/{fn}", f"radiologist_annotation/{fn}")
print("zip:", zp, os.path.getsize(zp), "bytes")
