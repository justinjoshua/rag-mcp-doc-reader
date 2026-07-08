"""Web UI for the document query reader.

Run:
    python -m docreader.web            # serves http://127.0.0.1:8000

A single self-contained page (no external CDN): a polished chat UI with
streaming + stop/copy/regenerate, markdown answers, browser upload, document
management, per-source relevance scores, and chat history persisted across
reloads (client-side localStorage).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .answer import (
    answer_question,
    answer_stream,
    current_backend,
    gemini_stats,
    set_backend,
)
from .config import CONFIG
from .ingest import SUPPORTED_EXTENSIONS, ingest_file
from .store import get_store

app = FastAPI(title="Doc Reader")


class Turn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[Turn] = []
    top_k: int | None = None


class DeleteRequest(BaseModel):
    source: str


class BackendRequest(BaseModel):
    backend: str


@app.get("/api/sources")
def sources() -> dict:
    return {
        "sources": get_store().sources(),
        "backend": current_backend(),
        "gemini_available": bool(CONFIG.gemini_api_key),
        "stats": gemini_stats(),
    }


@app.post("/api/backend")
def switch_backend(req: BackendRequest):
    try:
        active = set_backend(req.backend)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    if active == "gemini" and not CONFIG.gemini_api_key:
        return JSONResponse(status_code=400, content={
            "error": "Gemini backend selected but no GEMINI_API_KEY is configured."})
    return {"backend": active}


@app.post("/api/ask")
def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        return JSONResponse(status_code=400, content={"error": "Question is empty."})
    history = [t.model_dump() for t in req.history]
    try:
        result = answer_question(q, history=history, top_k=req.top_k)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {e}"})
    return {
        "answer": result.answer, "backend": result.backend, "error": result.error,
        "found_context": result.found_context,
        "citations": [{"title": c.title, "text": c.text, "score": c.score} for c in result.citations],
        "sources": result.sources,
    }


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        return JSONResponse(status_code=400, content={"error": "Question is empty."})
    history = [t.model_dump() for t in req.history]

    def gen():
        try:
            for ev in answer_stream(q, history=history, top_k=req.top_k):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'delta', 'text': f'Error: {e}'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'error': True, 'citations': [], 'sources': []})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    docs_dir = Path(CONFIG.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    added, skipped = [], []
    for f in files:
        safe = os.path.basename(f.filename or "")
        if not safe or safe in (".", ".."):
            skipped.append({"source": f.filename or "?", "reason": "invalid filename"})
            continue
        ext = Path(safe).suffix.lower()
        dest = docs_dir / safe
        dest.write_bytes(await f.read())
        if ext not in SUPPORTED_EXTENSIONS:
            skipped.append({"source": safe, "reason": f"unsupported type ({ext or 'no extension'})"})
            continue
        try:
            n = ingest_file(dest, source=safe)
        except Exception as e:  # noqa: BLE001
            skipped.append({"source": safe, "reason": f"could not read: {e}"})
            continue
        if n:
            added.append({"source": safe, "chunks": n})
        else:
            skipped.append({"source": safe,
                            "reason": "no extractable text (scanned/image PDF — needs OCR)"})
    return {"added": added, "skipped": skipped, "sources": get_store().sources()}


@app.post("/api/delete")
def delete_doc(req: DeleteRequest):
    removed = get_store().delete_source(req.source)
    docs_dir = Path(CONFIG.docs_dir).resolve()
    file_removed = False
    try:
        fp = (docs_dir / req.source).resolve()
        if docs_dir in fp.parents and fp.is_file():
            fp.unlink()
            file_removed = True
    except Exception:  # noqa: BLE001
        pass
    return {"removed_chunks": removed, "file_removed": file_removed,
            "sources": get_store().sources()}


@app.on_event("startup")
def _warmup() -> None:
    """Load the embedding + re-ranker models in the background at boot so the
    first user question isn't stuck behind a ~40s cold start."""
    import threading

    def load():
        try:
            from .retrieve import retrieve
            retrieve("warmup", top_k=1)  # pulls in embedder + cross-encoder
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=load, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Doc Reader</title>
<style>
  /* ---- Claude-inspired warm theme, light + dark ---------------------- */
  :root, :root[data-theme="dark"]{
    --bg:#232220; --bg-soft:#2a2926; --elev:#2f2e2a;
    --surface:#302f2b; --surface-2:#38362f;
    --border:rgba(255,255,255,.09); --border-2:rgba(255,255,255,.16);
    --fg:#f2efe8; --muted:#a9a299; --faint:#7d766c;
    --user-bubble:#3a382f; --code-bg:rgba(0,0,0,.28);
    --glow-1:rgba(217,119,87,.06); --glow-2:rgba(224,138,104,.035);
    --shadow:0 4px 16px rgba(0,0,0,.3); --grain:.04;
    --field:rgba(0,0,0,.22);
  }
  :root[data-theme="light"]{
    --bg:#faf9f5; --bg-soft:#f3f1ea; --elev:#ffffff;
    --surface:#ffffff; --surface-2:#f4f2ea;
    --border:rgba(60,52,42,.13); --border-2:rgba(60,52,42,.22);
    --fg:#282520; --muted:#6d655b; --faint:#a0968a;
    --user-bubble:#f0ede3; --code-bg:rgba(60,52,42,.07);
    --glow-1:rgba(217,119,87,.06); --glow-2:rgba(217,119,87,.03);
    --shadow:0 4px 14px rgba(70,58,42,.07); --grain:.025;
    --field:rgba(60,52,42,.04);
  }
  :root{
    --accent:#d97757; --accent-2:#bf5c39; --a1:#e0896b; --a2:#cf6a44;
    --glow:rgba(217,119,87,.4); --accent-soft:rgba(217,119,87,.14);
    --on-accent:#fff; --danger:#d9564b; --ok:#5a9e6f;
    --radius:18px;
    --font:"SF Pro Text",-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,system-ui,sans-serif;
    --serif:"Georgia","Iowan Old Style","Times New Roman",serif;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box;}
  html,body{height:100%;}
  body{
    margin:0; color:var(--fg); font:15px/1.65 var(--font); height:100%; overflow:hidden;
    background:
      radial-gradient(900px 520px at 10% -10%, var(--glow-1), transparent 55%),
      radial-gradient(1000px 640px at 92% 4%, var(--glow-2), transparent 55%),
      var(--bg);
    -webkit-font-smoothing:antialiased;
    transition:background-color .35s ease, color .35s ease;
  }
  body::before{
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:var(--grain);
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }
  body>*{position:relative; z-index:1;}
  ::selection{background:var(--accent-soft); color:var(--fg);}
  *::-webkit-scrollbar{width:10px;height:10px;}
  *::-webkit-scrollbar-thumb{background:var(--border-2); border-radius:8px; border:2px solid transparent; background-clip:padding-box;}
  *::-webkit-scrollbar-thumb:hover{background:var(--faint); background-clip:padding-box;}

  /* app shell */
  .app{display:flex; height:100vh;}
  .sidebar{width:266px; flex:none; display:flex; flex-direction:column; gap:9px;
    padding:16px 14px; border-right:1px solid var(--border);
    background:color-mix(in srgb, var(--bg-soft) 70%, var(--bg)); overflow-y:auto;}
  .main{flex:1; min-width:0; display:flex; flex-direction:column; position:relative;}
  .side-brand{display:flex; align-items:center; gap:9px; padding:2px 4px 8px;}
  .mark{width:27px; height:27px; border-radius:8px; display:grid; place-items:center;
    font-size:.82rem; color:var(--on-accent); font-weight:800;
    background:linear-gradient(135deg,var(--a1),var(--a2)); box-shadow:inset 0 1px 0 rgba(255,255,255,.22);}
  .side-brand h1{font-size:1rem; margin:0; font-weight:650; letter-spacing:.1px;}
  .side-btn{width:100%; text-align:left; padding:9px 12px; border-radius:10px; font-size:.85rem;
    font-weight:500; color:var(--fg); background:var(--surface); border:1px solid var(--border);}
  .side-btn:hover{border-color:var(--accent); background:var(--surface-2); filter:none; box-shadow:none;}
  .side-btn.primary{color:var(--on-accent); border:0; font-weight:600;
    background:linear-gradient(135deg,var(--a1),var(--a2));}
  .side-btn.primary:hover{filter:brightness(1.05);}
  .side-label{font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; color:var(--faint);
    font-weight:600; margin:8px 4px 1px; display:flex; align-items:center; gap:6px;}
  .side-label .count{background:var(--surface-2); border:1px solid var(--border); color:var(--muted);
    border-radius:999px; padding:0 7px; font-size:.68rem; text-transform:none; letter-spacing:0; font-weight:500;}
  .seg.wide{width:100%;}
  .seg.wide .seg-btn{flex:1; text-align:center;}
  .doclist{flex:1; min-height:40px; overflow-y:auto; display:flex; flex-direction:column; gap:4px;}
  .side-foot{display:flex; align-items:center; gap:8px; padding-top:10px; margin-top:6px;
    border-top:1px solid var(--border);}
  .menu-btn{display:none; position:absolute; top:12px; left:12px; z-index:5;
    background:var(--surface); color:var(--fg); border:1px solid var(--border-2);}
  .chips{display:flex; gap:6px; flex-wrap:wrap;}
  .chip{background:var(--surface); border:1px solid var(--border); border-radius:999px;
    padding:2px 10px; font-size:.72rem; color:var(--muted);}
  .chip.warn{color:var(--accent-2); border-color:var(--accent-soft); background:var(--accent-soft);}
  /* backend toggle */
  .seg{display:inline-flex; background:var(--surface); border:1px solid var(--border);
    border-radius:999px; padding:2px; gap:2px;}
  .seg-btn{background:transparent; color:var(--muted); border:0; border-radius:999px;
    padding:4px 12px; font-size:.72rem; font-weight:600; box-shadow:none; transition:all .18s ease;}
  .seg-btn:hover:not(:disabled):not(.active){color:var(--fg); box-shadow:none;}
  .seg-btn.active{color:var(--on-accent); background:linear-gradient(135deg,var(--a1),var(--a2));}
  .seg-btn:disabled{opacity:.4; cursor:not-allowed;}
  .spacer{flex:1;}
  button{font:inherit; font-size:.82rem; cursor:pointer; border:0; border-radius:9px; font-weight:600;
    color:var(--on-accent); padding:6px 12px; background:linear-gradient(135deg,var(--a1),var(--a2));
    transition:transform .08s ease, box-shadow .2s ease, filter .2s ease;}
  button:hover{filter:brightness(1.05);}
  button:active{transform:translateY(1px);}
  button:disabled{opacity:.45; cursor:default; box-shadow:none;}
  button.ghost{background:var(--surface); color:var(--muted); border:1px solid var(--border-2);
    font-weight:500; box-shadow:none;}
  button.ghost:hover{color:var(--fg); border-color:var(--accent); background:var(--surface-2); filter:none;}
  button.icon{padding:0; width:32px; height:30px; display:grid; place-items:center; font-size:.95rem;}
  .doc-row{display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:9px;
    border:1px solid var(--border); background:var(--surface);}
  .doc-row:hover{border-color:var(--border-2);}
  .doc-row .fic{flex:none; font-size:.9rem; opacity:.8;}
  .doc-row .name{flex:1; min-width:0; word-break:break-all; color:var(--fg); font-size:.82rem; line-height:1.35;}
  .del{background:transparent; color:var(--faint); border:0; border-radius:7px; box-shadow:none;
    padding:2px 6px; font-size:1rem; line-height:1; opacity:0; transition:opacity .15s ease;}
  .doc-row:hover .del{opacity:1;}
  .del:hover{color:var(--danger); background:var(--accent-soft); box-shadow:none; filter:none;}
  .doclist .empty{color:var(--faint); font-size:.8rem; padding:8px 4px; line-height:1.5;}

  /* chat */
  #chat{flex:1; overflow-y:auto; padding:20px 20px 8px; width:100%; max-width:760px; margin:0 auto;}
  .msg{display:flex; gap:12px; margin:0 0 18px; animation:rise .3s cubic-bezier(.2,.7,.3,1);}
  @keyframes rise{from{opacity:0; transform:translateY(7px);} to{opacity:1; transform:none;}}
  .avatar{flex:none; width:27px; height:27px; border-radius:8px; display:grid; place-items:center;
    font-size:.72rem; color:var(--on-accent); margin-top:1px;
    background:linear-gradient(135deg,var(--a1),var(--a2)); box-shadow:inset 0 1px 0 rgba(255,255,255,.22);}
  .col{flex:1; min-width:0; display:flex; flex-direction:column;}
  .msg.user{flex-direction:row-reverse;}
  .msg.user .col{align-items:flex-end;}
  /* assistant: flat text (no card); user: subtle bubble */
  .bubble{max-width:100%; padding:1px 0 2px; border:0; background:transparent; overflow-wrap:anywhere;}
  .msg.user .bubble{max-width:84%; padding:9px 14px; border:1px solid var(--border);
    background:var(--user-bubble); border-radius:15px;}
  .bubble p{margin:.55em 0;} .bubble p:first-child{margin-top:0;} .bubble p:last-child{margin-bottom:0;}
  .bubble h1,.bubble h2,.bubble h3{margin:.75em 0 .35em; line-height:1.3; font-weight:650;}
  .bubble h1{font-size:1.22rem;} .bubble h2{font-size:1.1rem;} .bubble h3{font-size:1rem;}
  .bubble ul,.bubble ol{margin:.5em 0; padding-left:1.4em;} .bubble li{margin:.2em 0;}
  .bubble strong{color:var(--fg); font-weight:650;}
  .bubble em{color:var(--muted);}
  .bubble code{background:var(--code-bg); padding:1.5px 6px; border-radius:6px; font:.86em var(--mono);}
  .bubble pre{background:var(--code-bg); padding:13px 15px; border-radius:12px; overflow-x:auto;
    border:1px solid var(--border);}
  .bubble pre code{background:none; padding:0;}
  .bubble table{border-collapse:collapse; margin:.7em 0; font-size:.9em; display:block; overflow-x:auto; max-width:100%;}
  .bubble th,.bubble td{border:1px solid var(--border-2); padding:7px 12px; text-align:left; vertical-align:top;}
  .bubble th{background:var(--surface-2); font-weight:650; color:var(--fg);}
  .bubble tbody tr:nth-child(even) td{background:color-mix(in srgb, var(--surface-2) 55%, transparent);}
  .dots{display:inline-flex; gap:5px; padding:3px 0;}
  .dots i{width:7px; height:7px; border-radius:50%; background:var(--accent); opacity:.5;
    animation:bounce 1.2s infinite ease-in-out;}
  .dots i:nth-child(2){animation-delay:.15s;} .dots i:nth-child(3){animation-delay:.3s;}
  @keyframes bounce{0%,80%,100%{transform:translateY(0); opacity:.4;} 40%{transform:translateY(-5px); opacity:1;}}
  .cursor{display:inline-block; width:7px; height:1.05em; background:var(--accent); border-radius:2px;
    vertical-align:text-bottom; animation:blink 1s steps(2) infinite;}
  @keyframes blink{50%{opacity:0;}}

  details.sources{margin-top:12px; font-size:.88rem;}
  details.sources>summary{cursor:pointer; color:var(--accent); list-style:none; font-weight:600; user-select:none;}
  details.sources>summary::before{content:"\25b8\00a0\00a0";}
  details.sources[open]>summary::before{content:"\25be\00a0\00a0";}
  .cite{padding:10px 14px; margin:9px 0; background:var(--surface-2); border:1px solid var(--border);
    border-left:2px solid var(--accent); border-radius:0 12px 12px 0; color:var(--muted);}
  .cite .title{font-weight:600; font-size:.74rem; color:var(--faint); margin-bottom:5px;
    display:flex; justify-content:space-between; gap:8px; align-items:center;}
  .score{font-weight:700; font-variant-numeric:tabular-nums; color:var(--accent-2); font-size:.72rem;
    padding:1px 8px; border-radius:999px; background:var(--accent-soft);}
  .actions{display:flex; gap:8px; margin-top:10px;}
  .act{background:transparent; color:var(--muted); border:1px solid var(--border-2); border-radius:9px;
    padding:4px 12px; font-size:.75rem; font-weight:500; box-shadow:none;}
  .act:hover{color:var(--fg); border-color:var(--accent); background:var(--surface); box-shadow:none; filter:none;}

  /* welcome / empty state */
  .welcome{max-width:600px; margin:5vh auto 0; text-align:center; animation:rise .4s ease;}
  .welcome .hero{width:46px; height:46px; border-radius:14px; margin:0 auto 16px; display:grid; place-items:center;
    font-size:1.25rem; color:var(--on-accent); background:linear-gradient(135deg,var(--a1),var(--a2));
    box-shadow:inset 0 1px 0 rgba(255,255,255,.3);}
  .welcome h2{font-size:1.5rem; font-weight:600; margin:0 0 8px; letter-spacing:-.2px; font-family:var(--serif); color:var(--fg);}
  .welcome p{color:var(--muted); margin:0 0 22px; font-size:.94rem;}
  .examples{display:flex; flex-wrap:wrap; gap:9px; justify-content:center;}
  .ex{background:var(--surface); color:var(--fg); border:1px solid var(--border-2); border-radius:12px;
    padding:10px 14px; font-size:.85rem; font-weight:400; box-shadow:none; text-align:left; max-width:100%;
    transition:transform .12s ease, border-color .18s ease;}
  .ex:hover{border-color:var(--accent); transform:translateY(-1px); filter:none;}

  /* composer */
  footer{padding:8px 20px 16px; position:sticky; bottom:0;
    background:linear-gradient(0deg,var(--bg) 60%,transparent);}
  .composer{display:flex; gap:8px; max-width:760px; margin:0 auto; align-items:flex-end;
    background:var(--elev); border:1px solid var(--border-2); border-radius:18px; padding:6px 6px 6px 12px;
    box-shadow:var(--shadow); transition:border-color .2s ease, box-shadow .2s ease;}
  .composer:focus-within{border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-soft);}
  textarea{flex:1; resize:none; min-height:24px; max-height:180px; padding:7px 6px; border:0; outline:none;
    background:transparent; color:var(--fg); font:inherit;}
  textarea::placeholder{color:var(--faint);}
  label.k{color:var(--faint); font-size:.72rem; white-space:nowrap; align-self:center;}
  input[type=number]{width:42px; padding:5px; border:1px solid var(--border-2); border-radius:8px;
    background:var(--field); color:var(--fg); font:inherit; text-align:center;}
  #send{width:36px; height:36px; padding:0; border-radius:11px; font-size:1.05rem; line-height:1;
    display:grid; place-items:center; flex:none;}
  .foot-hint{max-width:760px; margin:7px auto 0; text-align:center; color:var(--faint); font-size:.72rem;}
  .err{color:var(--danger);}
  #uploadStatus{font-size:.78rem; color:var(--muted); margin-top:8px;}
  @media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important;}}
  @media (max-width:820px){
    .menu-btn{display:grid;}
    .sidebar{position:fixed; left:0; top:0; bottom:0; z-index:20; width:270px;
      transform:translateX(-100%); transition:transform .25s ease; box-shadow:var(--shadow);}
    .sidebar.open{transform:none;}
    #chat{padding-top:56px;}
  }
  @media (max-width:560px){ .msg.user .bubble{max-width:100%;} .welcome h2{font-size:1.5rem;} }
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar" id="sidebar">
    <div class="side-brand"><span class="mark">◆</span><h1>Doc Reader</h1></div>
    <button id="newChat" type="button" class="side-btn primary">＋ New chat</button>

    <input id="file" type="file" multiple style="display:none">
    <button class="side-btn" id="uploadBtn" type="button">⤒ Add documents</button>

    <div class="side-label">Answer mode</div>
    <div id="backendToggle" class="seg wide" role="group" aria-label="Answer mode" title="Switch how answers are produced">
      <button class="seg-btn" data-b="gemini" type="button">Gemini</button>
      <button class="seg-btn" data-b="extractive" type="button">Passages</button>
    </div>

    <div class="side-label side-docs-label">Documents <span id="indexed" class="count"></span></div>
    <div id="manageList" class="doclist"></div>
    <div id="uploadStatus"></div>

    <div class="side-foot">
      <span id="usage" class="chip" title="Gemini API requests this session (in-app estimate, not Google's official count). Resets on server restart."></span>
      <span class="spacer"></span>
      <button class="ghost icon" id="themeToggle" type="button" aria-label="Toggle light / dark theme" title="Toggle light / dark theme"></button>
    </div>
  </aside>

  <main class="main">
    <button class="ghost icon menu-btn" id="menuBtn" type="button" aria-label="Toggle sidebar" title="Toggle sidebar">☰</button>
    <div id="chat"></div>
    <footer>
      <div class="composer">
        <textarea id="q" rows="1" placeholder="Ask about your documents…"></textarea>
        <label class="k">passages <input type="number" id="topk" min="1" max="20" value="6"></label>
        <button id="send" type="button" aria-label="Send">↑</button>
      </div>
      <div class="foot-hint">Enter to send · Shift+Enter for a new line · answers are grounded in your documents</div>
    </footer>
  </main>
</div>

<script>
const $ = (id) => document.getElementById(id);
const LS_KEY = 'docreader_chat_v1';
const THEME_KEY = 'docreader_theme';
let history = [];
let busy = false;
let controller = null;

/* ---- theme (light / dark) ---- */
function applyTheme(t){
  document.documentElement.setAttribute('data-theme', t);
  const b = $('themeToggle');
  if(b){ b.textContent = t === 'dark' ? '☀️' : '☾';
         b.title = 'Switch to ' + (t === 'dark' ? 'light' : 'dark') + ' mode'; }
}
function initTheme(){
  let t = null;
  try{ t = localStorage.getItem(THEME_KEY); }catch(e){}
  if(!t){ t = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'; }
  applyTheme(t);
}
function toggleTheme(){
  const cur = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  const next = cur === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  try{ localStorage.setItem(THEME_KEY, next); }catch(e){}
}
initTheme();

const EXAMPLES = [
  "Summarize the key points across my documents.",
  "What are the most important figures or numbers?",
  "What dates or deadlines are mentioned?",
];
const WELCOME =
  '<div class="welcome" id="welcome">' +
    '<div class="hero">◆</div>' +
    '<h2>Ask your documents</h2>' +
    '<p>Grounded answers with sources — hybrid retrieval &amp; re-ranking, answered by Gemini or fully offline.</p>' +
    '<div class="examples">' +
      EXAMPLES.map(e=>'<button class="ex">'+esc(e)+'</button>').join('') +
    '</div>' +
  '</div>';

function esc(s){ return s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function ficon(name){
  const e = (name.split('.').pop()||'').toLowerCase();
  return ({pdf:'📕',docx:'📘',doc:'📘',pptx:'📙',ppt:'📙',txt:'📄',md:'📝',csv:'📊',json:'🗂️',rst:'📄'})[e] || '📄';
}
function scrollBottom(){ $('chat').scrollTop = $('chat').scrollHeight; }
function clearWelcome(){ const w=$('welcome'); if(w) w.remove(); }
function showWelcome(){ $('chat').innerHTML = WELCOME; }

function md(src){
  const blocks = esc(src).split(/\n{2,}/); const out = [];
  for(let b of blocks){
    b = b.trim(); if(!b) continue;
    const fence = b.match(/^```[a-z]*\n([\s\S]*?)```$/i);
    if(fence){ out.push('<pre><code>'+fence[1]+'</code></pre>'); continue; }
    const h = b.match(/^(#{1,3})\s+(.*)$/);
    if(h){ const n=h[1].length; out.push(`<h${n}>`+inline(h[2])+`</h${n}>`); continue; }
    const lines = b.split('\n');
    // markdown table: header row, a |---|---| separator, then body rows
    if(lines.length>=2 && /\|/.test(lines[0]) && /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/.test(lines[1])){
      const cells = (l)=> l.replace(/^\s*\|/,'').replace(/\|\s*$/,'').split('|').map(c=>c.trim());
      const head = cells(lines[0]);
      const body = lines.slice(2).filter(l=>l.trim()).map(cells);
      let t = '<table><thead><tr>'+head.map(c=>'<th>'+inline(c)+'</th>').join('')+'</tr></thead><tbody>';
      for(const r of body){ t += '<tr>'+head.map((_,i)=>'<td>'+inline(r[i]||'')+'</td>').join('')+'</tr>'; }
      out.push(t+'</tbody></table>'); continue;
    }
    if(lines.every(l=>/^\s*[-*]\s+/.test(l))){
      out.push('<ul>'+lines.map(l=>'<li>'+inline(l.replace(/^\s*[-*]\s+/,''))+'</li>').join('')+'</ul>'); continue;
    }
    if(lines.every(l=>/^\s*\d+\.\s+/.test(l))){
      out.push('<ol>'+lines.map(l=>'<li>'+inline(l.replace(/^\s*\d+\.\s+/,''))+'</li>').join('')+'</ol>'); continue;
    }
    out.push('<p>'+lines.map(inline).join('<br>')+'</p>');
  }
  return out.join('');
}
function inline(s){
  return s.replace(/`([^`]+)`/g,'<code>$1</code>')
          .replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
          .replace(/\*([^*]+)\*/g,'<em>$1</em>');
}

function saveState(){ try{ localStorage.setItem(LS_KEY, JSON.stringify({history, html:$('chat').innerHTML})); }catch(e){} }
function restoreState(){
  try{
    const s = JSON.parse(localStorage.getItem(LS_KEY)||'null');
    if(s && s.html && (s.history||[]).length){ $('chat').innerHTML = s.html; history = s.history; scrollBottom(); return true; }
  }catch(e){}
  return false;
}

function addMsg(role, textOrHtml, isHtml){
  clearWelcome();
  const wrap = document.createElement('div'); wrap.className = 'msg '+role;
  if(role==='assistant'){ const av=document.createElement('div'); av.className='avatar'; av.textContent='◆'; wrap.appendChild(av); }
  const col = document.createElement('div'); col.className='col';
  const bubble = document.createElement('div'); bubble.className='bubble';
  if(isHtml) bubble.innerHTML = textOrHtml; else bubble.textContent = textOrHtml;
  col.appendChild(bubble); wrap.appendChild(col);
  $('chat').appendChild(wrap); scrollBottom();
  return bubble;
}

function addActions(msg){
  const col = msg.querySelector('.col') || msg;
  const a = document.createElement('div'); a.className='actions';
  const copy = document.createElement('button'); copy.className='act act-copy'; copy.textContent='Copy';
  a.appendChild(copy); col.appendChild(a);
}
function markLastRegen(){
  document.querySelectorAll('#chat .act-regen').forEach(b=>b.remove());
  const msgs = document.querySelectorAll('#chat .msg.assistant');
  const last = msgs[msgs.length-1];
  if(last && last.dataset.answer !== undefined){
    let a = last.querySelector('.actions'); if(!a){ addActions(last); a = last.querySelector('.actions'); }
    const b = document.createElement('button'); b.className='act act-regen'; b.textContent='Regenerate'; a.appendChild(b);
  }
}
function setBusy(on){ busy = on; $('send').textContent = on ? '■' : '↑'; $('send').setAttribute('aria-label', on?'Stop':'Send'); }

function renderSources(bubble, citations){
  if(!citations || !citations.length) return;
  const rows = citations.map(c=>{
    const sc = (typeof c.score==='number' && c.score) ? '<span class="score">'+c.score.toFixed(3)+'</span>' : '';
    return '<div class="cite"><div class="title"><span>'+esc(c.title)+'</span>'+sc+'</div>'+esc(c.text)+'</div>';
  }).join('');
  const d = document.createElement('details'); d.className='sources';
  d.innerHTML = '<summary>Sources ('+citations.length+')</summary>'+rows;
  bubble.parentElement.appendChild(d); scrollBottom();
}

async function runAnswer(q, addUser){
  setBusy(true);
  if(addUser) addMsg('user', q, false);
  const bubble = addMsg('assistant', '<span class="dots"><i></i><i></i><i></i></span>', true);
  const wrap = bubble.parentElement.parentElement;   // .msg
  wrap.dataset.q = q;
  controller = new AbortController();
  let acc = '', done = null, errMsg = '', first = true;
  try{
    const resp = await fetch('/api/ask/stream', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:q, history, top_k: parseInt($('topk').value,10)||6}),
      signal: controller.signal
    });
    const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf='';
    while(true){
      const {value, done:fin} = await reader.read(); if(fin) break;
      buf += dec.decode(value, {stream:true});
      const parts = buf.split('\n\n'); buf = parts.pop();
      for(const p of parts){
        const line = p.trim(); if(!line.startsWith('data:')) continue;
        const ev = JSON.parse(line.slice(5).trim());
        if(ev.type==='delta'){ if(first){ acc=''; first=false; } acc += ev.text; bubble.innerHTML = md(acc)+'<span class="cursor"></span>'; scrollBottom(); }
        else if(ev.type==='done'){ done = ev; }
      }
    }
  }catch(e){ if(e.name !== 'AbortError'){ errMsg = String(e); } }

  if(errMsg){ bubble.innerHTML = '<span class="err">'+esc(errMsg)+'</span>'; }
  else { bubble.innerHTML = md(acc) || '<em style="color:var(--faint)">(stopped)</em>'; if(done){ renderSources(bubble, done.citations); renderUsage(done.stats); } }
  wrap.dataset.answer = acc;
  addActions(wrap);
  history.push({role:'user', content:q}); history.push({role:'assistant', content:acc});
  markLastRegen(); saveState();
  setBusy(false); controller=null; $('q').focus();
}

function send(){
  if(busy) return;
  const q = $('q').value.trim(); if(!q) return;
  $('q').value=''; $('q').style.height='auto';
  runAnswer(q, true);
}

function renderUsage(stats){
  const el = $('usage'); if(!el || !stats) return;
  const n = stats.calls||0, e = stats.quota_errors||0;
  el.textContent = 'Gemini: ' + n + (n===1?' request':' requests') + (e? ' · '+e+' quota hit'+(e>1?'s':''):'');
  el.classList.toggle('warn', e>0);
  el.style.display = '';
}

function setBackendToggle(backend, geminiAvailable){
  document.querySelectorAll('#backendToggle .seg-btn').forEach(b=>{
    b.classList.toggle('active', b.dataset.b === backend);
    if(b.dataset.b === 'gemini'){
      b.disabled = !geminiAvailable;
      b.title = geminiAvailable ? 'Written answers via Google Gemini'
        : 'No GEMINI_API_KEY configured — add one in .env to enable';
    } else {
      b.title = 'Retrieval only: returns the matching passages, no API needed';
    }
  });
}

async function switchBackend(backend){
  const prev = document.querySelector('#backendToggle .seg-btn.active');
  setBackendToggle(backend, true);            // optimistic
  try{
    const r = await fetch('/api/backend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({backend})});
    const d = await r.json();
    if(!r.ok){ throw new Error(d.error||'switch failed'); }
    loadSources();
    $('uploadStatus').innerHTML = 'Answer mode: <strong>'+esc(d.backend==='gemini'?'Gemini (written answers)':'Passages (retrieval only)')+'</strong>';
    setTimeout(()=>{ if($('uploadStatus').textContent.startsWith('Answer mode')) $('uploadStatus').textContent=''; }, 3000);
  }catch(e){
    if(prev) prev.classList.add('active');    // revert
    loadSources();
    $('uploadStatus').innerHTML = '<span class="err">'+esc(String(e.message||e))+'</span>';
  }
}

async function loadSources(){
  try{
    const d = await (await fetch('/api/sources')).json();
    setBackendToggle(d.backend, d.gemini_available);
    renderUsage(d.stats);
    const list = d.sources||[];
    $('indexed').textContent = list.length;
    $('manageList').innerHTML = list.length
      ? list.map(s=>'<div class="doc-row"><span class="fic">'+ficon(s)+'</span>'
          +'<span class="name" title="'+esc(s)+'">'+esc(s)+'</span>'
          +'<button class="del" data-src="'+esc(s)+'" title="Delete" aria-label="Delete">&times;</button></div>').join('')
      : '<div class="empty">No documents yet. Use <b>Add documents</b> to upload PDFs, Word, PowerPoint, or text files.</div>';
    $('manageList').querySelectorAll('.del').forEach(b=>b.addEventListener('click',()=>deleteDoc(b.dataset.src)));
  }catch(e){ $('indexed').innerHTML='<span class="chip err">sources unavailable</span>'; }
}

async function deleteDoc(source){
  if(!confirm('Delete "'+source+'" from the index?')) return;
  try{ await fetch('/api/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source})}); loadSources(); }
  catch(e){ $('uploadStatus').innerHTML='<span class="err">Delete failed: '+esc(String(e))+'</span>'; }
}

async function doUpload(fileList){
  if(!fileList.length) return;
  $('uploadStatus').textContent = 'Uploading & indexing '+fileList.length+' file(s)…';
  const fd = new FormData(); for(const f of fileList) fd.append('files', f);
  try{
    const d = await (await fetch('/api/upload',{method:'POST',body:fd})).json();
    const added = (d.added||[]).map(a=>a.source+' ('+a.chunks+' chunks)').join(', ');
    const skipped = (d.skipped||[]).map(s=>(s.source||s)+(s.reason?' — '+s.reason:'')).join('; ');
    let msg='';
    if(added) msg += '✓ Indexed: '+added;
    if(skipped) msg += (msg?'  ·  ':'') + '⚠ Skipped: '+skipped;
    $('uploadStatus').innerHTML = esc(msg || 'Nothing added').replace('⚠','<span class="err">⚠</span>');
    loadSources();
  }catch(e){ $('uploadStatus').innerHTML = '<span class="err">Upload failed: '+esc(String(e))+'</span>'; }
}

$('send').addEventListener('click', ()=>{ if(busy){ if(controller) controller.abort(); } else send(); });
$('chat').addEventListener('click', (e)=>{
  const t = e.target;
  if(t.classList && t.classList.contains('ex')){ $('q').value = t.textContent; send(); }
  else if(t.classList && t.classList.contains('act-copy')){
    const msg = t.closest('.msg'); navigator.clipboard.writeText((msg && msg.dataset.answer)||'');
    t.textContent='Copied'; setTimeout(()=>t.textContent='Copy',1200);
  }
  else if(t.classList && t.classList.contains('act-regen')){
    if(busy) return; const msg = t.closest('.msg'); const q = msg.dataset.q; if(!q) return;
    history.pop(); history.pop(); msg.remove(); runAnswer(q, false);
  }
});
$('q').addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); } });
$('q').addEventListener('input', ()=>{ $('q').style.height='auto'; $('q').style.height=Math.min($('q').scrollHeight,190)+'px'; });
$('backendToggle').addEventListener('click', (e)=>{
  const b = e.target.closest('.seg-btn'); if(!b || b.disabled || b.classList.contains('active')) return;
  switchBackend(b.dataset.b);
});
$('themeToggle').addEventListener('click', toggleTheme);
$('menuBtn').addEventListener('click', ()=> $('sidebar').classList.toggle('open'));
$('uploadBtn').addEventListener('click', ()=>$('file').click());
$('file').addEventListener('change', e=>doUpload(e.target.files));
$('newChat').addEventListener('click', ()=>{ history=[]; localStorage.removeItem(LS_KEY); showWelcome(); });

if(!restoreState()) showWelcome();
markLastRegen();
loadSources();
</script>
</body>
</html>"""
