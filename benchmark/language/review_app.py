"""
benchmark/language/review_app.py
================================
A click-through front end for ``review.py``: one turn at a time, keep / drop /
fix, written straight back into ``data/review.csv`` after every click.

    python3 benchmark/language/review_app.py              # the flagged turns
    python3 benchmark/language/review_app.py --all        # every undecided turn
    python3 benchmark/language/review_app.py --all --redo # every turn, decided or not

Then open the printed URL. It binds to localhost only and the dataset never
leaves this machine. Keys: k keep, d drop, f fix, arrows to move.

``--queue`` defaults to ``data/model_pass.csv`` (row, id, turn, suggest, why):
a pass over the paraphrases done by Claude, shown as a suggestion to accept or
override, never written as a decision. Nothing is pre-filled.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import html
import json
import os
import socket
import sys
import tempfile
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "data" / "review.csv"
QUEUE = HERE / "data" / "model_pass.csv"
DECISIONS = ("ok", "fix", "drop")


# --------------------------------------------------------------------------- data

def load(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        return list(rd.fieldnames or []), list(rd)


def save(path: Path, cols: list[str], rows: list[dict]) -> None:
    """Write through a temp file in the same directory, then replace: a crash
    or a killed server never leaves a half-written review behind."""
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".review-", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        os.chmod(tmp, mode)                # mkstemp is 0600; keep the file's own mode
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def load_queue(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    out = {}
    for r in load(path)[1]:
        try:
            out[int(r["row"])] = {"suggest": r.get("suggest", ""), "why": r.get("why", "")}
        except (KeyError, ValueError):
            continue
    return out


def word_diff(a: str, b: str) -> tuple[str, str]:
    """``a`` and ``b`` marked up word by word, so a dropped 'Then' is visible."""
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=aw, b=bw, autojunk=False)
    left, right = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ta = html.escape(" ".join(aw[i1:i2]))
        tb = html.escape(" ".join(bw[j1:j2]))
        if tag == "equal":
            left.append(ta)
            right.append(tb)
        else:
            if ta:
                left.append(f'<span class="gone">{ta}</span>')
            if tb:
                right.append(f'<span class="new">{tb}</span>')
    return " ".join(left), " ".join(right)


# --------------------------------------------------------------------------- server

class App:
    def __init__(self, csv_path: Path, queue: dict[int, dict], which: str, redo: bool):
        self.path = csv_path
        self.cols, self.rows = load(csv_path)
        self.queue = queue
        if which == "flagged":
            idx = sorted(i for i in queue if i < len(self.rows))
        else:
            idx = list(range(len(self.rows)))
        if not redo:
            idx = [i for i in idx if self.rows[i].get("decision", "").strip().lower()
                   not in DECISIONS]
        self.order = idx

    def payload(self) -> dict:
        items = []
        for i in self.order:
            r = self.rows[i]
            tl, pr = word_diff(r["template"], r["paraphrase"])
            q = self.queue.get(i, {})
            items.append({"row": i, "id": r["id"], "turn": r["turn"], "town": r["town"],
                          "functions": r["functions"], "route": r["route_before"],
                          "gold": r["gold"], "template": r["template"],
                          "paraphrase": r["paraphrase"], "auto": r["auto_check"],
                          "t_diff": tl, "p_diff": pr,
                          "suggest": q.get("suggest", ""), "why": q.get("why", ""),
                          "decision": r.get("decision", ""),
                          "fixed": r.get("fixed_utterance", ""), "note": r.get("note", "")})
        done = sum(1 for r in self.rows
                   if r.get("decision", "").strip().lower() in DECISIONS)
        return {"items": items, "total_rows": len(self.rows), "decided": done,
                "csv": str(self.path)}

    def decide(self, row: int, decision: str, fixed: str, note: str) -> dict:
        if not 0 <= row < len(self.rows):
            raise ValueError("row out of range")
        if decision not in DECISIONS and decision != "":
            raise ValueError("decision must be ok / fix / drop, or empty to clear")
        if decision == "fix" and not fixed.strip():
            raise ValueError("a fix needs a sentence")
        r = self.rows[row]
        r["decision"] = decision
        r["fixed_utterance"] = fixed.strip() if decision == "fix" else ""
        r["note"] = note.strip()
        save(self.path, self.cols, self.rows)
        done = sum(1 for x in self.rows
                   if x.get("decision", "").strip().lower() in DECISIONS)
        return {"ok": True, "decided": done, "total_rows": len(self.rows)}


class Handler(BaseHTTPRequestHandler):
    app: App

    def log_message(self, *a):            # one line per click is just noise
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/rows":
            self._send(200, json.dumps(self.app.payload()).encode(),
                       "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/decide":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            d = json.loads(self.rfile.read(n) or b"{}")
            out = self.app.decide(int(d["row"]), str(d.get("decision", "")),
                                  str(d.get("fixed", "")), str(d.get("note", "")))
        except Exception as e:                                  # back to the page
            self._send(400, json.dumps({"error": str(e)}).encode(),
                       "application/json; charset=utf-8")
            return
        self._send(200, json.dumps(out).encode(), "application/json; charset=utf-8")


# --------------------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paraphrase review</title>
<style>
  :root{--bg:#fbfaf8;--fg:#1a1a18;--dim:#6b6b66;--line:#e2e0da;--card:#fff;
        --gone:#c0392b;--goneb:#fdecea;--new:#1a6b3c;--newb:#e8f5ec;
        --keep:#1a6b3c;--drop:#b05a12;--fix:#2b5fa8;--warn:#fff8e6;--warnl:#d9a406}
  @media (prefers-color-scheme:dark){:root:not([data-theme=light]){
        --bg:#17171a;--fg:#ececea;--dim:#9a9a94;--line:#32323a;--card:#1f1f24;
        --gone:#ff8f84;--goneb:#3a1f1d;--new:#7fd6a0;--newb:#15301f;
        --keep:#7fd6a0;--drop:#e8a35c;--fix:#8fb8f0;--warn:#2e2a1c;--warnl:#d9a406}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,
       BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;padding:0 16px 64px}
  .wrap{max-width:860px;margin:0 auto}
  header{position:sticky;top:0;background:var(--bg);padding:18px 0 12px;
         border-bottom:1px solid var(--line);z-index:5}
  .bar{height:5px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:10px}
  .bar>i{display:block;height:100%;background:var(--keep);transition:width .2s}
  .meta{display:flex;gap:14px;flex-wrap:wrap;align-items:baseline;
        font-size:13px;color:var(--dim)}
  .meta b{color:var(--fg);font-size:15px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;
        padding:18px;margin-top:18px}
  .row{display:grid;grid-template-columns:92px 1fr;gap:10px;padding:5px 0;
       border-top:1px solid var(--line);font-size:14px}
  .row:first-child{border-top:0}
  .row>span:first-child{color:var(--dim);font-size:12.5px;text-transform:uppercase;
       letter-spacing:.04em;padding-top:3px}
  code{font:13.5px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
       word-break:break-word}
  .sent{font-size:17px;line-height:1.55;margin:4px 0}
  .gone{background:var(--goneb);color:var(--gone);border-radius:3px;padding:0 3px;
        text-decoration:line-through}
  .new{background:var(--newb);color:var(--new);border-radius:3px;padding:0 3px}
  .sug{margin-top:16px;padding:12px 14px;background:var(--warn);
       border-left:3px solid var(--warnl);border-radius:0 8px 8px 0;font-size:14px}
  .sug b{font-variant:all-small-caps;letter-spacing:.05em}
  .btns{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}
  button{font:inherit;font-weight:600;padding:11px 18px;border-radius:8px;
         border:1px solid var(--line);background:var(--card);color:var(--fg);
         cursor:pointer}
  button:hover{border-color:var(--dim)}
  button.on{color:#fff}
  #keep.on{background:var(--keep);border-color:var(--keep)}
  #drop.on{background:var(--drop);border-color:var(--drop)}
  #fix.on{background:var(--fix);border-color:var(--fix)}
  .nav{display:flex;gap:10px;justify-content:space-between;margin-top:14px;
       font-size:14px;color:var(--dim);align-items:center}
  .nav button{font-weight:400;padding:7px 14px}
  kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;
      padding:1px 5px;background:var(--bg)}
  textarea{width:100%;font:inherit;padding:10px;border-radius:8px;
           border:1px solid var(--line);background:var(--bg);color:var(--fg);
           margin-top:10px;resize:vertical}
  .hide{display:none}
  .done{text-align:center;padding:60px 0;color:var(--dim)}
  .done b{display:block;color:var(--fg);font-size:19px;margin-bottom:8px}
  .err{color:var(--gone);font-size:13.5px;margin-top:8px}
  footer{margin-top:22px;font-size:12.5px;color:var(--dim);text-align:center}
</style></head><body><div class="wrap">
<header>
  <div class="meta">
    <b id="pos">…</b><span id="rowid"></span><span id="tag"></span>
    <span style="margin-left:auto" id="tally"></span>
  </div>
  <div class="bar"><i id="bar" style="width:0"></i></div>
</header>
<div id="main"></div>
<footer id="foot"></footer>
</div>
<script>
let items=[],at=0,state={};

const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",
  ">":"&gt;",'"':"&quot;"}[c]));

async function boot(){
  const r=await fetch("/api/rows"); const d=await r.json();
  items=d.items; state={decided:d.decided,total:d.total_rows};
  document.getElementById("foot").textContent="写回 "+d.csv+" · 每点一次立即保存";
  at=0; render();
}

function render(){
  const m=document.getElementById("main");
  document.getElementById("tally").textContent=
    "全库已判 "+state.decided+" / "+state.total;
  if(at>=items.length){
    document.getElementById("pos").textContent="队列完成";
    document.getElementById("rowid").textContent="";
    document.getElementById("tag").textContent="";
    document.getElementById("bar").style.width="100%";
    m.innerHTML='<div class="done"><b>这一队判完了</b>'+
      '全库 '+state.decided+' / '+state.total+' 行已有判断。<br>'+
      '凑满 632 行后跑 <code>python3 benchmark/language/review.py import</code>。</div>';
    return;
  }
  const it=items[at];
  document.getElementById("pos").textContent=(at+1)+" / "+items.length;
  document.getElementById("rowid").textContent=it.id+" · turn "+it.turn;
  document.getElementById("tag").textContent=it.town+" · "+it.functions;
  document.getElementById("bar").style.width=(at/items.length*100)+"%";

  const auto=it.auto!=="ok"?'<div class="row"><span>auto</span><code>'+
    esc(it.auto)+'  — 这是被拒的那次改写的理由,下面的句子已回退成模板句</code></div>':"";
  const sug=it.suggest?'<div class="sug"><b>Claude 的建议:'+esc(it.suggest)+
    '</b><br>'+esc(it.why)+'</div>':"";
  const old=it.decision?'<div class="row"><span>已判</span><code>'+esc(it.decision)+
    (it.note?"  — "+esc(it.note):"")+'</code></div>':"";

  m.innerHTML=
   '<div class="card">'+
    '<div class="row"><span>路线</span><code>'+esc(it.route)+'</code></div>'+
    '<div class="row"><span>gold</span><code>'+esc(it.gold)+'</code></div>'+
    auto+old+
   '</div>'+
   '<div class="card">'+
    '<div class="row"><span>模板</span><div class="sent">'+it.t_diff+'</div></div>'+
    '<div class="row"><span>改写</span><div class="sent">'+it.p_diff+'</div></div>'+
    sug+
    '<div class="btns">'+
      '<button id="keep">保留改写 <kbd>k</kbd></button>'+
      '<button id="drop">退回模板 <kbd>d</kbd></button>'+
      '<button id="fix">自己改写 <kbd>f</kbd></button>'+
    '</div>'+
    '<div id="fixbox" class="hide">'+
      '<textarea id="fixtext" rows="2" placeholder="写一句忠于 gold 的话…">'+
        esc(it.fixed||it.paraphrase)+'</textarea>'+
      '<textarea id="note" rows="1" placeholder="备注(可空)">'+esc(it.note)+
      '</textarea>'+
      '<div class="btns"><button id="save">存下这句</button></div>'+
    '</div>'+
    '<div class="err" id="err"></div>'+
   '</div>'+
   '<div class="nav"><button id="prev">← 上一条</button>'+
     '<span>k 保留 · d 退回 · f 改写 · ←/→ 翻</span>'+
     '<button id="next">下一条 →</button></div>';

  document.getElementById("keep").onclick=()=>send("ok");
  document.getElementById("drop").onclick=()=>send("drop");
  document.getElementById("fix").onclick=()=>{
    const b=document.getElementById("fixbox");
    b.classList.toggle("hide");
    document.getElementById("fix").classList.toggle("on");
    if(!b.classList.contains("hide")) document.getElementById("fixtext").focus();
  };
  document.getElementById("save").onclick=()=>send("fix");
  document.getElementById("prev").onclick=()=>{if(at>0){at--;render();}};
  document.getElementById("next").onclick=()=>{at++;render();};
}

async function send(decision){
  const it=items[at];
  const fx=document.getElementById("fixtext");
  const nt=document.getElementById("note");
  const body={row:it.row,decision:decision,
              fixed:fx?fx.value:"",
              note:(nt&&!document.getElementById("fixbox").classList.contains("hide"))
                   ?nt.value:it.note};
  const btn=document.getElementById(decision==="ok"?"keep":
            decision==="drop"?"drop":"fix");
  if(btn) btn.classList.add("on");
  const r=await fetch("/api/decide",{method:"POST",
    headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const d=await r.json();
  if(!r.ok){document.getElementById("err").textContent=d.error||"写入失败";
            if(btn) btn.classList.remove("on"); return;}
  it.decision=decision; state.decided=d.decided;
  setTimeout(()=>{at++;render();},110);
}

addEventListener("keydown",e=>{
  if(e.target.tagName==="TEXTAREA"){
    if(e.key==="Enter"&&(e.metaKey||e.ctrlKey)) send("fix");
    return;
  }
  if(e.key==="k") send("ok");
  else if(e.key==="d") send("drop");
  else if(e.key==="f") document.getElementById("fix")?.click();
  else if(e.key==="ArrowRight"){at++;render();}
  else if(e.key==="ArrowLeft"&&at>0){at--;render();}
});

boot();
</script></body></html>
"""


# --------------------------------------------------------------------------- main

def free_port(preferred: int) -> int:
    for p in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return s.getsockname()[1]
            except OSError:
                continue
    raise SystemExit("[review-app] no free port")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--queue", default=str(QUEUE),
                    help="row,id,turn,suggest,why - shown as a suggestion only")
    ap.add_argument("--all", action="store_true", help="every turn, not just the flagged")
    ap.add_argument("--redo", action="store_true", help="include turns already decided")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"[review-app] {path} is missing - run review.py export first")
        return 1

    queue = load_queue(Path(args.queue))
    which = "all" if args.all else "flagged"
    if which == "flagged" and not queue:
        print(f"[review-app] {args.queue} is missing; use --all to walk every turn")
        return 1

    app = App(path, queue, which, args.redo)
    if not app.order:
        print("[review-app] nothing left in this queue "
              "(--all for every turn, --redo to revisit decided ones)")
        return 0

    Handler.app = app
    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}/"
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[review-app] {len(app.order)} turns in this queue ({which})")
    print(f"[review-app] {url}   (ctrl-c to stop; every click writes {path.name})")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[review-app] stopped")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
