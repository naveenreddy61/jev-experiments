"""The labeling web page.

A small stdlib HTTP server with htmx for partial updates. No build step, no
database. Each click appends one row to `data/labels/<user>.jsonl`.

Run: `joke-pref label` and open http://127.0.0.1:8765.
"""

from __future__ import annotations

import html
import json
import threading
import urllib.parse
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from joke_pref.data import (
    LABELS,
    SKIP,
    Premise,
    append_label,
    load_labels,
    load_premises,
    validate_user,
)

HTMX = "https://cdn.jsdelivr.net/npm/htmx.org@2.0.4/dist/htmx.min.js"

CSS = """
:root{--bg:#f7f5f0;--card:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e6e2d8;
--bad:#c0392b;--good:#2e7d32;--great:#6a1b9a;--accent:#1d4ed8}
@media(prefers-color-scheme:dark){:root{--bg:#141416;--card:#1f1f23;--ink:#f2f2f2;--muted:#a0a0a8;--line:#33333a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:17px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:720px;margin:0 auto;padding:24px 16px 64px}
header{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:16px}
header h1{font-size:18px;margin:0;font-weight:600}header .meta{color:var(--muted);font-size:14px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:8px 0 20px}
.bar div{height:100%;background:var(--accent)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px 22px 14px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.setup{font-size:22px;font-weight:600;line-height:1.35;margin:0 0 6px;white-space:pre-line}
.tag{color:var(--muted);font-size:13px;margin-bottom:16px}
.pl{display:flex;gap:12px;align-items:flex-start;padding:14px 0;border-top:1px solid var(--line)}
.pl .txt{flex:1;white-space:pre-line}.pl .key{color:var(--muted);font-size:13px;min-width:18px;padding-top:3px}
.btns{display:flex;gap:6px;flex-shrink:0}
button{font:inherit;font-size:14px;padding:6px 12px;border-radius:999px;border:1px solid var(--line);background:transparent;color:var(--ink);cursor:pointer}
button:hover{border-color:var(--ink)}
button.on.bad{background:var(--bad);border-color:var(--bad);color:#fff}
button.on.good{background:var(--good);border-color:var(--good);color:#fff}
button.on.great{background:var(--great);border-color:var(--great);color:#fff}
.pl.done .txt{color:var(--muted)}
.foot{display:flex;justify-content:space-between;align-items:center;padding-top:14px;border-top:1px solid var(--line);margin-top:4px;flex-wrap:wrap;gap:8px}
.foot .hint{color:var(--muted);font-size:13px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.primary{background:var(--accent);border-color:var(--accent);color:#fff;padding:8px 18px}
input[type=text]{font:inherit;padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);width:240px}
table{width:100%;border-collapse:collapse;font-size:15px}td,th{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line)}
.count{color:var(--muted)}
"""

JS = """
document.addEventListener('keydown', function (e) {
  if (e.target.tagName === 'INPUT' || e.metaKey || e.ctrlKey || e.altKey) return;
  var card = document.getElementById('card'); if (!card) return;
  if (e.key === 'n' || e.key === 'Enter') { var nx = card.querySelector('[data-next]'); if (nx) { nx.click(); e.preventDefault(); } return; }
  var idx = {'1': 'bad', '2': 'good', '3': 'great'}[e.key]; if (!idx) return;
  var row = card.querySelector('.pl:not(.done)') || card.querySelector('.pl');
  if (!row) return;
  var btn = row.querySelector('button.' + idx); if (btn) { btn.click(); e.preventDefault(); }
});
"""


class LabelApp:
    def __init__(self, premises_path: str | Path, labels_dir: str | Path):
        self.premises_path = Path(premises_path)
        self.labels_dir = Path(labels_dir)
        self.premises: list[Premise] = load_premises(self.premises_path)
        self.by_id = {p.id: p for p in self.premises}
        self.total = sum(len(p.punchlines) for p in self.premises)
        self._lock = threading.Lock()

    # --- state -----------------------------------------------------------

    def labels(self, user: str) -> dict[str, str]:
        return load_labels(self.labels_dir, user)

    def next_premise(self, labels: dict[str, str], after: str | None = None) -> Premise | None:
        ids = [p.id for p in self.premises]
        start = ids.index(after) + 1 if after in self.by_id else 0
        order = self.premises[start:] + self.premises[:start]
        for p in order:
            if any(pl.id not in labels for pl in p.punchlines):
                return p
        return None

    def record(self, user: str, pl_id: str, label: str) -> None:
        premise_id = pl_id.rsplit(".", 1)[0]
        premise = self.by_id[premise_id]
        premise.punchline(pl_id)  # raises KeyError for an unknown id
        with self._lock:
            append_label(self.labels_dir, user, pl_id, label)

    # --- rendering -------------------------------------------------------

    def page(self, body: str, user: str | None, labels: dict[str, str] | None) -> str:
        done = len(labels) if labels else 0
        pct = 100.0 * done / self.total if self.total else 0
        who = f"{html.escape(user)} · <a href='/review'>review</a> · <a href='/logout'>switch</a>" if user else ""
        progress = (
            f"<div class='bar'><div style='width:{pct:.1f}%'></div></div>" if user else ""
        )
        meta = f"<span class='meta'>{done} / {self.total} punchlines · {who}</span>" if user else ""
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Joke labels</title>
<script src="{HTMX}"></script><style>{CSS}</style></head><body><main>
<header><h1>Joke labels</h1>{meta}</header>{progress}
{body}
<script>{JS}</script></main></body></html>"""

    def login(self) -> str:
        return """<div class="card"><p class="setup">Who is labeling?</p>
<p class="tag">Your labels go to <code>data/labels/&lt;name&gt;.jsonl</code> on this machine. They are not committed.</p>
<form method="post" action="/start"><input type="text" name="user" placeholder="first name" autofocus required pattern="[A-Za-z0-9][A-Za-z0-9_-]{0,31}">
<button class="primary" type="submit">Start</button></form>
<p class="hint" style="color:var(--muted);font-size:14px;margin-top:16px">For each punchline choose <b>bad</b>, <b>good</b> or <b>great</b>. Go with your gut. Keys <b>1</b> <b>2</b> <b>3</b> label the next unlabeled line; <b>n</b> moves on.</p></div>"""

    def card(self, premise: Premise | None, labels: dict[str, str], *, position: int | None = None) -> str:
        if premise is None:
            return """<div class="card" id="card"><p class="setup">All punchlines are labeled.</p>
<p class="tag">Thank you. You can <a href="/review">review</a> and change any label.</p></div>"""
        rows = []
        for i, pl in enumerate(premise.punchlines, 1):
            current = labels.get(pl.id)
            btns = "".join(
                f"<button class='{name}{' on' if current == name else ''}' "
                f"hx-post='/label' hx-vals='{json.dumps({'punchline_id': pl.id, 'label': name})}' "
                f"hx-target='#card' hx-swap='outerHTML'>{name}</button>"
                for name in LABELS
            )
            rows.append(
                f"<div class='pl{' done' if current else ''}'><span class='key'>{i}</span>"
                f"<div class='txt'>{html.escape(pl.text)}</div><div class='btns'>{btns}</div></div>"
            )
        all_done = all(pl.id in labels for pl in premise.punchlines)
        pos = f"premise {position} of {len(self.premises)} · " if position else ""
        nxt = (
            f"<button class='primary' data-next hx-get='/card?after={premise.id}' hx-target='#card' hx-swap='outerHTML'>Next</button>"
            if all_done
            else f"<button data-next hx-get='/card?after={premise.id}' hx-target='#card' hx-swap='outerHTML'>Skip for now</button>"
        )
        return f"""<div class="card" id="card">
<p class="setup">{html.escape(premise.premise)}</p>
<p class="tag">{pos}{html.escape(premise.domain)} · {html.escape(premise.format)}</p>
{''.join(rows)}
<div class="foot"><span class="hint">1 / 2 / 3 label the next line · n for next</span>{nxt}</div></div>"""

    def review(self, labels: dict[str, str]) -> str:
        rows = []
        for i, p in enumerate(self.premises, 1):
            done = sum(pl.id in labels for pl in p.punchlines)
            marks = " ".join(
                f"<span class='{labels[pl.id]}' style='color:var(--{labels[pl.id]})'>{labels[pl.id]}</span>" if pl.id in labels else "<span class='count'>·</span>"
                for pl in p.punchlines
            )
            rows.append(
                f"<tr><td class='count'>{i}</td><td><a href='/?p={p.id}'>{html.escape(p.premise[:90])}</a></td>"
                f"<td class='count'>{done}/{len(p.punchlines)}</td><td>{marks}</td></tr>"
            )
        counts = {name: sum(1 for v in labels.values() if v == name) for name in LABELS}
        summary = " · ".join(f"{k} {v}" for k, v in counts.items())
        return f"""<div class="card"><p class="tag">{summary} · <a href="/">back to labeling</a></p>
<table><tr><th></th><th>setup</th><th></th><th>labels</th></tr>{''.join(rows)}</table></div>"""


def make_handler(app: LabelApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "joke-pref/0.1"

        def log_message(self, fmt: str, *args) -> None:  # quiet
            return

        # --- helpers ---
        def _user(self) -> str | None:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            morsel = cookie.get("user")
            if not morsel:
                return None
            try:
                return validate_user(morsel.value)
            except ValueError:
                return None

        def _send(self, body: str, status: HTTPStatus = HTTPStatus.OK, headers: dict[str, str] | None = None) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _redirect(self, location: str, headers: dict[str, str] | None = None) -> None:
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", location)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _form(self) -> dict[str, str]:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n).decode("utf-8") if n else ""
            return {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}

        # --- routes ---
        def do_GET(self) -> None:
            url = urllib.parse.urlsplit(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
            user = self._user()
            if url.path == "/":
                if not user:
                    return self._send(app.page(app.login(), None, None))
                labels = app.labels(user)
                premise = app.by_id.get(q["p"]) if "p" in q else app.next_premise(labels)
                pos = [p.id for p in app.premises].index(premise.id) + 1 if premise else None
                return self._send(app.page(app.card(premise, labels, position=pos), user, labels))
            if url.path == "/card":
                if not user:
                    return self._send("", HTTPStatus.UNAUTHORIZED)
                labels = app.labels(user)
                premise = app.next_premise(labels, after=q.get("after"))
                pos = [p.id for p in app.premises].index(premise.id) + 1 if premise else None
                return self._send(app.card(premise, labels, position=pos))
            if url.path == "/review":
                if not user:
                    return self._redirect("/")
                labels = app.labels(user)
                return self._send(app.page(app.review(labels), user, labels))
            if url.path == "/logout":
                return self._redirect("/", {"Set-Cookie": "user=; Path=/; Max-Age=0"})
            if url.path == "/health":
                return self._send("ok")
            self._send("not found", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            url = urllib.parse.urlsplit(self.path)
            form = self._form()
            if url.path == "/start":
                try:
                    user = validate_user(form.get("user", ""))
                except ValueError as exc:
                    return self._send(app.page(f"<div class='card'><p>{html.escape(str(exc))}</p>{app.login()}</div>", None, None), HTTPStatus.BAD_REQUEST)
                return self._redirect("/", {"Set-Cookie": f"user={user}; Path=/; Max-Age=31536000; SameSite=Lax"})
            if url.path == "/label":
                user = self._user()
                if not user:
                    return self._send("", HTTPStatus.UNAUTHORIZED)
                pl_id, label = form.get("punchline_id", ""), form.get("label", "")
                if label not in LABELS and label != SKIP:
                    return self._send("bad label", HTTPStatus.BAD_REQUEST)
                try:
                    app.record(user, pl_id, label)
                except KeyError:
                    return self._send("unknown punchline", HTTPStatus.BAD_REQUEST)
                labels = app.labels(user)
                premise = app.by_id[pl_id.rsplit(".", 1)[0]]
                pos = [p.id for p in app.premises].index(premise.id) + 1
                return self._send(app.card(premise, labels, position=pos))
            self._send("not found", HTTPStatus.NOT_FOUND)

    return Handler


def serve(premises_path: str | Path, labels_dir: str | Path, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    app = LabelApp(premises_path, labels_dir)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    return server
