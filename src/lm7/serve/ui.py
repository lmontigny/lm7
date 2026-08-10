"""A single-page chat client, served at ``/`` so `lm7 model serve` is testable.

Deliberately one string in one module rather than a static file: a `.html` asset
would need package-data configuration in `pyproject.toml` and a filesystem read
at request time, both of which are more machinery than a dev-facing page is
worth. There is no build step, no bundler, and no framework.

It is also deliberately **offline**. No CDN, no web font, no external stylesheet:
the whole point of `lm7 model serve` is a model running on the machine in front
of you, and a page that phones out to a CDN to render it fails on exactly the
airgapped box where local inference matters most.

The page holds no state the server does not: it reads `/health` and `/metrics`
for the header and drives `/v1/chat/completions` with SSE, which means it is
also a working demonstration of the endpoints rather than a privileged path
into the engine.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# The conversation lives in the page, not the server: LM7's engine is one static
# KV cache with no notion of a session, so every request resends the transcript
# exactly as an OpenAI client would. Clearing the chat is therefore genuinely
# clearing it -- there is nothing else holding the history.
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>lm7 serve</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #16181d; --muted: #6b7280; --line: #e5e7eb;
  --user: #eef2ff; --assistant: #f6f7f9; --accent: #4f46e5; --error: #b42318;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e6e8ec; --muted: #9aa3af; --line: #262b33;
    --user: #1d2333; --assistant: #171a21; --accent: #818cf8; --error: #f97066;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.55 ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  display: flex; flex-direction: column; height: 100vh;
}
header {
  border-bottom: 1px solid var(--line); padding: 10px 16px;
  display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
}
header b { font-size: 14px; letter-spacing: .02em; }
header .meta { color: var(--muted); font-size: 12.5px; }
header .meta code {
  font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--assistant); padding: 1px 5px; border-radius: 4px;
}
header .spacer { flex: 1; }
button {
  font: inherit; font-size: 13px; color: var(--fg); background: transparent;
  border: 1px solid var(--line); border-radius: 6px; padding: 4px 10px; cursor: pointer;
}
button:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
button:disabled { opacity: .45; cursor: default; }
main { flex: 1; overflow-y: auto; padding: 18px 16px; }
.wrap { max-width: 760px; margin: 0 auto; display: flex; flex-direction: column; gap: 12px; }
.msg { padding: 10px 13px; border-radius: 10px; white-space: pre-wrap; word-wrap: break-word; }
.msg.user { background: var(--user); align-self: flex-end; max-width: 82%; }
.msg.assistant { background: var(--assistant); }
.msg .who {
  display: block; font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 4px;
}
.msg.error { background: transparent; border: 1px solid var(--error); color: var(--error); }
.hint { color: var(--muted); font-size: 13.5px; text-align: center; margin-top: 8vh; }
.hint code {
  font: 12.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.cursor::after {
  content: "\\258e"; color: var(--accent);
  animation: blink 1s steps(2, start) infinite;
}
@keyframes blink { to { visibility: hidden; } }
footer { border-top: 1px solid var(--line); padding: 12px 16px; }
.status {
  max-width: 760px; margin: 0 auto 8px; min-height: 17px;
  color: var(--muted); font-size: 12.5px;
  display: flex; align-items: center; gap: 7px;
}
.status .dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--accent); flex: none;
}
.status.busy .dot { animation: pulse 1.1s ease-in-out infinite; }
.status.idle .dot { background: var(--muted); opacity: .5; }
.status.warn { color: var(--error); }
.status.warn .dot { background: var(--error); animation: none; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .25; } }
form { max-width: 760px; margin: 0 auto; display: flex; gap: 8px; align-items: flex-end; }
textarea {
  flex: 1; resize: none; font: inherit; color: var(--fg); background: var(--bg);
  border: 1px solid var(--line); border-radius: 8px; padding: 9px 11px; max-height: 40vh;
}
textarea:focus { outline: none; border-color: var(--accent); }
form button[type=submit] { padding: 9px 16px; }
</style>
</head>
<body>
<header>
  <b>lm7 serve</b>
  <span class="meta" id="meta">connecting&hellip;</span>
  <span class="spacer"></span>
  <button id="clear" type="button">Clear</button>
</header>
<main><div class="wrap" id="log">
  <p class="hint">
    One model, one static KV cache, one request at a time.<br>
    The first message compiles the graphs and is slower than the rest.<br>
    <code>POST /v1/chat/completions</code> &middot; <code>/docs</code> for the schema
  </p>
</div></main>
<footer>
  <div class="status idle" id="status"><span class="dot"></span><span id="statustext"></span></div>
  <form id="form">
    <textarea id="input" rows="1" placeholder="Message the model…" autofocus></textarea>
    <button type="submit" id="send">Send</button>
  </form>
</footer>

<script>
// Empty means "the server that sent this page", which is the case when LM7
// serves it at `/`. A base URL is substituted in when the page is served
// beside a different server -- `--ui-port` next to `--backend vllm`, where
// vLLM owns the API port and ships no page of its own.
const API = "__LM7_API_BASE__";

const log = document.getElementById("log");
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");
const meta = document.getElementById("meta");

const status = document.getElementById("status");
const statusText = document.getElementById("statustext");

// Sent verbatim on every turn. The server keeps no session, so this array is
// the entire conversation -- see the module docstring.
let messages = [];
let maxModelLen = 2048;
let busy = false;
let latest = null;
let modelId = "";

function setStatus(text, kind) {
  status.className = "status " + (kind || "idle");
  statusText.textContent = text;
}

async function json(path) {
  // Returns null rather than throwing, because half of these endpoints are
  // LM7's own and the page also runs against a plain vLLM server that has
  // /v1/models but answers /metrics in Prometheus text and /health with an
  // empty body.
  try {
    const response = await fetch(API + path);
    if (!response.ok) return null;
    return await response.json();
  } catch (err) {
    return null;
  }
}

async function refreshHeader() {
  const models = await json("/v1/models");
  if (!models) {
    meta.textContent = "server unreachable";
    if (!busy) setStatus("server unreachable", "warn");
    return;
  }
  // The one endpoint every OpenAI-compatible server has. Everything after this
  // is LM7's own and is treated as absent when it is.
  modelId = (models.data && models.data[0] && models.data[0].id) || "";
  const metrics = await json("/metrics");
  if (!metrics || metrics.max_model_len === undefined) {
    latest = null;
    meta.innerHTML = `<code>${escapeHtml(modelId)}</code> &middot; ${escapeHtml(API || "local")}`;
    if (!busy) setStatus("", "idle");
    return;
  }
  {
    latest = metrics;
    maxModelLen = metrics.max_model_len;
    const mib = (metrics.kv_cache_bytes / 1048576).toFixed(0);
    meta.innerHTML =
      `<code>${escapeHtml(metrics.model)}</code> &middot; ${escapeHtml(metrics.target)}` +
      ` &middot; backend ${escapeHtml(metrics.backend)}` +
      (metrics.dtype ? ` &middot; ${escapeHtml(metrics.dtype)}` : "") +
      ` &middot; ${maxModelLen} ctx &middot; kv ${mib} MiB` +
      (metrics.weights_bytes ? ` &middot; weights ${gib(metrics.weights_bytes)}` : "") +
      memoryLabel(metrics);
    if (!busy) {
      // A token that triggers a compile is the one regression the split into
      // separate prefill and decode graphs exists to prevent, so it outranks
      // every other thing this line could be saying.
      if (metrics.steady_frames > 0) {
        setStatus(
          `${metrics.steady_frames} compile(s) during decode — a token is` +
            ` triggering recompilation`,
          "warn"
        );
      } else if (!metrics.warm) {
        setStatus("cold — the first message compiles the graphs and will be slow", "idle");
      }
    }
  }
}

function gib(bytes) {
  // MiB below a gigabyte, because "0.2 GiB" for a 135M model reads as nothing.
  return bytes >= 1073741824
    ? `${(bytes / 1073741824).toFixed(2)} GiB`
    : `${(bytes / 1048576).toFixed(0)} MiB`;
}

function memoryLabel(metrics) {
  // Named for what it is, never just "memory". A device figure is this
  // process's allocator on the accelerator -- less than nvidia-smi shows,
  // which excludes the CUDA context and anything else on the card. A process
  // figure is RSS on a CPU target, which is the whole interpreter and mostly
  // PyTorch itself. Calling both "memory" would invite comparing them.
  if (!metrics.memory_bytes) return "";
  const used = gib(metrics.memory_bytes);
  if (metrics.memory_kind === "device") {
    const total = metrics.memory_total_bytes;
    return total ? ` &middot; gpu ${used} of ${gib(total)}` : ` &middot; gpu ${used}`;
  }
  return ` &middot; rss ${used}`;
}

function escapeHtml(text) {
  const node = document.createElement("div");
  node.textContent = text === undefined || text === null ? "" : String(text);
  return node.innerHTML;
}

function bubble(role, text) {
  const hint = log.querySelector(".hint");
  if (hint) hint.remove();
  const el = document.createElement("div");
  el.className = "msg " + role;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = role === "user" ? "you" : role === "error" ? "error" : "assistant";
  const body = document.createElement("span");
  body.textContent = text;
  el.append(who, body);
  log.append(el);
  scroll();
  return body;
}

function scroll() {
  const main = document.querySelector("main");
  main.scrollTop = main.scrollHeight;
}

function setBusy(state) {
  busy = state;
  send.disabled = state;
  send.textContent = state ? "…" : "Send";
}

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = input.scrollHeight + "px";
});

// Enter sends, Shift+Enter breaks the line -- the convention every chat client
// uses, and the reason the input is a textarea rather than an <input>.
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.getElementById("clear").addEventListener("click", () => {
  if (busy) return;
  messages = [];
  log.replaceChildren();
  bubble("assistant", "Cleared. The server held none of that — the page did.");
  refreshHeader();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (!content || busy) return;
  input.value = "";
  input.style.height = "auto";
  messages.push({ role: "user", content });
  bubble("user", content);

  setBusy(true);
  // A cold server is about to compile; a warm one is only running prefill.
  // Saying which is the difference between "slow" and "hung".
  const cold = !latest || !latest.warm;
  setStatus(cold ? "compiling prefill and decode graphs…" : "prefill…", "busy");

  const before = latest;
  const target = bubble("assistant", "");
  target.parentElement.classList.add("cursor");
  let answer = "";
  try {
    answer = await stream(target);
    messages.push({ role: "assistant", content: answer });
    await summarize(before);
  } catch (err) {
    target.parentElement.remove();
    bubble("error", String(err.message || err));
    // Dropped so a failed turn is not resent as context on the next one.
    messages.pop();
    setStatus("failed", "warn");
  } finally {
    target.parentElement.classList.remove("cursor");
    setBusy(false);
    input.focus();
  }
});

async function summarize(before) {
  // Token counts come from the server, which counts tokens; the page would be
  // counting SSE deltas, which are text fragments and not the same thing.
  await refreshHeader();
  if (!latest || !before) {
    // A server without LM7's /metrics -- vLLM, say -- can still be timed from
    // this side, and that is all this line claims.
    if (firstTokenMs !== null) {
      setStatus(`${Math.round(firstTokenMs)} ms to first token`, "idle");
    }
    return;
  }
  const tokens = latest.generated_tokens - before.generated_tokens;
  const parts = [`${tokens} tokens`];
  if (firstTokenMs !== null) parts.push(`${Math.round(firstTokenMs)} ms to first token`);
  if (decodeSeconds > 0 && tokens > 1) {
    parts.push(`${((tokens - 1) / decodeSeconds).toFixed(1)} tok/s`);
  }
  parts.push(`${latest.prefill_lengths} prefill graph(s)`);
  parts.push(latest.steady_frames === 0 ? "0 decode recompiles" : "RECOMPILED");
  setStatus(parts.join(" · "), latest.steady_frames === 0 ? "idle" : "warn");
}

let firstTokenMs = null;
let decodeSeconds = 0;

async function stream(target) {
  const response = await fetch(API + "/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages,
      stream: true,
      temperature: 0.7,
      // Optional for LM7, which holds one model, but required by vLLM -- and
      // this page is pointed at either. Taken from /v1/models so it is whatever
      // that server actually calls its model.
      model: modelId,
      // No max_tokens on purpose: the server fills whatever the static cache has
      // left after this prompt. The transcript is resent every turn and so grows
      // without bound, which means any number the page picked here would be
      // impossible once the conversation passed that share of the cache -- half
      // the cache was a 400 on every turn from the moment the transcript crossed
      // half the cache, which is a wall rather than a warning.
    }),
  });
  if (!response.ok) {
    // LM7's refusals say what to do about them, so show the server's own words.
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ? detailText(body.detail) : `HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const started = performance.now();
  let buffer = "";
  let answer = "";
  let firstAt = null;
  firstTokenMs = null;
  decodeSeconds = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE events are separated by a blank line; the last fragment may be a
    // partial event, so it stays in the buffer until its terminator arrives.
    const events = buffer.split("\\n\\n");
    buffer = events.pop();
    for (const event of events) {
      const line = event.trim();
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") continue;
      const delta = JSON.parse(payload).choices[0].delta.content;
      if (delta) {
        if (firstAt === null) {
          // Wall clock from the page, so it includes HTTP over loopback. It is
          // an indicator, not a benchmark -- see docs/serving.md.
          firstAt = performance.now();
          firstTokenMs = firstAt - started;
        }
        answer += delta;
        decodeSeconds = (performance.now() - firstAt) / 1000;
        target.textContent = answer;
        const rate = decodeSeconds > 0 ? ` · ${(answer.length / decodeSeconds).toFixed(0)} char/s` : "";
        setStatus(`generating${rate}`, "busy");
        scroll();
      }
    }
  }
  return answer;
}

function detailText(detail) {
  if (typeof detail === "string") return detail;
  // FastAPI's 422 body is a list of per-field errors rather than a sentence.
  if (Array.isArray(detail)) return detail.map((item) => item.msg || String(item)).join("; ");
  return JSON.stringify(detail);
}

refreshHeader();
</script>
</body>
</html>
"""

_API_PLACEHOLDER = "__LM7_API_BASE__"


def render(api_base: str = "") -> str:
    """The chat page, pointed at ``api_base`` or at whoever serves it.

    ``api_base`` is empty for LM7's own server, which serves the page from the
    same origin as the API. It is a full origin like ``http://127.0.0.1:8200``
    when the page sits beside a server that owns the API port and has no page --
    vLLM. The browser then talks to that server directly; nothing proxies.
    """
    return PAGE.replace(_API_PLACEHOLDER, api_base.rstrip("/"))


__all__ = ["PAGE", "render", "serve_page"]


def serve_page(port: int, api_base: str, host: str = "127.0.0.1") -> Any:
    """Serve the chat page, and nothing else, on ``port``.

    ``http.server`` from the standard library rather than the ``serve`` extra's
    FastAPI, because this hands out one string and needs no routing, no
    validation and no dependency -- and because the case it exists for is
    ``--backend vllm``, where LM7 has handed the model to another process and
    should not be starting a web framework behind it.

    Returned already started on a daemon thread. It answers ``/`` and 404s
    everything else: the API lives at ``api_base``, on a different server.
    """
    body = render(api_base).encode()

    class Handler(BaseHTTPRequestHandler):
        # Overridden to silence it. The default logs every request to stderr,
        # interleaved with the vLLM output this page sits next to.
        def log_message(self, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, name="lm7-ui", daemon=True).start()
    return server
