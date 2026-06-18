"""Render the chat history as a self-contained WhatsApp-style HTML page.

The page embeds the conversation JSON and renders a left sidebar of asked
questions (with time and id) plus a main chat panel. Selecting a sidebar entry
shows the user's question as an outgoing bubble followed by one incoming bubble
per RAG setup, each headed by the setup name.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.chat.history import ChatEntry

DEFAULT_CHAT_HTML_PATH = Path("dashboard/chat.html")

_STYLE = """
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;font-family:'Segoe UI',Helvetica,Arial,sans-serif;color:#e9edef;
background:#0b141a;height:100vh;overflow:hidden}
.app{display:flex;height:100vh}
.sidebar{width:360px;min-width:300px;background:#111b21;border-right:1px solid #222d34;
display:flex;flex-direction:column}
.sidebar-head{padding:16px 18px;background:#202c33;border-bottom:1px solid #222d34}
.sidebar-head h1{margin:0;font-size:17px;color:#e9edef}
.sidebar-head p{margin:4px 0 0;font-size:12px;color:#8696a0}
.conv-list{overflow-y:auto;flex:1}
.conv-item{padding:12px 18px;border-bottom:1px solid #1d272d;cursor:pointer}
.conv-item:hover{background:#182229}
.conv-item.active{background:#2a3942}
.conv-q{font-size:14px;color:#e9edef;display:-webkit-box;-webkit-line-clamp:2;
-webkit-box-orient:vertical;overflow:hidden}
.conv-meta{display:flex;align-items:center;gap:8px;margin-top:6px;
font-size:11px;color:#8696a0}
.conv-badge{background:#005c4b;color:#d9fdd3;border-radius:9px;padding:1px 7px;font-size:11px}
.conv-id{font-family:ui-monospace,Menlo,monospace;color:#5b6b75}
.main{flex:1;display:flex;flex-direction:column;min-width:0;
background-color:#0b141a;
background-image:linear-gradient(rgba(11,20,26,.96),rgba(11,20,26,.96))}
.main-head{padding:14px 22px;background:#202c33;border-bottom:1px solid #222d34}
.main-head .q{font-size:15px;color:#e9edef;font-weight:600}
.main-head .sub{margin-top:4px;font-size:12px;color:#8696a0;
display:flex;gap:12px;flex-wrap:wrap}
.main-head .sub a{color:#53bdeb;text-decoration:none}
.chat{flex:1;overflow-y:auto;padding:24px 12%;display:flex;flex-direction:column;gap:14px}
.bubble{max-width:78%;padding:10px 12px;border-radius:10px;font-size:14px;
line-height:1.5;box-shadow:0 1px 1px rgba(0,0,0,.25)}
.out{align-self:flex-end;background:#005c4b;border-bottom-right-radius:2px}
.in{align-self:flex-start;background:#202c33;border-bottom-left-radius:2px;max-width:88%}
.in .setup{font-weight:700;color:#00a884;font-size:13px;margin-bottom:6px;
display:flex;align-items:center;gap:8px}
.in .setup .key{font-family:ui-monospace,Menlo,monospace;font-size:11px;
color:#8696a0;font-weight:400}
.body{word-wrap:break-word;font-size:14px;line-height:1.55}
.body p{margin:8px 0}
.body p:first-child{margin-top:0}
.body strong{color:#f3f6f8}
.body h3.ans-h,.body h4.ans-h,.body h5.ans-h{margin:14px 0 6px;color:#aee6d6;
font-weight:700;line-height:1.3;font-size:14px}
.body h3.ans-h{font-size:15px}
.body h3.ans-h:first-child,.body h4.ans-h:first-child{margin-top:0}
.body ol,.body ul{margin:6px 0;padding-left:22px}
.body li{margin:5px 0}
.ans-section{margin:10px 0}
.ans-section.summary{background:rgba(0,168,132,.10);border-left:3px solid #00a884;
padding:8px 13px 5px;border-radius:6px;margin:8px 0 14px}
.ans-section.summary .ans-h{margin-top:2px;color:#7ff0d4}
a.cite{display:inline-block;font-size:10px;line-height:1;vertical-align:super;
background:#0b3b33;color:#7ff0d4;border:1px solid #155f53;border-radius:4px;
padding:1px 4px;margin:0 1px;text-decoration:none}
a.cite:hover{background:#155f53;color:#d9fdd3}
.cite-missing{display:inline-block;font-size:10px;line-height:1;vertical-align:super;
background:#2a3942;color:#8696a0;border-radius:4px;padding:1px 4px;margin:0 1px}
.body.clamp{max-height:340px;overflow:hidden;
-webkit-mask-image:linear-gradient(180deg,#000 72%,transparent);
mask-image:linear-gradient(180deg,#000 72%,transparent)}
.body.clamp.expanded{max-height:none;-webkit-mask-image:none;mask-image:none}
.more{background:none;border:none;color:#53bdeb;cursor:pointer;
padding:6px 0 0;font-size:12px;font-weight:600}
.more:hover{text-decoration:underline}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.chip{background:#0b141a;border:1px solid #2a3942;color:#8696a0;
border-radius:8px;padding:2px 7px;font-size:11px;font-family:ui-monospace,Menlo,monospace}
.error{color:#f5a3a3}
details.refs{margin-top:10px;background:#0b141a;border:1px solid #2a3942;border-radius:8px;padding:6px 10px}
details.refs summary{cursor:pointer;color:#8696a0;font-size:12px}
details.refs ul{margin:8px 0 2px;padding-left:6px;list-style:none}
details.refs li{font-size:12px;margin:5px 0;display:flex;gap:8px;align-items:baseline}
details.refs a{color:#53bdeb;text-decoration:none}
details.refs a:hover{text-decoration:underline}
details.refs .rnum{font-family:ui-monospace,Menlo,monospace;color:#7ff0d4;flex:0 0 auto}
details.refs .vid{font-family:ui-monospace,Menlo,monospace;color:#5b6b75}
details.cmd{margin-top:6px}
details.cmd summary{cursor:pointer;color:#8696a0;font-size:11px}
details.cmd pre{margin:6px 0 0;background:#0b141a;border:1px solid #2a3942;
border-radius:6px;padding:8px;font-size:11px;color:#c7e1ff;overflow:auto;white-space:pre-wrap}
.empty{margin:auto;color:#8696a0;text-align:center;padding:40px}
"""

_SCRIPT = r"""
const DATA = __DATA__;

function escapeHtml(s){
  return String(s == null ? '' : s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
// The agent sometimes wraps its answer in a JSON payload (trailing, or inside a
// ```json fence). When present, that object's "answer" field is the canonical
// markdown answer, so prefer it and drop the surrounding prose/preamble/fences.
function extractJsonAnswer(t){
  const m = t.match(/\{\s*"(?:question|answer|references)"\s*:/);
  if(!m) return null;
  const start = m.index;
  let depth = 0, inStr = false, esc = false;
  for(let i = start; i < t.length; i++){
    const ch = t[i];
    if(inStr){
      if(esc) esc = false;
      else if(ch === '\\') esc = true;
      else if(ch === '"') inStr = false;
      continue;
    }
    if(ch === '"'){ inStr = true; continue; }
    if(ch === '{') depth++;
    else if(ch === '}'){
      depth--;
      if(depth === 0){
        try {
          const obj = JSON.parse(t.slice(start, i + 1));
          return obj && typeof obj.answer === 'string' ? obj.answer : null;
        } catch(e){ return null; }
      }
    }
  }
  return null;
}
// Strip agent noise: prefer the embedded JSON "answer", else drop a trailing
// payload, code fences, and a short meta preamble before the first heading.
function cleanAnswer(text){
  let t = String(text == null ? '' : text);
  const fromJson = extractJsonAnswer(t);
  if(fromJson != null && fromJson.trim()) return fromJson.trim();
  const jsonIdx = t.search(/\n\s*(?:```json\s*)?\{\s*"(question|answer|references)"\s*:/);
  if(jsonIdx !== -1) t = t.slice(0, jsonIdx);
  t = t.replace(/```[a-z]*\s*$/i, '').trim();
  t = t.replace(
    /^\s*[^#\n][^\n]*\n+(?=#|```)/,
    (m)=> /\b(evidence|comprehensive|here(?:'s| is)|based on|let me|i'?ll|i will|sufficient|i now have|compile|gathered)\b/i.test(m) ? '' : m
  );
  return t.replace(/```[a-z]*\s*$/i, '').trim();
}
function fmtSeconds(s){
  if(s == null) return '';
  s = Math.floor(s);
  return Math.floor(s/60)+':'+String(s%60).padStart(2,'0');
}
// Map citation number -> reference object, keyed by the digits in each label.
function buildRefMap(refs){
  const map = {};
  (refs||[]).forEach(r=>{
    const m = String(r.label||'').match(/\d+/);
    if(m) map[m[0]] = r;
  });
  return map;
}
// Inline formatting: escape, bold, and turn [n] citations into linked chips.
function inline(s, refMap){
  let out = escapeHtml(s).replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  out = out.replace(/\[(\d+)\]/g, (m,n)=>{
    const ref = refMap[n];
    if(ref){
      const url = escapeHtml(ref.timestamp_url || ref.source_url || '#');
      return '<a class="cite" href="'+url+'" target="_blank" title="Open source at timestamp">'+n+'</a>';
    }
    return '<span class="cite-missing">'+n+'</span>';
  });
  return out;
}
// Paragraphs and ordered/unordered lists (headings handled by parseSections).
function renderBlocks(text, refMap){
  const lines = String(text||'').split('\n');
  let html = '', list = null, para = [];
  const flushPara = ()=>{ if(para.length){ html += '<p>'+inline(para.join(' '),refMap)+'</p>'; para = []; } };
  const closeList = ()=>{ if(list){ html += '</'+list+'>'; list = null; } };
  for(const raw of lines){
    const line = raw.replace(/\s+$/,'');
    if(!line.trim()){ flushPara(); continue; }
    let m;
    if(m = line.match(/^\s*\d+\.\s+(.*)$/)){
      flushPara(); if(list!=='ol'){ closeList(); html+='<ol>'; list='ol'; }
      html += '<li>'+inline(m[1],refMap)+'</li>'; continue;
    }
    if(m = line.match(/^\s*[-*]\s+(.*)$/)){
      flushPara(); if(list!=='ul'){ closeList(); html+='<ul>'; list='ul'; }
      html += '<li>'+inline(m[1],refMap)+'</li>'; continue;
    }
    closeList(); para.push(line.trim());
  }
  flushPara(); closeList();
  return html;
}
function parseSections(text){
  const intro = [], sections = [];
  let cur = null;
  for(const line of String(text||'').split('\n')){
    const m = line.match(/^(#{1,6})\s+(.*)$/);
    if(m){ cur = {level:m[1].length, title:m[2], body:[]}; sections.push(cur); }
    else if(cur){ cur.body.push(line); }
    else { intro.push(line); }
  }
  return {intro: intro.join('\n'), sections};
}
function renderAnswer(text, refs){
  const refMap = buildRefMap(refs);
  const {intro, sections} = parseSections(text);
  let html = renderBlocks(intro, refMap);
  for(const s of sections){
    const highlight = /key findings|summary|tl;?dr|bottom line|takeaways?/i.test(s.title);
    const tag = 'h'+Math.min(s.level+1, 5);
    html += '<section class="ans-section'+(highlight?' summary':'')+'">'
      +'<'+tag+' class="ans-h">'+inline(s.title, refMap)+'</'+tag+'>'
      +renderBlocks(s.body.join('\n'), refMap)
      +'</section>';
  }
  return html;
}
function fmtTime(iso){
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString();
}
function sortedConversations(){
  return DATA.conversations.slice().sort((a,b)=> (a.asked_at < b.asked_at ? 1 : -1));
}
function metaChips(a){
  const chips = ['~'+a.token_estimate+' tok', a.chunk_count+' chunks'];
  if(a.llm_calls != null) chips.push(a.llm_calls+' LLM calls');
  if(a.iterations != null) chips.push(a.iterations+' iterations');
  if(a.elapsed_seconds) chips.push(a.elapsed_seconds+'s');
  if(a.terminated_reason) chips.push(escapeHtml(a.terminated_reason));
  return '<div class="chips">'+chips.map(c=>'<span class="chip">'+c+'</span>').join('')+'</div>';
}
function renderRefs(refs){
  if(!refs || !refs.length) return '';
  const items = refs.map(r=>{
    const url = escapeHtml(r.timestamp_url || r.source_url || '#');
    const label = escapeHtml(r.label || '[?]');
    const at = r.start_seconds != null ? ' at '+fmtSeconds(r.start_seconds) : '';
    const vid = escapeHtml(r.video_id || '');
    return '<li><span class="rnum">'+label+'</span>'
      +'<a href="'+url+'" target="_blank">open'+escapeHtml(at)+'</a>'
      +'<span class="vid">'+vid+'</span></li>';
  }).join('');
  return '<details class="refs"><summary>Sources ('+refs.length+')</summary><ul>'+items+'</ul></details>';
}
function toggleClamp(btn){
  const body = btn.previousElementSibling;
  const expanded = body.classList.toggle('expanded');
  btn.textContent = expanded ? 'Show less ▴' : 'Show full answer ▾';
}
function answerBubble(a){
  let inner = '<div class="setup">'+escapeHtml(a.title)+'<span class="key">'+escapeHtml(a.key)+'</span></div>';
  if(a.error){
    inner += '<div class="body error">Error: '+escapeHtml(a.error)+'</div>';
  } else {
    const cleaned = cleanAnswer(a.answer);
    const long = cleaned.length > 1400;
    inner += '<div class="body'+(long?' clamp':'')+'">'+renderAnswer(cleaned, a.references)+'</div>';
    if(long) inner += '<button class="more" onclick="toggleClamp(this)">Show full answer ▾</button>';
    inner += metaChips(a);
    inner += renderRefs(a.references);
  }
  inner += '<details class="cmd"><summary>command</summary><pre>'+escapeHtml(a.command)+'</pre></details>';
  return '<div class="bubble in">'+inner+'</div>';
}
function renderSidebar(){
  const list = document.getElementById('conv-list');
  const convs = sortedConversations();
  if(!convs.length){
    list.innerHTML = '<div class="empty">No questions yet.<br>Ask one from the CLI chat.</div>';
    return;
  }
  list.innerHTML = convs.map(c=>{
    return '<div class="conv-item" data-id="'+escapeHtml(c.id)+'" onclick="selectConversation(\''+c.id+'\')">'
      +'<div class="conv-q">'+escapeHtml(c.question)+'</div>'
      +'<div class="conv-meta"><span class="conv-badge">'+c.answers.length+' answer'+(c.answers.length===1?'':'s')+'</span>'
      +'<span>'+fmtTime(c.asked_at)+'</span><span class="conv-id">'+escapeHtml(c.id)+'</span></div>'
      +'</div>';
  }).join('');
}
function selectConversation(id){
  const conv = DATA.conversations.find(c=>c.id===id);
  if(!conv) return;
  document.querySelectorAll('.conv-item').forEach(el=>{
    el.classList.toggle('active', el.getAttribute('data-id')===id);
  });
  const head = document.getElementById('main-head');
  let sub = '<span>'+fmtTime(conv.asked_at)+'</span><span class="conv-id">'+escapeHtml(conv.id)+'</span>';
  if(conv.url) sub += '<a href="'+escapeHtml(conv.url)+'" target="_blank">restricted to one video</a>';
  head.innerHTML = '<div class="q">'+escapeHtml(conv.question)+'</div><div class="sub">'+sub+'</div>';
  const chat = document.getElementById('chat');
  let html = '<div class="bubble out"><div class="body">'+escapeHtml(conv.question)+'</div></div>';
  html += conv.answers.map(answerBubble).join('');
  chat.innerHTML = html;
  chat.scrollTop = 0;
}
function init(){
  renderSidebar();
  const convs = sortedConversations();
  if(convs.length){
    selectConversation(convs[0].id);
  } else {
    document.getElementById('chat').innerHTML =
      '<div class="empty">Select a question to view the conversation.</div>';
  }
}
init();
"""


def render_chat_html(
    entries: list[ChatEntry], generated_at: datetime | None = None
) -> str:
    moment = generated_at or datetime.now(timezone.utc)
    payload = {"conversations": [entry.to_dict() for entry in entries]}
    # Embed safely inside the inline <script> tag.
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    script = _SCRIPT.replace("__DATA__", data_json)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Transcript RAG Chat</title>",
            f"<style>{_STYLE}</style>",
            "</head>",
            "<body>",
            '<div class="app">',
            '<aside class="sidebar">',
            '<div class="sidebar-head">',
            "<h1>Transcript RAG Chat</h1>",
            f"<p>{len(entries)} question(s) — "
            f"generated {moment.strftime('%Y-%m-%d %H:%M UTC')}</p>",
            "</div>",
            '<div class="conv-list" id="conv-list"></div>',
            "</aside>",
            '<section class="main">',
            '<div class="main-head" id="main-head"></div>',
            '<div class="chat" id="chat"></div>',
            "</section>",
            "</div>",
            f"<script>{script}</script>",
            "</body>",
            "</html>",
        ]
    )


def write_chat_html(
    entries: list[ChatEntry], path: Path = DEFAULT_CHAT_HTML_PATH
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_chat_html(entries), encoding="utf-8")
    return path
