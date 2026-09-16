/* Reader bridge: the only script the workbench talks to inside a guide.
 *
 * The guide renders in a sandboxed iframe, so its origin is opaque and the
 * parent can only reach it by postMessage. Three messages in, two out:
 *
 *   parent → guide  { type: 'guide:theme',  theme: 'dark' | 'light' }
 *   parent → guide  { type: 'guide:scroll', id: 'pipeline-p3' }
 *   parent → guide  { type: 'guide:highlight', id: 'pipeline-p3' }
 *   guide  → parent { type: 'guide:ready', title, sections: [{ id, label }] }
 *   guide  → parent { type: 'guide:selection', quote, anchor, sectionId, rect }
 *
 * Nothing here reads the page's content beyond what the reader selected, and
 * nothing here fetches. Plain script, no build step, so a guide opened from
 * disk still works (the messages just have nobody to talk to).
 */
(function () {
  'use strict';
  var parentWin = window.parent !== window ? window.parent : null;

  function post(message) {
    if (parentWin) parentWin.postMessage(message, '*');
  }

  function applyTheme(theme) {
    if (theme === 'dark' || theme === 'light') {
      document.documentElement.setAttribute('data-theme', theme);
    }
  }

  function nearestAnchor(node) {
    var el = node && node.nodeType === 1 ? node : node && node.parentElement;
    while (el && el !== document.body) {
      if (el.id) return el.id;
      el = el.parentElement;
    }
    return null;
  }

  function sectionOf(node) {
    var el = node && node.nodeType === 1 ? node : node && node.parentElement;
    while (el && el !== document.body) {
      if (el.tagName === 'SECTION' && el.id) return el.id;
      el = el.parentElement;
    }
    return null;
  }

  function sections() {
    var out = [];
    var nodes = document.querySelectorAll('main section[id], section[id]');
    for (var i = 0; i < nodes.length; i++) {
      var s = nodes[i];
      var h = s.querySelector('h2, h3');
      out.push({ id: s.id, label: h ? h.textContent.replace(/\s+/g, ' ').trim() : s.id });
    }
    return out;
  }

  window.addEventListener('message', function (event) {
    var data = event.data || {};
    if (data.type === 'guide:theme') applyTheme(data.theme);
    if (data.type === 'guide:scroll' && data.id) {
      var target = document.getElementById(data.id);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    if (data.type === 'guide:highlight' && data.id) {
      var prev = document.querySelectorAll('.guide-hl');
      for (var i = 0; i < prev.length; i++) prev[i].classList.remove('guide-hl');
      var el = document.getElementById(data.id);
      if (el) el.classList.add('guide-hl');
    }
  });

  var pending = null;
  document.addEventListener('selectionchange', function () {
    if (pending) clearTimeout(pending);
    pending = setTimeout(function () {
      var sel = document.getSelection();
      var quote = sel ? sel.toString().replace(/\s+/g, ' ').trim() : '';
      if (!sel || sel.rangeCount === 0 || !quote) {
        post({ type: 'guide:selection', quote: '', anchor: null, sectionId: null, rect: null });
        return;
      }
      var range = sel.getRangeAt(0);
      var box = range.getBoundingClientRect();
      post({
        type: 'guide:selection',
        quote: quote.slice(0, 600),
        anchor: nearestAnchor(range.commonAncestorContainer),
        sectionId: sectionOf(range.commonAncestorContainer),
        rect: { top: box.top, left: box.left, width: box.width, height: box.height },
      });
    }, 150);
  });

  function ready() {
    post({ type: 'guide:ready', title: document.title, sections: sections() });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ready);
  } else {
    ready();
  }
})();
