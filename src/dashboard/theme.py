from __future__ import annotations


DARK_DASHBOARD_STYLE = [
    ":root{color-scheme:dark}",
    "body{font-family:Arial,sans-serif;line-height:1.45;margin:0;color:#dbe4ef;background:#0f141b}",
    "header{padding:20px 28px;background:#151c26;color:#f6f8fb;border-bottom:1px solid #2d3745}",
    "main{padding:24px 28px}",
    "a{color:#8cc8ff}",
    "table{border-collapse:collapse;width:100%;margin:16px 0;background:#151c26}",
    "th,td{border:1px solid #2d3745;padding:8px;text-align:left;vertical-align:top}",
    "th{background:#202a36;color:#f6f8fb;white-space:normal;word-break:break-word}",
    "td{overflow-wrap:anywhere}",
    "td.num{text-align:right;font-family:ui-monospace,Menlo,monospace}",
    "input,select{background:#10161f;border:1px solid #2d3745;color:#dbe4ef;padding:6px 8px}",
    "article{border-top:2px solid #2d3745;margin-top:28px;padding-top:16px}",
    "pre{white-space:pre-wrap;background:#10161f;border:1px solid #2d3745;color:#e7edf5;padding:12px;overflow:auto}",
    "details{margin:8px 0;padding:8px;border:1px solid #2d3745;background:#151c26}",
    "summary{cursor:pointer;font-weight:bold;color:#f6f8fb}",
    "code{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#c7e1ff}",
    ".metric{font-family:ui-monospace,Menlo,monospace}",
    ".metric-grid{display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:12px;margin-bottom:18px}",
    ".metric-grid .metric{background:#151c26;border:1px solid #2d3745;padding:12px}",
    ".metric strong{display:block;font-size:24px;color:#f6f8fb}",
    ".tabs{display:flex;gap:8px;margin-bottom:16px}",
    ".tab{border:1px solid #2d3745;background:#151c26;color:#dbe4ef;padding:8px 12px;cursor:pointer}",
    ".tab.active{background:#2f81f7;color:white;border-color:#2f81f7}",
    ".panel{display:none}.panel.active{display:block}",
    "td.summary{min-width:280px}",
    ".transcripts-table col:nth-child(1){width:10%}",
    ".transcripts-table col:nth-child(2){width:14%}",
    ".transcripts-table col:nth-child(3){width:14%}",
    ".transcripts-table col:nth-child(4){width:12%}",
    ".transcripts-table col:nth-child(5){width:14%}",
    ".transcripts-table col:nth-child(6){width:18%}",
    ".transcripts-table col:nth-child(7){width:8%}",
    ".transcripts-table col:nth-child(8){width:8%}",
    ".transcripts-table col:nth-child(9){width:6%}",
    ".transcripts-table col:nth-child(10){width:10%}",
    ".filters{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:12px 0}",
    ".scatter{width:100%;height:auto;background:#10161f;border:1px solid #2d3745}",
    ".failed{color:#ffb4a8}",
]


def dark_style_block() -> list[str]:
    return ["<style>", *DARK_DASHBOARD_STYLE, "</style>"]
