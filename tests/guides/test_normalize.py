from src.guides.normalize import normalize_page, section_links

PAGE = """<!doctype html><html><head><title>T</title></head><body>
<header class="hero" id="hero"><h1>Hi</h1></header>
<main>
<section id="thesis"><h2>Your judge is an <em>instrument</em>: why</h2><p id="thesis-p1">x</p></section>
<section id="pipeline"><h2>The pipeline — eight stages</h2></section>
<section id="sources"><h3>no h2 here</h3></section>
</main>
</body></html>"""


def test_normalize_adds_wrap_nav_and_bridge():
    out = normalize_page(PAGE, title="LLM-as-a-Judge")
    assert '<header class="hero wrap" id="hero">' in out
    assert '<main class="wrap">' in out
    assert out.index('<nav class="site"') < out.index("<header")
    assert '<a href="#thesis">Your judge is an</a>' in out
    assert '<a href="#pipeline">The pipeline</a>' in out
    assert "#sources" not in out  # no h2 → no nav entry
    assert '<script src="/guides/guide-bridge.js"></script>\n</body>' in out
    assert normalize_page(out, title="LLM-as-a-Judge") == out  # idempotent


def test_normalize_leaves_a_complete_page_alone():
    page = (
        PAGE.replace(
            '<header class="hero" id="hero">',
            '<nav class="site"><div class="bar"></div></nav><header class="hero wrap" id="top">',
        )
        .replace("<main>", '<main class="wrap">')
        .replace("</body>", '<script src="/guides/guide-bridge.js"></script></body>')
    )
    out = normalize_page(page, title="T")
    assert out.count('<nav class="site"') == 1 and out.count("guide-bridge.js") == 1
    assert section_links(page)[0] == ("thesis", "Your judge is an")
