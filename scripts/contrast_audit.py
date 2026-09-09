import re, sys, itertools

css = open('frontend/src/theme.css').read()

def block(sel):
    m = re.search(re.escape(sel) + r"\s*\{(.*?)\n\}", css, re.S)
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{3,8});", m.group(1)))

dark = block(":root[data-theme='dark']")
light = block(":root[data-theme='light']")

def lum(hexs):
    h = hexs.lstrip('#')
    if len(h) == 3: h = ''.join(c*2 for c in h)
    r, g, b = (int(h[i:i+2], 16)/255 for i in (0, 2, 4))
    f = lambda c: c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4
    return 0.2126*f(r) + 0.7152*f(g) + 0.0722*f(b)

def ratio(a, b):
    la, lb = lum(a), lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)

# (foreground, background, minimum, what it is)
PAIRS = [
    ('text','bg',4.5,'body text on page'), ('text','panel',4.5,'body text on panel'),
    ('text2','bg',4.5,'secondary on page'), ('text2','panel',4.5,'secondary on panel'),
    ('text2','panel3',4.5,'secondary on panel3'),
    ('muted','bg',4.5,'muted on page'), ('muted','panel',4.5,'muted on panel'),
    ('muted','panel3',4.5,'muted on panel3'), ('muted','panel2',4.5,'muted on panel2'),
    ('dim','bg',3.0,'dim label on page'), ('dim','panel',3.0,'dim label on panel'),
    ('dim','panel3',3.0,'dim label on panel3'),
    ('accent','bg',3.0,'accent on page'), ('accent','panel',3.0,'accent on panel'),
    ('accent2','bg',4.5,'accent2 text on page'), ('accent2','panel',4.5,'accent2 text on panel'),
    ('accent2','accent-dim',4.5,'accent2 on accent-dim (nav .on, chips)'),
    ('accent2','panel2',4.5,'accent2 on panel2'),
    ('accent2','panel3',4.5,'accent2 on panel3'),
    ('on-accent','accent',4.5,'primary button label'),
    ('good','good-dim',4.5,'good badge'), ('warn','warn-dim',4.5,'warn badge'),
    ('bad','bad-dim',4.5,'bad badge'),
    ('good','panel',4.5,'good on panel'), ('bad','panel',4.5,'bad on panel'),
    ('warn','panel',4.5,'warn on panel'),
]

fails = 0
for name, tokens in (('dark', dark), ('light', light)):
    for fg, bg, minimum, what in PAIRS:
        if fg not in tokens or bg not in tokens:
            fails += 1
            print(f"  FAIL {name}: missing token {fg} or {bg} ({what})")
            continue
        r = ratio(tokens[fg], tokens[bg])
        ok = r >= minimum
        if not ok:
            fails += 1
            print(f"  FAIL {name:5} {fg:10} on {bg:12} {r:.2f} < {minimum}  ({what})")
print(f"\n{fails} failing pair(s) across both themes")
sys.exit(1 if fails else 0)
