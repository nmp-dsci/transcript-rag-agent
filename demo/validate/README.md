# Frontend acceptance scripts

One script per vertical slice. Each drives a real browser through the click path
a reviewer would take at http://127.0.0.1:8021 and asserts on what the page then
shows — not on what the API returns.

```bash
uv run python -m src.cli serve --port 8021          # in one terminal
PYTHONPATH=. uv run --group demo python -m demo.validate.v0_judge   # in another
```

Each run writes `artifacts/<slice>/verdict.json` — the checks, their pass/fail,
and the screenshot each was taken against. Screenshots are gitignored; verdicts
are committed, because "the slice passed" is a claim that should survive in the
repo with its evidence.

## The rule these scripts exist to enforce

A slice whose hypothesis cannot be seen in the frontend fails review even if
every automated gate passed. So the script asserts the *visible* claim: the
metric table lists eight rows, the capped cell carries its reason, the theme
opens to chunks from two different videos.

## Who writes them

Not the agent that built the slice. Generation and evaluation run as separate
agents with separate contexts: the builder writes the feature and its unit
tests, an independent evaluator writes the script here and decides pass/fail.
An implementer validating their own work tests what they remember building.
