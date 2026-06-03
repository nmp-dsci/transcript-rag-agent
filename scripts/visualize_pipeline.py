"""Generate a LangChain graph visualization of the RAG pipeline and save to static/.

Uses the native graph.draw_mermaid_png() from langchain_core.runnables.graph_mermaid,
which base64-encodes the Mermaid syntax and calls the mermaid.ink API to render a PNG.
"""

from __future__ import annotations

import pathlib

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.runnables.graph import MermaidDrawMethod

ROOT = pathlib.Path(__file__).parent.parent
STATIC = ROOT / "static"
STATIC.mkdir(exist_ok=True)

# ── Query path nodes ─────────────────────────────────────────────────────────
# name= sets the graph node label; with_config(run_name=) only affects tracing
embed_query = RunnableLambda(lambda x: x, name="embed_query")
retrieve_chunks = RunnableLambda(lambda x: x, name="ContextProvider")
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{rag_system_prompt}"),
        ("system", "{transcript_context}"),
        ("human", "{question}"),
    ]
)
first_pass_llm = RunnableLambda(lambda x: x, name="ChatOpenAI_first_pass")
parse_first = RunnableLambda(lambda x: x, name="FirstPassResult")

# ── Recursive expansion nodes ─────────────────────────────────────────────────
dedup_filter = RunnableLambda(lambda x: x, name="DedupFilter")
followup_retrieval = RunnableLambda(lambda x: x, name="FollowupRetrieval")
synthesis_llm = RunnableLambda(lambda x: x, name="ChatOpenAI_synthesis")

recursive_expansion = dedup_filter | followup_retrieval | synthesis_llm

answer_branch = RunnableBranch(
    (lambda x: isinstance(x, dict) and x.get("followups_requested"), recursive_expansion),
    RunnableLambda(lambda x: x, name="SingleHopAnswer"),
)

# ── Full query pipeline ───────────────────────────────────────────────────────
query_chain = embed_query | retrieve_chunks | rag_prompt | first_pass_llm | parse_first | answer_branch

# ── Generate graph and render via native draw_mermaid_png() ───────────────────
graph = query_chain.get_graph()

out_png = STATIC / "langchain_pipeline.png"
print("Rendering via graph.draw_mermaid_png() → mermaid.ink API ...")
graph.draw_mermaid_png(
    output_file_path=str(out_png),
    draw_method=MermaidDrawMethod.API,
    background_color="white",
)
print(f"Saved → {out_png}")
