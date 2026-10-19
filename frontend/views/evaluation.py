"""Evaluation: what was measured, against what, and what it changed."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from services.eval_data import abstention_counts, ab_deltas, load_results, sweep_rows
from services.shell import is_dark

# Validated with the dataviz palette checker (light: surface #F5F7F4, dark: #11161B).
# Orange is reserved for thresholds (the relevance floor, production k), never a series.
SERIES = {False: ["#1C8466", "#5B5FCF", "#B8457E"], True: ["#2A9C7A", "#7E83E2", "#D2669B"]}
THRESHOLD = {False: "#C9661E", True: "#E8913F"}
MUTED = {False: "#5E6B66", True: "#93A19B"}

dark = is_dark()
results = load_results()


def _date(stamp: str) -> str:
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"


st.title("Evaluation", anchor=False)
st.markdown(
    "Every number here comes from a harness in `eval/`, run against the live index. "
    "Relevance labels were set by reading the documents, never by copying what the retriever returned."
)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
retrieval, sweep = results.get("retrieval"), results.get("sweep")
if retrieval:
    st.header("Retrieval", anchor=False)
    agg = retrieval["aggregate"]
    with st.container(horizontal=True):
        st.metric("Recall@10", f"{agg['mean_recall_at_k']:.3f}", border=True)
        st.metric("MRR", f"{agg['mean_mrr']:.3f}", border=True)
        st.metric("nDCG@10", f"{agg['mean_ndcg_at_k']:.3f}", border=True)
    st.caption(f"{retrieval['total_queries']} questions · namespace `{retrieval['namespace']}` · "
               f"run {_date(retrieval['timestamp'])}")

if sweep:
    st.subheader("How many passages to send: the k sweep", anchor=False)
    df = pd.DataFrame(sweep_rows(sweep))
    order = ["Recall@k", "nDCG@k", "Precision@k"]
    color = alt.Color("metric:N", sort=order, scale=alt.Scale(domain=order, range=SERIES[dark]),
                      legend=alt.Legend(orient="top", title=None))
    x = alt.X("k:Q", title="Passages retrieved (k)", scale=alt.Scale(domain=[1, 12.2]),
              axis=alt.Axis(values=sorted(df.k.unique().tolist()), format="d", grid=False))
    y = alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".1f", tickCount=5))
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["k"], empty=False)
    base = alt.Chart(df).encode(x=x, y=y, color=color)
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(filled=True, size=64).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("metric:N"), alt.Tooltip("k:Q"), alt.Tooltip("value:Q", format=".3f")],
    ).add_params(nearest)
    crosshair = alt.Chart(df).mark_rule(color=MUTED[dark], strokeWidth=1).encode(x="k:Q").transform_filter(nearest)
    # Direct labels at the line ends, nudged apart where series converge (Recall/nDCG at k=10).
    ends = df[df.k == df.k.max()].sort_values("value", ascending=False).reset_index(drop=True)
    ends["label_y"] = [v + 0.035 * (1 - i) for i, v in enumerate(ends.value)]
    labels = alt.Chart(ends).mark_text(align="left", dx=8, fontSize=12).encode(
        x="k:Q", y=alt.Y("label_y:Q", scale=alt.Scale(domain=[0, 1])), text="metric:N", color=color)
    rules = pd.DataFrame([{"k": 5, "note": "production k = 5"}, {"k": int(sweep.get("knee_k", 8)), "note": "knee"}])
    rule = alt.Chart(rules).mark_rule(strokeDash=[4, 3], strokeWidth=1.5).encode(
        x="k:Q", color=alt.value(THRESHOLD[dark]))
    rule_text = alt.Chart(rules).mark_text(align="left", dx=4, dy=-6, fontSize=11).encode(
        x="k:Q", y=alt.value(0), text="note:N", color=alt.value(THRESHOLD[dark]))
    st.altair_chart((lines + points + crosshair + labels + rule + rule_text).properties(height=300, padding={"left": 12}),
                    width="stretch")
    a5, a8 = sweep["aggregate"]["5"], sweep["aggregate"][str(sweep.get("knee_k", 8))]
    st.markdown(
        f"Recall keeps rising after k = 5, but precision falls from `{a5['mean_precision_at_k']:.2f}` to "
        f"`{a8['mean_precision_at_k']:.2f}` by the knee. Production sends **5** passages: fewer, higher-signal "
        "passages for the model to read."
    )

# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
ab = results.get("rerank_ab")
if ab:
    st.header("Reranker: measured, then switched off", anchor=False)
    deltas, latency = ab_deltas(ab)
    dd = pd.DataFrame(deltas)
    dd["effect"] = dd.delta.map(lambda d: "better" if d > 0 else ("worse" if d < 0 else "no change"))
    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        bars = alt.Chart(dd).mark_bar(cornerRadiusEnd=4, height=14).encode(
            y=alt.Y("metric:N", sort=None, title=None, axis=alt.Axis(labelLimit=140, minExtent=90)),
            x=alt.X("delta:Q", title=f"{ab['rerank_model']} minus baseline",
                    axis=alt.Axis(format="+.2f", tickCount=4), scale=alt.Scale(domainMax=0.02)),
            color=alt.Color("effect:N", legend=None, scale=alt.Scale(
                domain=["better", "no change", "worse"], range=[SERIES[dark][0], MUTED[dark], SERIES[dark][2]])),
            tooltip=[alt.Tooltip("metric:N"), alt.Tooltip("delta:Q", format="+.3f"), alt.Tooltip("effect:N")],
        )
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED[dark]).encode(x="x:Q")
        values = alt.Chart(dd).mark_text(align="left", dx=6, fontSize=11, color=MUTED[dark]).encode(
            y=alt.Y("metric:N", sort=None), x=alt.datum(0), text=alt.Text("delta:Q", format="+.3f"))
        st.altair_chart((bars + zero + values).properties(height=34 * len(dd)), width="stretch")
    with right:
        st.metric("Added latency", f"+{latency} ms", border=True, help="Per query, same run")
    st.caption(f"{ab['total_queries']} questions · {ab['candidates']} candidates reranked to top {ab['top_k']} · "
               f"run {_date(ab['timestamp'])}. Every quality metric got worse or stayed flat, so reranking is off.")

# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------
answers = results.get("answers")
if answers:
    st.header("Answers and abstention", anchor=False)
    s, counts = answers["summary"], abstention_counts(answers["rows"])
    if counts["errors"]:
        st.warning(
            f"Partial run: {counts['errors']} of {len(answers['rows'])} questions hit the free-tier daily "
            "token limit and are not scored. Reference claims are drafts pending human review.",
            icon=":material/hourglass_bottom:",
        )
    with st.container(horizontal=True):
        st.metric("Claim recall", f"{s['mean_claim_recall']:.2f}", border=True,
                  help="Share of reference facts the answer states")
        st.metric("Faithfulness", f"{s['mean_faithfulness']:.2f}", border=True,
                  help="Share of the answer supported by the retrieved passages")
        st.metric("Citation precision", f"{s['mean_citation_precision']:.2f}", border=True,
                  help="Cited passages that come from the labelled evidence documents")
    st.markdown("**Did it answer when it should, and decline when it should?**")
    st.dataframe(pd.DataFrame(
        {"Answered": [counts["answered_ok"], counts["answered_unanswerable"]],
         "Declined": [counts["false_abstain"], counts["correct_abstain"]]},
        index=["In the knowledge base", "Not in the knowledge base"],
    ), width="stretch")
    cfg = answers["config"]
    st.caption(f"Generator `{cfg['generator']}`, judged by a different model, `{cfg['judge']}` · "
               f"run {_date(answers['timestamp'])}")

# ---------------------------------------------------------------------------
# Adversarial
# ---------------------------------------------------------------------------
adv = results.get("adversarial")
if adv:
    st.header("Attacks", anchor=False)
    rows = adv["rows"]
    structural = [r for r in rows if r["structural"]]
    model_dep = [r for r in rows if not r["structural"]]
    with st.container(horizontal=True):
        st.metric("Structural defences held", f"{sum(not r['attack_succeeded'] for r in structural)} / {len(structural)}",
                  border=True, help="Hold even if the model obeys every instruction; also checked in CI")
        st.metric("Attacks that worked", f"{sum(r['attack_succeeded'] for r in model_dep)} / {len(model_dep)}",
                  border=True, help="Depend on the model honouring its instructions; one live run")
    st.dataframe(pd.DataFrame([{
        "Attack": r["id"], "Kind": r["category"].replace("_", " "),
        "Result": "✓ Held" if not r["attack_succeeded"] else "✗ Breached",
        "Defence": "structural" if r["structural"] else "model",
        "LLM calls": r["llm_calls"],
    } for r in rows]), hide_index=True, width="stretch")
    st.caption(f"Model `{adv['summary']['model']}` · run {_date(adv['timestamp'])}. "
               "Success is detected with canary tokens, not judged by a model.")

# ---------------------------------------------------------------------------
# Per question
# ---------------------------------------------------------------------------
if retrieval:
    with st.expander("Retrieval, question by question"):
        st.dataframe(pd.DataFrame([{
            "Question": q["query"], "Recall@10": q["recall_at_k"], "MRR": q["mrr"], "nDCG@10": q["ndcg_at_k"],
        } for q in retrieval["per_query"]]), hide_index=True, width="stretch", column_config={
            m: st.column_config.ProgressColumn(m, min_value=0.0, max_value=1.0, format="%.2f")
            for m in ("Recall@10", "MRR", "nDCG@10")})

st.caption("Refresh after a run: `python eval/publish_results.py`")
