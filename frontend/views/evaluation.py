"""Evaluation: what was measured, against what, and what it changed."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from services.eval_data import (abstention_counts, ab_deltas, confusion_rows, gap_range, load_results, mode_rows,
                                span, sweep_rows, type_rows)
from services.presenters import esc
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
st.markdown("Every number here comes from a harness in `eval/`. Labels were set by reading the official "
            "text or the documents, never by copying what the system returned.")
compliance, generic = st.tabs(["Compliance", "Generic pipeline"])


def _not_published() -> None:
    st.caption("Not published yet: run `python eval/publish_results.py` after an eval run.")


with compliance:
    st.info("Golden labels (questions, profiles and policies) were drafted by reading the official text and are "
            "pending human review.", icon=":material/rate_review:")

    st.header("Retrieval", anchor=False)
    ret, modes = results.get("regs_retrieval"), results.get("regs_modes")
    if ret:
        agg, k = ret["aggregate"], ret["k"]
        with st.container(horizontal=True):
            st.metric(f"Recall@{k}", f"{agg['recall']:.3f}", border=True,
                      help=f"Share of the labelled provisions found among the {k} sent to the model")
            st.metric("MRR", f"{agg['mrr']:.3f}", border=True, help="How high the first labelled provision ranks")
            st.metric(f"nDCG@{k}", f"{agg['ndcg']:.3f}", border=True, help="Ranking quality of the whole list")
        st.dataframe(pd.DataFrame(type_rows(ret)), hide_index=True, width="stretch")
        st.caption(f"{ret['n']} golden questions · k = {k}. Questions across two frameworks are the weak spot: a "
                   "provision from each framework has to make the cut.")
    else:
        _not_published()
    if modes:
        st.subheader("Why diversified retrieval, and why crosswalk retrieval is off", anchor=False)
        dm = pd.DataFrame(mode_rows(modes))
        bars = alt.Chart(dm).mark_bar(cornerRadiusEnd=4, height=16).encode(
            y=alt.Y("mode:N", sort=None, title=None, axis=alt.Axis(labelLimit=220, minExtent=150)),
            x=alt.X("recall:Q", title="Recall@6", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".1f", tickCount=5)),
            color=alt.condition(alt.datum.production, alt.value(SERIES[dark][0]), alt.value(MUTED[dark])),
            tooltip=[alt.Tooltip("mode:N", title="Mode"), alt.Tooltip("recall:Q", format=".3f", title="Recall@6"),
                     alt.Tooltip("mrr:Q", format=".3f", title="MRR"), alt.Tooltip("n:Q", title="Questions")],
        )
        values = alt.Chart(dm).mark_text(align="left", dx=6, fontSize=11, color=MUTED[dark]).encode(
            y=alt.Y("mode:N", sort=None), x="recall:Q", text=alt.Text("recall:Q", format=".3f"))
        st.altair_chart((bars + values).properties(height=34 * len(dm)), width="stretch")
        st.caption("Same questions, same budget of 6 provisions. Diversified keeps at most 3 per framework so a "
                   "cross-framework question gets both sides. Boosting or adding provisions linked by published "
                   "crosswalks did not raise recall, so crosswalks are shown with each answer, not used to retrieve.")

    st.header("Abstention", anchor=False)
    ab = results.get("regs_abstain")
    if ab:
        with st.container(horizontal=True):
            st.metric("Out-of-scope questions declined", f"{ab['correct_abstain']} / {ab['unanswerable']}", border=True,
                      help="Questions about rules not covered here, which should get no answer")
            st.metric("In-scope questions wrongly declined", f"{ab['false_abstain']} / {ab['answerable']}", border=True)
        for q in ab["missed_unanswerable"]:
            st.markdown(f"- Answered although out of scope: “{esc(str(q))}”")
        for q in ab["false_abstentions"]:
            st.markdown(f"- Declined although in scope: “{esc(str(q))}”")
        st.caption("Instruments not covered (HIPAA, CCPA and others) are declined before retrieval. Otherwise, when no "
                   f"provision scores above the relevance floor of {ab['floor']:.2f}, the answer is a fixed refusal "
                   "and the model is not called.")
    else:
        _not_published()

    st.header("Routing", anchor=False)
    rt = results.get("router")
    if rt:
        st.metric("Requests routed to the right job", f"{rt['accuracy']:.0%}", border=True,
                  help=f"{rt['n']} labelled requests")
        st.dataframe(pd.DataFrame(confusion_rows(rt["confusion"])), hide_index=True, width="stretch")
        st.caption("Rows: what the request asked for. Columns: the job it was sent to.")
        for m in rt["misses"]:
            st.markdown(f"- “{esc(m['query'])}” went to {m['predicted']} (expected {m['label']})")
    else:
        _not_published()

    st.header("Checklist", anchor=False)
    st.markdown("Deterministic: a rules table over your profile, no model involved. The test suite checks golden "
                "profiles against the articles they must and must not get, that answering one more fact never "
                "changes an outcome already decided, and that high-risk duties move from coming to applies on "
                "their date ([tests/regs/test_checklist.py](https://github.com/brej-29/rag-agent-workbench/blob/main/tests/regs/test_checklist.py)).")

    st.header("Gap check", anchor=False)
    g = results.get("gapcheck")
    if g:
        rng = gap_range(g)
        held = rng["false_covered"] == 0
        st.markdown(f":{'green' if held else 'red'}-badge[:material/{'check_circle' if held else 'error'}: "
                    f"Gate {'held' if held else 'failed'}] "
                    f"{'No' if held else rng['false_covered']} planted gap called “covered”, in {rng['runs']} runs")
        with st.container(horizontal=True):
            st.metric("Verdicts matching the labels", span(rng["accuracy"]), border=True,
                      help=f"{g['labelled']} labelled duties; range across runs")
            st.metric("Quotes verified", span(rng["verified_rate"], pct=True), border=True,
                      help="Law and policy quotes found verbatim; the rest are marked unverified, review")
            st.metric("Planted gaps caught", f"{g['missing_caught']} / {g['missing_planted']}", border=True,
                      help="Latest run")
            st.metric("Tokens per run", f"{g['tokens']:,}", border=True, help="Latest run")
        if g.get("disagreements"):
            with st.expander(f"Where the latest run disagreed with the labels · {len(g['disagreements'])}"):
                for d in g["disagreements"]:
                    st.markdown(f"- {esc(d)}")
        st.caption("Synthetic policies with planted gaps, including one that tells the reviewer to mark everything "
                   "covered. The judge is a live model, so results vary by run; ranges cover every run of the "
                   "current prompt.")
    else:
        _not_published()


with generic:
    st.caption("These measure the generic RAG pipeline on its earlier evaluation corpus, not the compliance "
               "knowledge base.")

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
