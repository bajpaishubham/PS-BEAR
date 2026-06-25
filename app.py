from __future__ import annotations

import streamlit as st

import descriptions
import searcher
import solver
import topics
from formulae_table_generator import load_formula_table

# Persisted (non-widget) session key for the physics problem, so it survives switching
# sections (Streamlit drops the state of widgets that aren't rendered on a given run).
PHYSICS_PROBLEM_KEY = "physics_problem"


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------
def _ranked_section(title: str, results: list[dict], kind: str, empty_hint: str) -> None:
    """Render a ranked list as clickable buttons plus a score bar chart. Clicking a button
    records the selection so the detail pane can show it; `kind` is 'formula', 'topic',
    or 'subtopic'."""
    st.markdown(f"**{title}**")
    if not results:
        st.caption(empty_hint)
        return
    for index, result in enumerate(results):
        label = result.get("label", "")
        score = float(result.get("score", 0.0))
        if st.button(f"{label}  ·  {score:.3f}", key=f"srch_{kind}_{index}"):
            st.session_state["srch_selection"] = {"kind": kind, "name": label}
    st.bar_chart(
        [{"item": r.get("label", ""), "score": round(float(r.get("score", 0.0)), 4)} for r in results],
        x="item",
        y="score",
        horizontal=True,
    )


def _latex_or_code(latex: str) -> None:
    try:
        st.latex(latex)
    except Exception:  # noqa: BLE001 - fall back to raw text if the LaTeX won't render
        st.code(latex)


def _detail_formula(name: str, rows: list[dict]) -> None:
    matches = [row for row in rows if row.get("name") == name]
    if not matches:
        st.warning(f"No stored row for formula “{name}” in the formula table.")
        return
    for row in matches:
        st.caption(f"{row.get('topic', '')} › {row.get('subtopic', '')}")
        if row.get("equation_latex"):
            _latex_or_code(row["equation_latex"])
        if row.get("formula_in_plain_english"):
            st.markdown(f"**In plain English:** {row['formula_in_plain_english']}")
        if row.get("description_from_image"):
            st.markdown(f"**From the image:** {row['description_from_image']}")
        if row.get("description_from_gemini"):
            st.markdown(f"**Description:** {row['description_from_gemini']}")
        if row.get("sympy_formula"):
            st.caption(f"Symbolic form: `{row['sympy_formula']}`")
        st.divider()


def _detail_topic(name: str, rows: list[dict], data: dict) -> None:
    description = data.get("topics", {}).get(name)
    st.write(description) if description else st.caption("No stored description for this topic yet.")
    topic_rows = [row for row in rows if row.get("topic") == name]
    subtopics = list(dict.fromkeys(row.get("subtopic", "") for row in topic_rows if row.get("subtopic")))
    if subtopics:
        st.markdown("**Subtopics**")
        for subtopic in subtopics:
            st.markdown(f"- {subtopic}")
    formula_names = list(dict.fromkeys(row.get("name", "") for row in topic_rows if row.get("name")))
    if formula_names:
        st.markdown(f"**Formulae ({len(formula_names)})**")
        for formula_name in formula_names:
            st.markdown(f"- {formula_name}")


def _detail_subtopic(name: str, rows: list[dict], data: dict) -> None:
    description = data.get("subtopics", {}).get(name)
    st.write(description) if description else st.caption("No stored description for this subtopic yet.")
    subtopic_rows = [row for row in rows if row.get("subtopic") == name]
    if not subtopic_rows:
        st.caption("No formulae stored for this subtopic.")
        return
    st.markdown(f"**Formulae ({len(subtopic_rows)})**")
    for row in subtopic_rows:
        st.markdown(f"**{row.get('name') or '(unnamed)'}**")
        if row.get("equation_latex"):
            _latex_or_code(row["equation_latex"])


def _render_detail_pane(rows: list[dict], data: dict) -> None:
    st.markdown("### Details")
    selection = st.session_state.get("srch_selection")
    if not selection:
        st.info("Click a formula, topic, or subtopic above to see its details here.")
        return
    kind, name = selection["kind"], selection["name"]
    st.caption({"formula": "Formula", "topic": "Topic", "subtopic": "Subtopic"}[kind])
    st.markdown(f"#### {name}")
    if kind == "formula":
        _detail_formula(name, rows)
    elif kind == "topic":
        _detail_topic(name, rows, data)
    else:
        _detail_subtopic(name, rows, data)


def render_searcher() -> None:
    st.header("Searcher")
    # value= reseeds from the persisted key when returning to this section (the widget's own
    # state is dropped while we are on another section); the mirror below keeps it in sync.
    query = st.text_area(
        "Enter the physics problem",
        value=st.session_state.get(PHYSICS_PROBLEM_KEY, ""),
        height=120,
        key="srch_query",
    )
    st.session_state[PHYSICS_PROBLEM_KEY] = query

    if st.button("Search"):
        with st.spinner("Ranking formulae, topics, and subtopics..."):
            st.session_state["srch_results"] = {
                "query": query,
                "formulae": searcher.search_formulae(query),
                "topics": searcher.search_topics(query),
                "subtopics": searcher.search_subtopics(query),
            }
        st.session_state.pop("srch_selection", None)  # clear the detail pane for a new search

    results = st.session_state.get("srch_results")
    if not results:
        st.info("Enter a physics problem and press Search.")
        return

    desc_hint = "No topic/subtopic descriptions available."
    formula_hint = "No formulae available."

    # Details are pulled live from the stored CSV (formula table) and JSON (descriptions).
    rows = load_formula_table()
    data = descriptions.load_descriptions()

    st.caption("Click an item to see its details in the pane below.")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        _ranked_section("A. Top formulae", results["formulae"], "formula", formula_hint)
    with col_b:
        _ranked_section("B. Top topics", results["topics"], "topic", desc_hint)
    with col_c:
        _ranked_section("C. Top subtopics", results["subtopics"], "subtopic", desc_hint)

    st.divider()
    _render_detail_pane(rows, data)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def _candidate_formulae(results: dict) -> list[dict]:
    all_rows = load_formula_table()
    source = st.radio(
        "Formula source",
        [
            "A formula from the search results",
            "All formulae of a relevant topic",
            "All formulae of a relevant subtopic",
        ],
        key="solv_source",
    )

    if source == "A formula from the search results":
        names = [r["name"] for r in results.get("formulae", [])]
        if not names:
            st.info("No ranked formulae available. Run the Searcher first.")
            return []
        chosen_name = st.selectbox("Formula (list A)", names, key="solv_A_name")
        return [row for row in all_rows if row["name"] == chosen_name]

    if source == "All formulae of a relevant topic":
        topic_labels = [r["label"] for r in results.get("topics", [])] or topics.topic_list()
        chosen_topic = st.selectbox("Topic (list B)", topic_labels, key="solv_B_topic")
        return [row for row in all_rows if row["topic"] == chosen_topic]

    subtopic_labels = [r["label"] for r in results.get("subtopics", [])]
    if not subtopic_labels:
        st.info("No ranked subtopics available. Run the Searcher first.")
        return []
    chosen_subtopic = st.selectbox("Subtopic (list C)", subtopic_labels, key="solv_C_subtopic")
    return [row for row in all_rows if row["subtopic"] == chosen_subtopic]


def render_solver() -> None:
    st.header("Solver")
    problem = st.session_state.get(PHYSICS_PROBLEM_KEY, "")
    if problem.strip():
        st.markdown("**Physics problem**")
        st.info(problem)
    else:
        st.caption("No physics problem entered yet — type one in the Searcher section.")

    results = st.session_state.get("srch_results", {})

    setup_col, solution_col = st.columns(2)

    # ----- Left column: pick a formula and the symbol to solve for -----
    with setup_col:
        candidates = _candidate_formulae(results)
        if not candidates:
            return

        name_to_row = {f"{row['name']}  [{row['subtopic']}]": row for row in candidates}
        chosen_label = st.selectbox("Choose a formula to solve", list(name_to_row.keys()), key="solv_formula")
        row = name_to_row[chosen_label]

        if row.get("equation_latex"):
            st.latex(row["equation_latex"])
        st.caption(f"Symbolic form: `{row.get('sympy_formula', '')}`")

        if not row.get("sympy_formula", "").strip():
            st.warning("This formula has no symbolic expression to solve.")
            return

        try:
            equation = solver.parse_sympy(row["sympy_formula"])
        except solver.FormulaParseError as exc:
            st.error(str(exc))
            return

        symbol_objects = solver.symbols_in(equation)
        if not symbol_objects:
            st.warning("No symbols found in this formula.")
            return
        symbol_names = [s.name for s in symbol_objects]

        target_name = st.selectbox("Solve for", symbol_names, key="solv_target")
        target = next(s for s in symbol_objects if s.name == target_name)

    # ----- Right column: analytical and numerical solutions -----
    with solution_col:
        try:
            solutions = solver.solve_analytical(equation, target)
        except solver.FormulaParseError as exc:
            st.error(str(exc))
            return
        if not solutions:
            st.warning(f"Could not isolate {target_name} in this formula.")
            return

        st.subheader("Analytical solution")
        for solution in solutions:
            st.latex(f"{target_name} = {solver.to_latex(solution)}")

        st.subheader("Numerical solution")
        if not solver.is_algebraic(equation):
            st.info("This formula contains a derivative, integral, or vector operation, so only the analytical form is shown.")
            return

        branch_index = 0
        if len(solutions) > 1:
            branch_index = st.selectbox(
                "Solution branch",
                list(range(len(solutions))),
                format_func=lambda i: f"{target_name} = {solutions[i]}",
                key="solv_branch",
            )
        solution_expr = solutions[branch_index]

        input_symbols = [s for s in symbol_objects if s.name != target_name]
        unit_options = solver.unit_names()
        value_unit_map = {}
        st.caption("Enter a value and unit for each input quantity. Values are converted to SI for the computation.")
        for symbol in input_symbols:
            vcol, ucol = st.columns([0.6, 0.4])
            with vcol:
                value = st.number_input(f"{symbol.name}", value=1.0, format="%.6g", key=f"solv_val_{symbol.name}")
            with ucol:
                unit = st.selectbox(f"Unit of {symbol.name}", unit_options, key=f"solv_unit_{symbol.name}")
            value_unit_map[symbol] = (value, unit)

        output_unit = st.selectbox("Result unit", unit_options, key="solv_output_unit")
        if st.button("Compute numerical result"):
            try:
                _, pretty = solver.solve_numerical(solution_expr, value_unit_map, output_unit)
                st.success(f"{target_name} = {pretty}")
            except solver.FormulaParseError as exc:
                st.error(str(exc))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="American Black PS-BEAR", layout="wide")
    st.title("American Black PS-BEAR")

    with st.sidebar:
        st.header("Sections")
        section = st.radio("Section", ["Searcher", "Solver"], key="active_section")

    if section == "Searcher":
        render_searcher()
    else:
        render_solver()


if __name__ == "__main__":
    main()
