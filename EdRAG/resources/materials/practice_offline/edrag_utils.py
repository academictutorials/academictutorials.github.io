"""Helper code for the EdRAG tutorial notebooks (ACM SIGCITE 2026).

The notebooks keep only the code that explains how RAG works. This file holds everything else:
downloading files, talking to the tutorial gateway, guesses, self-checks, and the scoring tables
and charts. You are welcome to read it, but you do not need to change it.

Both notebooks use this file, on the tutorial JupyterHub and on Google Colab.
"""
import json
import os
import pathlib
import re
import sys
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

# --------------------------------------------------------------------------------------------
# Where things are
# --------------------------------------------------------------------------------------------

MATERIALS_URL = "https://academictutorials.github.io/EdRAG/resources/materials/"
ON_COLAB = "google.colab" in sys.modules                 # running on Google Colab?
ON_HUB = "JUPYTERHUB_API_TOKEN" in os.environ              # running on the tutorial JupyterHub?

GATEWAY = "http://127.0.0.1:8000/v1"                       # the tutorial gateway (JupyterHub only)
HUB_TOKEN = os.environ.get("JUPYTERHUB_API_TOKEN", "")     # your JupyterHub token, used as the API key

DATA = pathlib.Path("data")                                # downloaded documents go here
# Your scores: in your home folder on the JupyterHub; in the working folder on Colab
SCORE_FILE = pathlib.Path("my_scores.json") if ON_COLAB else pathlib.Path.home() / "my_scores.json"

# The five test questions used in both notebooks
QUESTIONS = {
    "Q1": "What is the tuition rate for a graduate student?",
    "Q2": "What is the tuition rate for an international student?",
    "Q3": "I am having problems with Canvas. Who should I contact, and how?",
    "Q4": "Can I take two online courses this semester?",
    "Q5": "May I know Prof. Yong Zheng's office room number and office hours?",
}
CRITERIA = ["Retrieval relevance", "Grounding", "Citation accuracy", "No unsupported claims", "Correct refusal"]
APPROACH_NAMES = {"notebooklm": "1. NotebookLM", "file_search": "2. Gemini File Search",
                  "anythingllm": "3. AnythingLLM", "from_scratch": "4. Built from scratch"}
COLORS = {"notebooklm": "#1479b8", "file_search": "#e88a42", "anythingllm": "#2e9e6b", "from_scratch": "#822dd6"}


def download(files):
    """Download tutorial files into the data folder, skipping files that are already there.

    files: a dict that maps a file name to the folder it sits in on the tutorial website,
           for example {"ITM_Student_Handbook.md": "converted/"}.
    """
    DATA.mkdir(exist_ok=True)
    for name, folder in files.items():
        target = DATA / name
        if not target.exists():
            urllib.request.urlretrieve(MATERIALS_URL + folder + name, target)
        print(f"{name:42s} {target.stat().st_size:>9,} bytes")


def gateway_ok():
    """Return True if the tutorial gateway answers its health check."""
    import httpx
    try:
        return httpx.get(GATEWAY.rsplit("/v1", 1)[0] + "/healthz", timeout=10).status_code == 200
    except Exception:
        return False


# --------------------------------------------------------------------------------------------
# Guesses and self-checks
# --------------------------------------------------------------------------------------------

GUESS_TRUTH = {}   # what actually happened for each guess, filled in by the notebook as it runs


def _scores():
    """Read everything saved in the score file (scores and guesses); empty if nothing is saved yet."""
    return json.loads(SCORE_FILE.read_text()) if SCORE_FILE.exists() else {}


def guess(key, prompt, options):
    """Show a multiple-choice question with radio buttons and save the choice to the score file.

    key:     a short name for this guess, such as "step1"
    prompt:  the question shown to you
    options: the possible answers
    """
    import ipywidgets as widgets
    buttons = widgets.RadioButtons(options=options, value=None, layout={"width": "max-content"})
    out = widgets.Output()

    def on_change(change):
        # Called when you pick an option: store it next to your scores
        if change["name"] == "value" and change["new"]:
            data = _scores()
            data.setdefault("guesses", {})[key] = change["new"]
            SCORE_FILE.write_text(json.dumps(data, indent=2))
            with out:
                out.clear_output()
                print("Saved your guess. Now run the next cells to find out.")

    buttons.observe(on_change)
    display(Markdown(f"**Guess:** {prompt}"), buttons, out)


def guess_summary():
    """Print how many of your guesses matched what the notebook actually found."""
    guesses = _scores().get("guesses", {})
    right = [k for k in GUESS_TRUTH if guesses.get(k) == GUESS_TRUTH[k]]
    print(f"Your guesses: {len(right)} of {len(GUESS_TRUTH)} matched what happened.")
    for k, truth in GUESS_TRUTH.items():
        print(f"  {k}: you guessed {guesses.get(k)!r}, the result was {truth!r}")


def ok(passed, message):
    """Print one line of a self-check, with a tick or a cross, and return whether it passed."""
    print(("✓ " if passed else "✗ ") + message)
    return passed


# --------------------------------------------------------------------------------------------
# Small helpers for looking at text and chunks
# --------------------------------------------------------------------------------------------

def show_answer(text):
    """Display a model's answer as formatted Markdown, with every dollar sign shown correctly.

    Jupyter normally treats text between two $ signs as a math formula, so an answer such as
    "($34,650) [1], New Undergraduate Students ($53,198)" would be shown partly in math italics,
    with the $ signs and the words in between swallowed. Two things prevent that:
      1. each $ is written as the HTML entity &#36;, so it is never read as the start of a formula;
      2. the answer sits in a block marked "tex2jax_ignore mathjax_ignore", which tells the math
         renderer to leave it alone.
    Bold text, lists, and links still display normally, and copied text contains the $ signs.
    """
    display(Markdown('<div class="tex2jax_ignore mathjax_ignore">\n\n'
                     + text.replace("$", "&#36;")
                     + "\n\n</div>"))


def show_region(text, keyword, before=200, after=500):
    """Print the part of a long text around the first place where `keyword` appears."""
    i = text.find(keyword)
    print(text[max(0, i - before): i + after] if i >= 0 else f"'{keyword}' not found")


def food_rows(markdown):
    """Return the Markdown table rows that mention Food, with extra spaces removed."""
    return [" ".join(line.split()) for line in markdown.splitlines() if line.startswith("|") and "Food" in line]


def chunks_with(chunks, keyword):
    """Return the chunks whose text contains `keyword`."""
    return [c for c in chunks if keyword in c["text"]]


def show_hits(hits):
    """Print a ranked list of (score, chunk) pairs: rank, similarity score, section, and the start of the text."""
    for rank, (score, c) in enumerate(hits, 1):
        label = c["section"] or c["source"]
        body = c["text"][len(c["section"]):].strip()
        print(f"{rank}. {score:.3f}  [{label[:70]}]  {body[:110]!r}")


# --------------------------------------------------------------------------------------------
# Step 1: checking the converted tables
# --------------------------------------------------------------------------------------------

def _money(cell):
    """Return the dollar amount in a table cell as an integer, or None if there is none."""
    m = re.search(r"\$\s*([\d,]+)", cell)
    return int(m.group(1).replace(",", "")) if m else None


def table_sum_problems(markdown):
    """Check every Markdown table that has a Total row.

    For each column with a dollar total, add up the rows above it and count empty cells.
    Returns a list of problems; an empty list means every table adds up with no empty cells.
    """
    problems, block = [], []
    for line in markdown.splitlines() + [""]:
        if line.startswith("|"):                                   # still inside a table
            block.append([c.strip() for c in line.strip("|").split("|")])
            continue
        rows = [r for r in block if not set("".join(r)) <= set("-: ")]   # drop the |---| separator row
        total = next((r for r in rows if "Total" in r[0]), None)
        if total:
            body = [r for r in rows if r is not total and any(_money(c) for c in r[1:])]
            for col in range(1, len(total)):
                if _money(total[col]) is None:
                    continue
                got = sum(_money(r[col]) or 0 for r in body if col < len(r))
                empty = sum(1 for r in rows[1:] if r is not total and col < len(r) and not r[col])
                if got != _money(total[col]) or empty:
                    problems.append(f"{total[col]} column adds up to ${got:,}, {empty} empty cell(s)")
        block = []
    return problems


def check_step1(tuition_md, tuition_raw_md, handbook_md):
    """Self-check for step 1: preprocessed tuition tables add up, and the handbook Markdown is clean."""
    print("Step 1 check")
    a = ok(not table_sum_problems(tuition_md), "Tuition tables add up to their totals with no empty cells")
    b = ok("combined:" in tuition_md, "Merged cells are spelled out")
    c = ok("&amp;" not in handbook_md and not re.search(r"\.{10,}|[\ue000-\uf8ff]", handbook_md),
           "Handbook has no &amp;, table of contents, or private bullet symbols")
    print("Problems in the unprocessed version:", table_sum_problems(tuition_raw_md) or "none")
    return a and b and c


# --------------------------------------------------------------------------------------------
# Step 2: chunking by heading
# --------------------------------------------------------------------------------------------

def split_long(body, max_chars):
    """Split a section into paragraphs; split any paragraph that is still too long into lines.

    Returns a list of pieces, each at most `max_chars` characters (a single overlong line is cut).
    """
    pieces = []
    for para in [p for p in re.split(r"\n\s*\n", body) if p.strip()]:
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        buf = ""
        for line in para.splitlines():
            while len(line) > max_chars:                  # a single line longer than the limit
                if buf:
                    pieces.append(buf)
                    buf = ""
                pieces.append(line[:max_chars])
                line = line[max_chars:]
            if buf and len(buf) + len(line) + 1 > max_chars:
                pieces.append(buf)
                buf = line
            else:
                buf = f"{buf}\n{line}" if buf else line
        if buf:
            pieces.append(buf)
    return pieces


def chunk_by_heading(text, title, source, make_chunk, max_chars=1500):
    """Split Markdown into chunks that follow its headings.

    text:       the Markdown document
    title:      the document title, which starts every section path
    source:     the file name stored with each chunk
    make_chunk: a function (path, piece) -> chunk text; the notebook's section_chunk()
    max_chars:  the longest piece of body text in one chunk

    Each heading starts a new section. Its path, such as
    "Tuition, Fees, and Costs > New Graduate Student (Mies Campus)", is kept with every chunk.
    Short paragraphs of one section are packed together up to `max_chars`.
    """
    headings, sections, lines, path = {}, [], [], title
    for line in text.splitlines() + ["# END"]:                  # "# END" flushes the last section
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if not m:
            lines.append(line)
            continue
        body = "\n".join(lines).strip()
        if body:
            sections.append((path, body))
        lines, level = [], len(m.group(1))
        headings = {lv: name for lv, name in headings.items() if lv < level}   # forget deeper headings
        headings[level] = m.group(2).strip()
        path = " > ".join([title] + [headings[lv] for lv in sorted(headings) if headings[lv] != title])
    chunks = []
    for path, body in sections:
        piece = ""
        for para in split_long(body, max_chars):
            if piece and len(piece) + len(para) + 2 > max_chars:   # this paragraph would overflow
                chunks.append({"source": source, "section": path, "text": make_chunk(path, piece)})
                piece = para
            else:
                piece = f"{piece}\n\n{para}" if piece else para
        if piece:
            chunks.append({"source": source, "section": path, "text": make_chunk(path, piece)})
    return chunks


def chunk_stats(presets):
    """Print, for each preset, how many chunks it has and how long they are."""
    import statistics
    for name, chunks in presets.items():
        lengths = [len(c["text"]) for c in chunks]
        print(f"Preset {name}: {len(chunks):3d} chunks, median {statistics.median(lengths):,.0f} characters, "
              f"longest {max(lengths):,}")


def check_step2(presets):
    """Self-check for step 2: preset C chunks carry their section path and keep the Q3 answer whole."""
    print("Step 2 check")
    C = presets["C"]
    a = ok(all(c["text"].startswith(c["section"]) for c in C), "Every preset C chunk starts with its section path")
    q3 = [c for c in chunks_with(C, "supportdesk@illinoistech.edu") if "cannot assist" in c["text"]]
    b = ok(bool(q3), "The Q3 answer (email and 'cannot assist') sits in one preset C chunk")
    c = ok(max(len(x["text"]) for x in C) <= 1700, "No preset C chunk is much longer than the limit")
    return a and b and c


# --------------------------------------------------------------------------------------------
# Step 3: embeddings
# --------------------------------------------------------------------------------------------

EMBED_MODEL = "bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge's instruction for queries


def embed(texts, is_query=False):
    """Turn texts into vectors with the bge-small-en-v1.5 model on the tutorial gateway.

    texts:    a list of strings
    is_query: True for questions; adds bge's query prefix to each text

    Returns (vectors, info). vectors is a numpy array with one row of 384 numbers per text, each row
    scaled to length 1. info has the token count of each text, how many were longer than the model's
    512-token limit (and so cut off), and how many vectors came from the gateway's cache.
    """
    import httpx
    texts = [QUERY_PREFIX + t if is_query else t for t in texts]
    vectors, info = [], {"tokens": [], "truncated": 0, "cached": 0}
    for i in range(0, len(texts), 128):                          # send at most 128 texts per request
        r = httpx.post(f"{GATEWAY}/embeddings", timeout=300, headers={"Authorization": f"Bearer {HUB_TOKEN}"},
                       json={"model": EMBED_MODEL, "input": texts[i:i + 128]})
        r.raise_for_status()
        d = r.json()
        vectors += [e["embedding"] for e in d["data"]]
        info["tokens"] += d["edrag"]["token_counts"]
        info["truncated"] += d["edrag"]["truncated_inputs"]
        info["cached"] += d["edrag"]["cache_hits"]
    V = np.array(vectors, dtype=np.float32)
    return V / np.linalg.norm(V, axis=1, keepdims=True), info


def check_step3(similarity, index, presets, search):
    """Self-check for step 3: similarity() gives one score per chunk, and Q3's answer is retrieved."""
    print("Step 3 check")
    s = similarity(index["C"], index["C"][0])
    a = ok(s.shape == (len(presets["C"]),) and abs(float(s[0]) - 1) < 1e-3, "similarity() returns one score per chunk")
    hits = search(QUESTIONS["Q3"], "C", k=5)
    b = ok(any("supportdesk@illinoistech.edu" in c["text"] for _, c in hits), "The Q3 answer is in the top 5 for preset C")
    return a and b


# --------------------------------------------------------------------------------------------
# Step 4: checking generated answers
# --------------------------------------------------------------------------------------------

def check_step4(answer):
    """Self-check for step 4: Q3 has the right contact details, and Q5 refuses without inventing a room."""
    print("Step 4 check")
    q3 = answer(QUESTIONS["Q3"])
    a = ok("supportdesk@illinoistech.edu" in q3 and "3375" in q3, "Q3 gives the right email and phone number")
    q5 = answer(QUESTIONS["Q5"])
    refused = re.search(r"could not find|not (?:contain|include|provide|list|mention)|no information", q5, re.I)
    room = re.search(r"\b(?:room|office)\s*(?:number\s*)?(?:is\s*)?[A-Z]?\d{2,4}\b", q5, re.I)
    b = ok(bool(refused) and not room, "Q5 says the documents do not have it, without inventing a room number")
    return a and b


def quick_facts_check(answer, prompts):
    """Run Q1-Q5 with each prompt and check every answer for its key facts. Prints a table of ticks.

    answer:  the notebook's answer() function (called with quiet=True, so answers are not printed)
    prompts: a dict that maps a short name to a prompt template

    The checks are rough string tests of the expected answers (see the test questions page):
      Q1 names $34,650; Q2 names all three tuition figures; Q3 gives the right email and phone;
      Q4 states the one-course rule and the first-semester rule; Q5 says the documents do not have it
      and gives no room number. A tick does not replace reading the answer.
    """
    refused = r"could not find|not (?:contain|include|provide|list|mention|specify)|no (?:specific )?information"
    room = r"\b(?:room|office)\s*(?:number\s*)?(?:is\s*)?[A-Z]?\d{2,4}\b"
    checks = {
        "Q1": lambda t: "34,650" in t,
        "Q2": lambda t: all(x in t for x in ("34,650", "53,198", "58,435")),
        "Q3": lambda t: "supportdesk@illinoistech.edu" in t and "3375" in t,
        "Q4": lambda t: bool(re.search(r"\bone\b.{0,30}online course", t, re.I)) and "first semester" in t.lower(),
        "Q5": lambda t: bool(re.search(refused, t, re.I)) and not re.search(room, t, re.I),
    }
    table = {name: {q: "✓" if checks[q](answer(QUESTIONS[q], prompt=prompt, quiet=True)) else "✗"
                    for q in QUESTIONS} for name, prompt in prompts.items()}
    display(pd.DataFrame(table).T.rename_axis("Prompt"))


# --------------------------------------------------------------------------------------------
# Scores: saving them, and the tables and charts that compare approaches
# --------------------------------------------------------------------------------------------

def _load():
    """Return the saved overall and detailed scores, each a dict keyed by approach."""
    data = _scores()
    return {"overall": data.get("overall", {}), "detailed": data.get("detailed", {})}


def _store(kind, approach, scores):
    """Save one approach's scores ("overall" or "detailed") without touching anything else in the file."""
    data = _scores()
    data.setdefault(kind, {})[approach] = scores
    SCORE_FILE.write_text(json.dumps(data, indent=2))


def scored_approaches():
    """Return the approaches that already have overall scores saved, for example from Notebook 1."""
    return [APPROACH_NAMES[a] for a in APPROACH_NAMES if a in _load()["overall"]]


def save_overall(approach, scores):
    """Check and save one overall correctness score (0, 1, or 2) per question, then show them.

    approach: "notebooklm", "file_search", "anythingllm", or "from_scratch"
    scores:   a dict such as {"Q1": 2, "Q2": 1, ...}
    """
    missing = [q for q in QUESTIONS if scores.get(q) not in (0, 1, 2)]
    if missing:
        print("Please give an overall score (0, 1, or 2) for:", ", ".join(missing))
        return
    _store("overall", approach, {q: scores[q] for q in QUESTIONS})
    display(pd.DataFrame({"Overall correctness": {q: scores[q] for q in QUESTIONS}}).T)
    print(f"Saved. {APPROACH_NAMES[approach]}: {sum(scores[q] for q in QUESTIONS)} / 10 points")


def save_detailed(approach, scores):
    """Check and save five criterion scores (each 0, 1, or 2) per question, then show them.

    scores: a dict such as {"Q1": [2, 2, 1, 2, 2], ...}, in the order of CRITERIA
    """
    missing = [q for q in QUESTIONS
               if len(scores.get(q, [])) != 5 or any(v not in (0, 1, 2) for v in scores[q])]
    if missing:
        print("Please give five scores (0, 1, or 2) for:", ", ".join(missing))
        return
    _store("detailed", approach, {q: list(scores[q]) for q in QUESTIONS})
    table = pd.DataFrame({q: scores[q] for q in QUESTIONS}, index=CRITERIA).T
    table["Total"] = table.sum(axis=1)
    display(table)
    print(f"Saved. {APPROACH_NAMES[approach]}: {table['Total'].sum()} / 50 points")


def _plot(totals, by_question, total_max, question_max, by_criterion=None):
    """Draw the comparison charts: totals, scores by question, and (if given) scores by criterion.

    Every bar is labelled with its value, so a score of 0 shows as "0" instead of a missing bar.
    """
    from matplotlib.patches import Patch
    names = list(totals.index)
    colors = [COLORS[a] for a in by_question.attrs["order"]]
    n = 3 if by_criterion is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.8))
    axes[0].bar(names, totals.values, color=colors)
    axes[0].set_title(f"Total (max {total_max})")
    axes[0].set_ylim(0, total_max * 1.12)
    axes[0].tick_params(axis="x", rotation=15)
    by_question.plot.bar(ax=axes[1], rot=0, color=colors, legend=False, title=f"By question (max {question_max})")
    axes[1].set_ylim(0, question_max * 1.18)
    axes[1].set_yticks(range(0, question_max + 1, 1 if question_max <= 2 else 2))
    if by_criterion is not None:
        by_criterion.iloc[::-1].plot.barh(ax=axes[2], color=colors, legend=False, title="By criterion (max 10)")
        axes[2].set_xlim(0, 11.5)
    for ax in axes:
        for container in ax.containers:
            ax.bar_label(container, fmt="%g", padding=2, fontsize=8)
    fig.legend(handles=[Patch(color=c, label=nm) for c, nm in zip(colors, names)],
               loc="lower center", ncol=len(names), frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()


def compare_overall():
    """Show a table and charts of the overall correctness scores of every approach saved so far."""
    scores = _load()["overall"]
    order = [a for a in APPROACH_NAMES if a in scores]
    if not order:
        print("No overall scores saved yet.")
        return
    by_question = pd.DataFrame({APPROACH_NAMES[a]: scores[a] for a in order})
    by_question.attrs["order"] = order
    totals = by_question.sum()
    display(Markdown("**Overall correctness per question (max 2 each)**"))
    display(pd.concat([by_question, totals.to_frame("Total").T]))
    _plot(totals, by_question, 10, 2)


def compare_detailed():
    """Show a table and charts of the five-criterion scores of every approach saved so far."""
    scores = _load()["detailed"]
    order = [a for a in APPROACH_NAMES if a in scores]
    if not order:
        print("No detailed scores saved yet.")
        return
    by_question = pd.DataFrame({APPROACH_NAMES[a]: {q: sum(scores[a][q]) for q in QUESTIONS} for a in order})
    by_question.attrs["order"] = order
    by_criterion = pd.DataFrame({APPROACH_NAMES[a]: [sum(scores[a][q][i] for q in QUESTIONS) for i in range(5)]
                                 for a in order}, index=CRITERIA)
    totals = by_question.sum()
    display(Markdown("**Points per question by criterion (max 10 each)**"))
    display(pd.concat([by_question, totals.to_frame("Total").T]))
    _plot(totals, by_question, 50, 10, by_criterion)


def download_scores():
    """On Colab, download the score file so it survives the end of the session."""
    if not ON_COLAB:
        print("Your scores are saved in", SCORE_FILE)
    elif SCORE_FILE.exists():
        from google.colab import files
        files.download(str(SCORE_FILE))
    else:
        print("Nothing to download yet: save at least one score first.")


# --------------------------------------------------------------------------------------------
# Notebook 1: AnythingLLM answers recorded on the tutorial server
# --------------------------------------------------------------------------------------------

_recorded = None


def load_recorded():
    """Download (once) the AnythingLLM answers recorded on the tutorial server before class."""
    global _recorded
    if _recorded is None:
        url = MATERIALS_URL + "practice_offline/anythingllm_recorded.json"
        _recorded = json.loads(urllib.request.urlopen(url).read())
    return _recorded


def recorded_answer(workspace, question):
    """Return the recorded AnythingLLM answer for this workspace and question, or None if there is none."""
    rec = load_recorded()
    for item in rec["with_and_without_documents"]:
        if item["workspace"] == workspace and item["question"] == question:
            return item
    q = next((k for k, v in QUESTIONS.items() if v == question), None)
    return rec["edrag"].get(q) if workspace == "edrag" else None
