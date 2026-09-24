# Contextual Resume Compression with RAG - initial-implementation

A implementation, with an extended evaluation, of **"Contextual Compression of
Unstructured Resumes using VectorStore RAG"** (IEEE ICCRDA 2025,
[10.1109/icdsis65355.2025.11070875](https://doi.org/10.1109/icdsis65355.2025.11070875)).

The pipeline chunks a resume, embeds it into a per-resume FAISS index, retrieves
and compresses the chunks relevant to each schema section, and has an LLM emit
validated JSON. What the paper reports is a mean cosine similarity; what this
repository adds is the context needed to read that number: field-level F1, a
random-pairing floor, a five-way ablation with token/latency/cost, a
rule-based baseline, and an error breakdown.

> **Status:** complete and measured, including the paper's own method. 182
> tests, gold standard built from the real DataTurks corpus (220 resumes, 100
> test / 120 dev), ablations run once on the held-out test set - and reproducible
> end to end with no API key, because both the extraction model and the embedding
> model can be served locally.

---

## Results

**Held-out test set, n = 100, run once each.** Everything here was produced by
`eval/run_ablations.py`; the tables are copied from `results/`, never retyped.

| extractor | variant | cosine (dense) | **macro F1** | calls/resume |
|---|---|---:|---:|---:|
| `rules-v1` | **full_context** | 0.947 | **0.736** | 0 |
| `rules-v1` | rag_none | 0.932 | 0.662 | 0 |
| `rules-v1` | rag_embeddings_filter | 0.902 | 0.619 | 0 |
| `llama3.1:8b` | **rag_llm_extractor** (the paper's method) | 0.911 | 0.581 | 20.4 |
| `rules-v1` | rag_lexical | 0.910 | 0.553 | 0 |
| | **random-pairing floor** | **0.778** | | |

Same rule-based rows under TF-IDF cosine rather than dense: 0.745, 0.691, 0.676
and 0.634, against a floor of 0.161. The floor moves further than any of the
systems do, which is the point of the next section.

Dense = `nomic-embed-text` (768-d) via Ollama, chat model `llama3.1:8b`, also
local. No API key, no network, no spend. A matched rules-vs-LLM comparison on a
10-resume subset is two sections down.

Reproduced with `rapidfuzz`, `faiss-cpu` and `langchain-text-splitters`
installed as well as without them: the pure-Python fallbacks agreed to within
**0.003 macro F1**, and per-field F1 was identical. The numbers do not depend on
which of the two paths ran.

### What the floor does to the headline number

The paper reports 0.960 mean cosine. The regexes in this repo score 0.947 over
the full test set, and 0.964 on the ten resumes where an LLM was also run.
That doesn't mean the paper's method is no better. It means cosine can't tell
the two apart.

Three things to look at:

**Unrelated resumes already score 0.778.** Pair each gold JSON with a different
resume's prediction and dense cosine still comes back at 0.778, so only the
band from 0.778 to 1.0 carries information. TF-IDF puts the floor at 0.161 on
the same predictions. The floor belongs to the embedding model, not to the
system, so a cosine number quoted without both is not much use.

**Cosine barely moves while quality drops.** Across the four variants dense
cosine spans 0.045 and macro F1 spans 0.186, roughly four times the
resolution. A metric that sits at 0.9-something whatever you change is not
much help when choosing between designs.

**Cosine ranks two variants backwards.** `rag_lexical` gets a higher dense
cosine than `rag_embeddings_filter` (0.909 vs 0.902) and a lower F1 (0.549 vs
0.619). Tuning on cosine alone would pick the worse one.

Floor by section (dense): personal 0.775, education 0.635, work_experience
0.620, skills 0.405. Skills has the most headroom because skill lists differ
between people; `personal` has almost none, since every resume's personal block
is the same handful of fields.

If you re-run this paper, publish the floor next to the score, or publish F1
instead.

### Rules vs an LLM, on the same resumes

The paper's method needs a language model. Run one - `llama3.1:8b`, locally via
Ollama - against the rule-based baseline on an identical subset:

| extractor | variant | cosine (dense) | **macro F1** | s/resume |
|---|---|---:|---:|---:|
| **`rules-v1`** | full_context | **0.964** | **0.779** | 0.0 |
| `llama3.1:8b` | full_context | 0.924 | 0.762 | 72 |
| `rules-v1` | rag_none | 0.951 | 0.697 | 0.0 |
| `llama3.1:8b` | rag_none | 0.927 | **0.719** | 85 |
| `llama3.1:8b` | **`rag_llm_extractor`** *(the paper's method)* | 0.915 | 0.657 | 76 |

*Test-set subset, n = 10, floor 0.768-0.788. Small - a probe, not a verdict  - 
because a local 8B model costs ~76-85 seconds per resume.*

**The paper's method finishes last**, and the full run is harsher than the
probe was. Over all 100 held-out resumes `rag_llm_extractor` scores macro F1
**0.581** - below the rule-based baseline on full context (0.736), below it on
plain retrieval (0.662), and below chunk-level embedding filtering (0.619). The
n=10 subset had it at 0.657, so the small sample was flattering it.

Within the LLM itself the ordering is the same one retrieval shows everywhere
here: 0.762 with the whole document, 0.719 with retrieval, 0.657 with retrieval
plus compression (all n=10, matched resumes).

It is also the most expensive thing in the table. `LLMChainExtractor` spends one
call per retrieved chunk per section, so it runs **20.4 model calls per resume
against 4.1** for the other LLM variants, and its *total* token count goes up
rather than down - 4,531 per resume against 2,900 for uncompressed retrieval.
The calls that shrink the extraction prompt cost more than they save. The n=100
run took 104 minutes on an 8B local model; the rule-based rows take under a
minute.

On short templated documents there doesn't seem to be much left for compression
to remove that retrieval hasn't already dropped.

Two further things fall out of it.

**Regular expressions beat the 8B model with full context**, on both metrics,
taking 72 seconds per resume less to do it. The rules exploit the Indeed
template directly; the model has to rediscover it from the text every time.

**Retrieval reverses that.** Under `rag_none` the model wins (0.719 vs 0.697).
Chunking destroys the layout the rules depend on, while the model degrades more
gracefully - it reads meaning, not position. So the LLM's real advantage here is
robustness to messy input, not accuracy on clean input. On a corpus of
free-form PDFs rather than uniform Indeed exports, that ordering would likely
flip again.

**About that 0.964.** Tempting to put it next to the paper's 0.960 and call it
a win, but that comparison doesn't hold - different embedding models, and a
cosine only means something against the floor of its own model. The narrower
claim is the useful one: under a dense embedding model, pattern matching gets
0.964 against a 0.788 floor, so a mid-0.9s cosine says very little about how
good an extractor is.

### Retrieval makes this task worse, monotonically

Macro F1 falls at every step away from the whole document: 0.736 → 0.659 →
0.619 → 0.549. These resumes average ~4,200 characters and fit in any modern
context window, and they are Indeed exports whose *layout carries the meaning*  - 
the name is on line one, the employer on the line under the job title. Chunking
destroys that structure before the extractor ever sees it. `full_context` is not
a strawman ablation here; it wins outright.

Dev agrees with test (macro F1 0.719 vs 0.736 for `full_context`), so the rules
are not overfitted to dev.

### One call per section, or one call for everything (section 4.4)

The methodology asks for both to be compared. Per-section wins everywhere, and
by a wide margin where it matters most:

| variant | per_section | combined | delta |
|---|---:|---:|---:|
| full_context | **0.736** | 0.593 | −0.143 |
| rag_none | **0.662** | 0.540 | −0.122 |
| rag_embeddings_filter | **0.619** | 0.602 | −0.017 |
| rag_lexical | **0.553** | 0.534 | −0.019 |

*Rule-based extractor, test n = 100, macro F1.*

The pattern is informative: the penalty is largest exactly where the context is
richest. Give every section the whole document and the extractor has to decide
which parts belong to which field; give each section only its own retrieved
context and that decision is already made. Where compression has already
narrowed the context, the two converge - there is little left to confuse.

### Per-field results (`full_context`, test n = 100)

| field | P | R | F1 | gold values |
|---|---:|---:|---:|---:|
| `personal.name` | 1.00 | 1.00 | **1.00** | 100 |
| `personal.location` | 0.99 | 0.99 | **0.99** | 91 |
| `personal.profile_url` | 0.89 | 0.97 | 0.93 | 92 |
| `education.institution` | 0.71 | 0.91 | 0.80 | 141 |
| `education.degree` | 0.78 | 0.70 | 0.74 | 122 |
| `skills` | 0.65 | 0.63 | 0.64 | 1,111 |
| `work_experience.company` | 0.41 | 0.80 | 0.54 | 99 |
| `work_experience.designation` | 0.38 | 0.87 | 0.52 | 112 |
| `education.graduation_year` | 0.35 | 0.68 | 0.47 | 73 |
| `personal.email` | - | - | - | **0** |
| `personal.phone` | - | - | - | **0** |

Cosine by section: personal 0.945, education 0.810, work_experience 0.668,
skills 0.610. Note how flat cosine is compared with F1 - `work_experience`
scores 0.668 cosine on fields whose F1 is barely above 0.5. That divergence is
the argument for reporting both.

Recall beats precision almost everywhere: the baseline over-produces. For
`skills` that is partly the gold's fault (see the annotation notes below), and
`eval/error_analysis.py` is what separates the two.

**The paper's 0.960 is not comparable to anything above.** It used OpenAI
embeddings on a different gold standard; cosine scores do not transfer across
embedding models. Reproducing it needs a key and the `rag_llm_extractor` row.

---

## Why the extra metrics

**Cosine similarity alone cannot distinguish a good extraction from a plausible
one.** Two resumes from the same domain embed close together even when they
share no facts - measured at **0.778** here. The random-pairing floor (section
5.3 of the methodology) makes this visible: each gold JSON is scored against a
*different* resume's prediction. The headline result is the gap between that
floor and the system's score, not the score itself, and on this corpus that gap
turns out to be small enough to change what the paper's number means.

**Field-level P/R/F1** (`rapidfuzz` token-set ratio ≥ 85, one-to-one greedy
matching) answers the question a cosine score dodges: how many of the actual
degrees, employers and skills were found, and how many were invented. Email,
phone and graduation year are additionally scored on exact match, because a
near-miss email is a wrong email - though on this corpus email and phone have no
gold values at all (see below), so they report zero support rather than a score.

**Fuzzy matching needs a subset guard.** `rapidfuzz.token_set_ratio` returns
**100 for any subset**, so under the metric as literally specified a prediction
of `"Acme"` scores as a correct match for gold `"Acme Analytics Private
Limited"` - and every truncation error vanishes into the true positives. A match
here additionally requires the two values to share at least half of the longer
one's tokens (`TOKEN_COVERAGE_MIN`). Set it to `0.0` to reproduce the
unguarded metric; the guard's effect on F1 is worth reporting either way.

**The paper's correlation figures** (Pearson 0.0821, Spearman 0.0722) are not
reproduced here. The paper does not state which two variables were correlated,
and values that close to zero indicate no relationship, so reproducing them
would mean reproducing an ambiguity. Field-level F1 replaces them.

### Error analysis (section 5.6)

`eval/error_analysis.py` classifies every field-level disagreement into a
taxonomy - `truncated`, `over_extended`, `wrong_date`, `missed_date`,
`merged_entries`, `split_entries`, `hallucinated`, `missed`, `parse_failure`  - 
and samples failures **stratified by category**, so a rare-but-diagnostic
failure is not drowned out by the most common one.

Best variant (`full_context`), test set, 1,400 findings:

| category | count | share |
|---|---:|---:|
| `hallucinated` | 804 | 57.4% |
| `missed` | 463 | 33.1% |
| `split_entries` | 89 | 6.4% |
| `wrong_date` | 22 | 1.6% |
| `over_extended` | 12 | 0.9% |
| `merged_entries` | 5 | 0.4% |
| `truncated` | 4 | 0.3% |
| `missed_date` | 1 | 0.1% |

Nearly two-thirds of all findings are in `skills` (382 hallucinated, 394
missed), which is where the gold is weakest.

These are disagreements, not errors. The DataTurks skills annotations are
incomplete, so a lot of the "hallucinated" skills are real skills nobody
marked up. Quoting 804 as a hallucination rate would be wrong.
`notebooks/02_error_analysis.ipynb` is where you go through the sampled 20 by
eye; how many turn out to be gold gaps is worth reporting on its own.

Full output: `results/error_analysis.md`.

---

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                             # add your OPENAI_API_KEY
```

```bash
python data/download.py                          # needs the Kaggle CLI
python -m eval.build_gold --make-splits          # gold JSON + the fixed 100/120 split
```

Reproduce the rule-based rows with nothing installed but this repository:

```bash
python -m eval.run_ablations --split test --confirm-test --extractor heuristic --variants full_context rag_none rag_embeddings_filter rag_lexical --dump-predictions
python -m eval.error_analysis --predictions results/predictions_full_context.json
```

**Run the LLM rows locally and free, with [Ollama](https://ollama.com):**

```bash
ollama pull llama3.1 && ollama pull nomic-embed-text
python -m eval.run_ablations --split test --confirm-test --backend ollama --embedder ollama --dump-predictions
```

Or against the paid API, which is what the paper used:

```bash
python -m eval.estimate_cost --split test                 # price it first, offline
python -m eval.run_ablations --split dev --limit 20 --dump-predictions   # tune here
python -m eval.run_ablations --split test --confirm-test --dump-predictions
```

`--backend auto` (the default) prefers an `OPENAI_API_KEY`, falls back to a
running Ollama server, and records which one answered in the results file.

Serve it:

```bash
python -m src.api            # http://localhost:8000
```

```bash
curl -s -X POST localhost:8000/extract -F "file=@resume.pdf" | jq .prediction
```

Or with Docker. With no environment at all the container serves the rule-based
extractor and says so on `/health`; give it a key or a reachable Ollama and it
uses that instead. The image builds and `/health` answers from inside it; the
engine-selection behaviour below is covered by tests and was checked against a
running server with no credentials, but not re-checked inside the container
after the last change to it.

```bash
docker build -t resume-rag-extractor .
docker run -p 8000:8000 resume-rag-extractor                      # rules, no config
docker run -p 8000:8000 --env-file .env resume-rag-extractor      # OpenAI
docker run -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 resume-rag-extractor
```

---

## Architecture

```
PDF / text
   → loaders.normalise_text        NFKC, bullets unified, hyphenated breaks rejoined
   → chunking.split_text           ~600 chars, 100 overlap, recursive separators
   → embeddings                    text-embedding-3-small, or local TF-IDF (cached)
   → ResumeIndex                   one FAISS index per resume
   → per-section query x 4         personal / education / work_experience / skills
   → contextual compression        LLMChainExtractor | EmbeddingsFilter | Lexical
   → extraction                    HeuristicExtractor (rules) or Extractor (LLM)
   → Flask POST /extract
```

Every embedder and every extractor satisfies the same interface, so any
combination runs and every run records which pair produced it.
`src/ollama_backend.py` is a stdlib-only shim exposing the slice of the OpenAI
client surface this repository calls, so `Extractor` needs no branch for the
backend - and because Ollama returns `prompt_eval_count` / `eval_count`, the
token figures stay measured rather than estimated.

Output schema (`src/schema.py`): `personal{name,email,phone,location,profile_url}`,
`education[]{degree,institution,graduation_year}`,
`work_experience[]{company,designation,start_date,end_date,summary}`,
`skills[]`, `certifications[]`, `projects[]`.

### The five ablation variants

| variant | what it answers | needs a key |
|---|---|---|
| `full_context` | Is retrieval needed at all when a resume fits in the context window? | no |
| `rag_none` | Retrieval baseline, no compression | no |
| `rag_embeddings_filter` | Does cheap, chunk-level compression help? | no |
| `rag_lexical` | **Does compression need a model at all?** | no |
| `rag_llm_extractor` | The paper's method | yes |

`rag_lexical` is not in the paper. `LLMChainExtractor` spends one model call per
retrieved chunk, so the null hypothesis - that keeping the lines sharing
vocabulary with the query captures most of the benefit for nothing - has to be
tested before the expensive option is recommended.

Extraction is chosen separately from retrieval with `--extractor`:

| extractor | what it is | needs a key |
|---|---|---|
| `heuristic` | `rules-v1` - regex and the Indeed template. The baseline the paper omits. | no |
| `llm` | temperature-0 chat model, Pydantic-validated, one retry | no, with Ollama |

Each is scored on accuracy as well as tokens, latency and cost, since
compression is trading model calls for a smaller prompt.

---

## Data

| dataset | use | committed? |
|---|---|---|
| [DataTurks Resume Entities for NER](https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner) | 220 annotated resumes → gold JSON; 100 test / 120 dev | no - script + split IDs only |
| [Resume Dataset (snehaanbhawal)](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | ~2,480 resumes with real PDFs → ingestion path, generalisation set | no |
| [NER Annotated CVs](https://www.kaggle.com/datasets/shubhamanandjha/nerannotatedcvscollection) | attempted for skills-only P/R at scale - **rejected**, see below | no |

The split is fixed by `SPLIT_SEED` in `src/config.py` and the ID lists live in
`data/splits/`. `run_ablations.py` refuses `--split test` without
`--confirm-test`, so the held-out set cannot be spent on tuning by reflex.

### What the corpus actually contains

Building the gold standard surfaced three properties of the DataTurks file that
change how it can be scored. All three are handled in `eval/build_gold.py` and
counted in `data/gold/_cleaning_report.json`.

**1. The character offsets have drifted.** `content` was re-normalised after
annotation, so recorded positions run a few characters behind the text they
describe, and the drift accumulates down each document (shifts cluster on
multiples of 4, up to ~30). Slicing by offset returns a shifted fragment:

```
annotated text : 'Associate Consultant'
offset slice   : 'graphy\n\nAssociate Con'     <- same length, shifted left
```

220 of 3,558 spans mismatch, affecting 45 of 220 resumes. The annotated text is
authoritative wherever it can be found in the content, so each mismatched span
is realigned to the occurrence **nearest its recorded offset** - nearest, not
first, because company names recur and the first occurrence is often not the
annotated one. 216 spans realign; the remaining 4 keep the offset slice and are
counted as `spans_unlocatable`.

**2. "Skills" values are whole sections, not skills.** Half of the 352 gold
skill values exceed 60 characters and the longest is 2,472 - a single value can
be an entire skills section with its `(2 years)` qualifiers attached. Scored
against a system that predicts atomic skills, that measures the annotation
format and nothing else. The blobs are split on their own separators and the
qualifiers stripped, giving **2,555 atomic skills, median 11 characters,
~11.6 per resume**.

**3. "Email Address" contains no email addresses.** 189 of its 190 values are
Indeed profile URLs (`indeed.com/r/ada-lovelace/abc123`); none contain an `@`.
The label is mapped to `personal.profile_url`, not `personal.email`, so a URL
match is never reported as an email match. **Consequence:** the exact-match
email and phone metrics from methodology section 5.2 have no gold to score
against on this corpus - they report zero support rather than a fabricated
number. Scoring them needs the secondary dataset or manual annotation.

### The second corpus was rejected, with evidence

Section 2.3 of the methodology proposes a large skills-only precision/recall run
on the *NER Annotated CVs* collection. `eval/skills_eval.py` implements it and
was run over 1,000 CVs. The corpus does not support the evaluation:

| | |
|---|---|
| median gold "skills" per CV | **79** (max 575) |
| plausible for a real resume | 10-30 |
| most frequent labels | `com`, `english`, `gmail`, `skills`, `knowledge`, `nationality` |

At 79 labels per CV the annotation marks approximately every noun in the
document, so recall is bounded far below 1.0 for any extractor producing a
sensible skill list. The measured F1 is 0.064, and it describes the annotation
density rather than the extraction. `eval/skills_eval.py` therefore prints that
verdict alongside the number instead of reporting the F1 on its own, and this
repository does not use the corpus as a gold standard. Full output:
`results/skills_eval.md`.

### Field coverage - the recall ceiling

| field | resumes with ≥1 gold value |
|---|---:|
| `personal.name` | 218 / 220 (99%) |
| `work_experience.designation` | 202 / 220 (92%) |
| `personal.location` | 199 / 220 (90%) |
| `education.institution` | 194 / 220 (88%) |
| `education.degree` | 192 / 220 (87%) |
| `work_experience.company` | 190 / 220 (86%) |
| `skills` | 183 / 220 (83%) |
| **`education.graduation_year`** | **104 / 220 (47%)** |
| `personal.email` | 0 / 220 (0%) |

Graduation year is annotated in under half the corpus. A low F1 there is mostly
a statement about the annotations, and should be reported next to this number.

### The other cleaning rules

1. `end` is inclusive - the exclusive end is `end + 1`, and offsets are clamped
   to the document.
2. Leading and trailing whitespace is stripped by shrinking the span; spans that
   become empty are dropped (707 trimmed, 1 emptied).
3. Overlapping spans **of the same label** collapse to the longest one (46
   dropped). Cross-label overlaps are kept - a designation legitimately sits
   inside the same sentence as a company name.
4. Repeated mentions of one entity are deduplicated, first occurrence kept
   (1,265 dropped).
5. `UNKNOWN` and `Years of Experience` are outside the evaluated schema (48
   ignored).

**Known limitation, stated plainly:** the annotations are flat spans, so nothing
in the data says which degree belongs to which college. Education and work
entries are paired by document order, which is right for the common
single-degree resume and approximate otherwise. This is exactly why the primary
metric is per *field*, not per tuple.

---

## Development

```bash
pytest                    # 182 tests, no API key and no network required
```

The suite runs without `openai`, `faiss` or `rapidfuzz` installed: the chat
client is faked, FAISS falls back to exact cosine search (identical results at
resume scale), and `rapidfuzz` falls back to a `difflib` equivalent.

Four embedders and two chat backends. The results file always records which
pair produced a number, and the runner prints a warning whenever the cosine
figures are not OpenAI's:

| embedder | what it is | cost | reportable |
|---|---|---|---|
| `text-embedding-3-small` | the paper's setting | paid | yes |
| `nomic-embed-text` | 768-d dense, local via Ollama | free | yes, **as nomic** |
| `tfidf-local-*` | fitted on the corpus, deterministic | free | yes, **as TF-IDF** |
| `hash-*` | wiring checks and tests only | free | no - flagged in the output |

| chat backend | what it is | cost |
|---|---|---|
| `openai` | `gpt-4o-mini` by default | ~$0.65 per sweep |
| `ollama` | any local model, `llama3.1` by default | free |

Running cosine under **both** a dense and a lexical model is the point rather
than a compromise. Dense embeddings reproduce the failure the random-pairing
floor exists to expose - same-domain documents scoring 0.78 against each other  - 
and TF-IDF does not, because it is lexical. Seeing a result under both is how
you find out whether you measured the system or the embedding space.

```
resume-rag-extractor/
├── .github/workflows/  CI: tests on 3.11/3.12, Docker build + /health check
├── data/               download.py + splits/ (no resume data committed)
├── src/                loaders, chunking, embeddings, tfidf, ollama_backend,
│                       retriever, extractor, heuristic, schema, api, config
├── eval/               build_gold.py, metrics.py, run_ablations.py,
│                       error_analysis.py, estimate_cost.py, skills_eval.py
├── notebooks/          01_eda.ipynb, 02_error_analysis.ipynb
├── results/            ablation output, committed alongside the README table
├── tests/
├── Dockerfile
└── requirements.txt
```

Everything that moves a reported number - models, `k`, chunk size, filter
threshold, fuzzy threshold, prices - lives in `src/config.py` and is written
into `results/ablations.json` alongside the scores.

### One-line summary

> Re-implemented an IEEE resume-extraction paper and showed its headline metric
> could not separate the method from a regex baseline: cosine 0.947 for rules
> against a 0.778 random-pairing floor, with field-level F1 spanning 4x the
> range cosine did. Python, FAISS, Ollama, 182 tests.

---

## Cost

`python -m eval.estimate_cost` prices the sweep offline before anything is
spent. For 100 resumes on `gpt-4o-mini`:

| variant | chat calls/resume | tokens/resume | $ /100 |
|---|---:|---:|---:|
| full_context | 4 | ~5,800 | $0.13 |
| rag_none | 4 | ~4,600 | $0.11 |
| rag_embeddings_filter | 4 | ~3,700 | $0.10 |
| rag_llm_extractor | 24 | ~11,300 | $0.29 |
| **full sweep** | | | **~$0.63** |

So the sweep is cheap, and the thing to watch is not the total but the shape:
`rag_llm_extractor` issues **six times the calls** because it compresses each
retrieved chunk individually, and switching to `gpt-4o` multiplies every row by
roughly 17. Those are estimates - output length is assumed and retrieval is
modelled as `k` full-size chunks - so confirm with `run_ablations --split dev
--limit 10` and read the measured `cost_usd_per_100`.

The embedding cache (`results/cache/`) means the gold JSON is embedded once, not
once per variant.
