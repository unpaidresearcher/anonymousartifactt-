# CompoundBench - Replication Package

CompoundBench is a benchmark of **584 Python functions**, each carrying concerns from at least
two of five categories, each retaining at least one concern no automated resolver can fix, and
each paired with a repository test that was green when the item was admitted.

This package contains the benchmark artifact, the grader, and the code that produced every
number in the paper, ordered by the stage it belongs to.

---

## 1. Benchmark artifact

| file | contents |
|---|---|
| `benchmark/compoundbench_584.json` | the 584 items |
| `benchmark/control_single_concern_186.json` | the single-concern control population |
| `benchmark/statistics.json` | category, concern-count and repository distributions |
| `benchmark/grade_verif.py` | the grader, importable and runnable standalone |

### Item schema

```
id                 cb_0001 .. cb_0584
repo, file         source repository and path
func, start, end   function name and its line range
n_lines            length in lines
concerns           the admitted concern categories (>= 2)
rules              the Ruff rule codes behind them
nonfixable         concerns `ruff --fix` cannot resolve (>= 1 by construction)
autofix_cleared    categories `ruff --fix` does clear
orig_src           the function source, dedented
test_target        the covering test, where one is recorded
baseline_secs      that test's runtime when green
provenance         source commit and container image, for reproducing execution
order_dependent    whether forward vs reversed concern order flips correctness
final_set_differs  whether the two orders leave different concerns behind
```

### Distribution

- concern counts: **513** items at k=2, **65** at k=3, **6** at k=4
- categories: bugbear 399, modernize 380, simplify 263, comprehension 146, cleanup 57
- order-dependent: **66 of 584 (11.3%)**

### Grading

A prediction is correct only when all three hold.

1. every admitted concern category is gone and no new category appears
2. three anti-degeneracy guards pass, i.e. the function still exists, keeps at least half its
   statements, and stays within a 0.35 similarity floor of the original
3. the covering repository test still passes, for execution-based validation

```python
import json, grade_verif
items = json.load(open("benchmark/compoundbench_584.json"))
g = grade_verif.grade(items[0], my_rewrite)   # -> dict with 'correct' and per-guard detail
```

The static grader, i.e. steps 1 and 2, falsely accepts **13.5%** of what it marks correct when
checked against execution. Every absolute number in the paper carries that error rate.

---

## 2. Code, in the order it should be run

Each directory is a stage. Files are numbered in run order within a stage.

### `01_mine_and_select_analyzer`
Establishes how common compound functions are, and why Ruff.

1. `01_mine_funnel_12_repos.py` — full funnel over 12 repositories, 51,300 parseable functions
2. `02_concern_count_histogram.py` — concerns per function, flagged vs compound
3. `03_run_three_analyzers_sample.py` — Ruff, Pylint and SonarQube on a 4,000-function sample
4. `04_analyze_analyzer_comparison.py` — produces the analyzer comparison table
5. `05_why_ruff_not_pylint.py` — the two disqualifying properties, measured

### `02_build_benchmark`
Turns the mined pool into 584 admitted items.

1. `01_build_candidate_pool.py` — apply criteria 1 to 3
2. `02_coverage_generic.sh` — coverage.py over a repository's suite
3. `03_coverage_sympy.sh` / `04_coverage_django.sh` — per-repository coverage runs
4. `05_extend_via_swebench_images.py` — mine and verify inside era-matched containers
5. `06_consolidate_to_584.py` — merge, dedupe and emit the final set

### `03_grade`
1. `01_grader.py` — concern check plus the three guards
2. `02_prompt_builders.py` — every prompt variant, shared so arms cannot drift
3. `03_similarity_floor_sweep.py` — sweeps the 0.35 floor from 0.0 to 0.8

### `04_evaluate`
1. `01_zero_shot_arm.py` — no concern named
2. `02_all_pass_structures_local.py` — the seven pass structures, open-weight models
3. `03_hosted_api_arms.py` — the same arms through hosted APIs

### `05_execution_validation`
Measures what the static grader gets wrong.

1. `01_container_execution.py` — extension items, inside their SWE-bench images
2. `02_local_checkout_execution.py` — coverage-admitted items, in local checkouts
3. `03_execution_over_584.py` — both paths over the full set
4. `04_combine_and_score.py` — the execution-validated score with coverage bounds

### `06_experts`
1. `01_mine_star_training_pairs.py` — STaR pairs from the base model's own successes
2. `02_train_per_concern_adapter.py` — one LoRA adapter per concern
3. `03_arrow_routing.py` / `04_lorahub_weighting.py` — composition baselines

### `07_extra_experiments`
1. `01_order_dependence.py` — forward vs reversed concern order, all 584
2. `02_ruff_autofix_then_llm.py` — `ruff --fix` first, then the localized prompt

### `08_analysis_and_figures`
1. `01_paired_mcnemar.py` — paired tests over identical item sets
2. `02_make_figures.py` / `03_make_spread_figure.py` — the paper's figures

### A note on the scripts

The scripts under `code/` are the ones that produced the paper's numbers, unmodified except for
renaming and reordering. They therefore read and write the authors' working filenames, and they
expect the intermediate files each stage produces rather than the cleaned artifact in
`benchmark/`. Run a stage end to end to reproduce it; the published artifact is the output of
stages 01 and 02, with internal bookkeeping fields removed.

---

## 3. Results included

`results/` carries the outputs that take hours to reproduce.

| file | what it is |
|---|---|
| `order_dependence584_labels.json` | per-item order-dependence labels, 8h 54m run |
| `ruffthen_sonnet5.json` | ruff-autofix-then-LLM vs localized, paired, 38 min |
| `funnel12.json`, `funnel12_byrepo.json` | the mining funnel |
| `funnel_hist.json` | concerns-per-function census |

---

## 4. Requirements

- Python 3.8+, `ruff`, `coverage`
- Docker, for the container execution path and SWE-bench images
- A GPU for the open-weight arms and adapter training; hosted arms need API keys
- `transformers`, `peft`, `torch`, `anthropic` or `openai`

Determinism: open-weight arms use greedy decoding. Hosted arms ran once on a fixed date and
are not reproducible in the same sense.

