# CompoundBench — Replication Package

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
