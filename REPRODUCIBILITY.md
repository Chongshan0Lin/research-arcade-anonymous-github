# Reproducibility

Measured on one CPU core, Python 3.10, no GPU, no network. Nothing here loads a model or a database.

## 1. Commands actually executed to validate this release

| # | Command | Wall clock | Result |
|---|---|---|---|
| 1 | `python -m tasks.agentic_autoresearch.controlled_aggregate --root frozen_artifacts/autoresearch/controlled_task_a` | 20.6 s | exit 0, 9 CSV/JSON rewritten |
| 2 | `python -m tasks.agentic_autoresearch.ebc_aggregate --root frozen_artifacts/autoresearch/evidence_bundle_completion/agent_results` | 5.8 s | exit 0, claim `Supported` on 3 models |
| 3 | `python -m tasks.agentic_autoresearch.litexp_aggregate --root frozen_artifacts/autoresearch/literature_set_expansion/agent_results` | 1.9 s | exit 0, claim `Supported` on 2 models |
| 4 | `python -m tasks.agentic_autoresearch.appendix_tables --root frozen_artifacts/autoresearch` | 0.1 s | exit 0, 9 tables + `missing.json` |
| 5 | `python -m tasks.agentic_autoresearch.graph_autoresearch_report --root frozen_artifacts/autoresearch` | 0.4 s | **13/13 graded checks passed, 1 skipped** (in-repo) |
| 6 | `python tests/test_release_integrity.py` | 3.1 s | **7/7 release checks passed** |

Total under 35 seconds. Every command re-derives its numbers from the bundled episode rows; none
reads anything outside this repository.

### Checks skipped, and why

| Skipped | Reason |
|---|---|
| `previous canonical run preserved` | the earlier natural-action run is not bundled in this release; checkable only in the source tree |
| `git diff --check passes` | graded inside this repository; skipped only if the tree is exported without git metadata |

Both are reported as `SKIP` by the verifier and are excluded from the pass count. They are never
silently counted as passes.

### Not run, deliberately

Model inference, agent evaluation, training, GPU jobs, index reconstruction, corpus crawling, and
embedding computation were **not** run. None is required to reproduce a reported number.

## 2. What the graded verification checks

`graph_autoresearch_report` fails the build if any of these breaks:

frozen split hashes recorded · identical instance IDs across every paired condition · no duplicate
episode keys · one terminal row per expected episode · correct and rewired graph hashes **differ** ·
degree sequences **match** · correct/rewired prompt and tool-schema hashes identical · every cited
evidence ID joins to an observed candidate · **blocked calls returned no candidates** · **no
out-of-interface tool executed** · all 2,400 matched-pool hashes reproduce from the observed pools ·
the manipulation-uptake gate passed.

The last two in bold guard against a specific failure mode: a graph-free condition receiving
graph-derived candidates through a tool it was not granted. Both report zero.

## 3. The LSE 1,200 → 909 split (frequently asked)

- The 1,200 frozen LSE instances comprise **291 development** and **909 held-out screen** instances.
- **No frozen instances were removed after splitting.** 291 + 909 = 1,200 exactly, and the `split`
  label is stored per record inside `frozen_ids.jsonl`.
- **The canonical fixed-retrieval results use the 909-instance screen split.**
- **All 150 agent instances are nested within the screen split**, giving 1,200 ⊃ 909 ⊃ 150.

The split is a predeclared grouped assignment over (year, primary arXiv category) clusters — 13 of
116 clusters to development — so whole clusters fall on one side and the realised development share
is 24.3% rather than the nominal 20%. The rule is recorded in `construction_manifest.json` under
`predeclared_rules.splits`, and it was fixed before any arm score existed: the frozen IDs and their
split labels were written roughly 40 minutes before the first retrieval result file.

Metrics over all 1,200 were also computed and are retained for completeness; that superset also
passes the launch gate, so reporting the screen split is not a selection between a passing and a
failing variant. The two must not be mixed in one table.

Verify:

```bash
python tests/test_release_integrity.py     # asserts 291/909, the 1200 total, and the nesting
```

## 4. Statistical procedure

Paired over instances. Bootstrap confidence intervals with 10,000 resamples (unit = instance) and
paired sign-flip permutation tests with 10,000 permutations, seed `20260925`. Holm correction is
applied only within an explicitly declared confirmatory family, per model, never pooled across
models. Declared primary comparisons, fixed in code before inference: correct versus rewired
traversal on CSC recall@10 and on LSE recall@50.

## 5. Documented gaps

Nothing below is worked around or estimated.

| Gap | Status | What would be required |
|---|---|---|
| **Six-task academic benchmark results** | **not locatable in the source tree** | The aggregated metric files behind the six-task table could not be found; only per-condition ablation `config.json` files exist under `experiments/*/results/`. Releasing them needs the original evaluation harness outputs, which are not present. Only the ablation configs are shipped. |
| MAP and nDCG for CSC | not computed | a cheap CPU re-run of the screening aggregation; the set-valued screen reported set recall and set F1 instead |
| Semantic-retrieval arm for LSE | **does not exist** | a paper-level dense index; none exists in the source repository |
| Per-episode latency for LSE | present but not aggregated | `latency_ms` is in the episode rows; the aggregator emits no mean |
| Identical-answer rate across graph perturbations | not logged | recomputable from bundled rows: join correct and rewired on `instance_id`, compare `ranked` |
| Venue diversity for LSE | **not computable** | the corpus has no venue field; (year, category) clusters were used |
| Anytime quality curves (Task-A) | **collection failed** | models emitted the running top-5 in only 87 of 4,655 snapshots (1.9%). Reported missing, never imputed. The anytime *discovery* curve is valid because it is computed from observations rather than model output. |
| Qwen2.5-32B on LSE | **stopped, not failed** | reached 91/150 on one side of a paired comparison only; excluded from every number and not bundled |
| A2 reviewer-request task | **excluded** | 10 of 64 labels passed audit, so it supports no confirmatory claim; not included in this release |

## 6. Known construction bug, fixed before the reported run

The citation graph mixed arXiv identifier spellings: `cites` edges were version-stripped while about
20% of paragraph edges carried an explicit version (e.g. `…v2`), so 6,198 base identifiers existed as
two unconnected nodes. Masking on the literal source identifier therefore left a source paper
reachable through its versioned twin — the forbidden source-derived channel. Exposure was 45.4% of
CSC instances and 223 of 1,200 LSE sources.

All paper identity is now resolved on the version-stripped base identifier in every graph variant,
with degree preservation re-verified after the merge. Both workflows were rebuilt and re-audited:
CSC reports 0 violations over 2,204,525 checked candidates, LSE 0 future-dated candidates and 0
source-in-results with an independent re-audit of 15,000 returned identifiers. Superseded first-pass
artifacts were preserved in the source tree and used for nothing; they are not bundled here.

Measured size of the bug under the fixed-retrieval expansion policy was small (−0.003 R@50 on a
deliberately-broken control), but it was a real structural leak on the forbidden channel and its size
under an agent with a direct co-citation tool is not bounded by that figure.
