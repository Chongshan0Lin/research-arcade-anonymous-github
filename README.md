# ResearchArcade

This anonymous repository contains the code and frozen artifacts required to
reproduce the main results reported in the accompanying double-blind
submission.

**Every table in the paper's AutoResearch sections can be regenerated on a laptop CPU in under a
minute from the frozen outputs in this repository. No GPU, no model download, and no inference is
required.** The expensive agent rollouts were run once; their per-episode logs are bundled here and
every reported number is recomputed from those logs.

---

## 1. Overview

The repository covers two evidence-discovery workflows and a controlled interaction study:

| Component | What it measures | Frozen evaluation size |
|---|---|---|
| **Citation Set Completion (CSC)** | recover the remaining citations of a paragraph from one revealed anchor | 5,000 fixed-retrieval / 150 agent |
| **Literature Set Expansion (LSE)** | build a related-work set from a title and abstract alone | 909 fixed-retrieval / 150 agent |
| **Controlled interaction (Task-A)** | single masked citation, correct vs rewired graph under a controlled agent loop | 200 |
| **Scaling slice** | tool uptake by model scale under two interfaces | 50 |

The central manipulation throughout is **correct versus degree-preserving rewired connectivity**:
identical nodes, text, metadata, degrees, prompts, tool schemas and budgets, with only the backing
adjacency changed. That equality is asserted at run time by a prompt/schema hash recorded in every
episode row.

On disk, CSC is named `evidence_bundle_completion` and LSE is `literature_set_expansion`.

## 2. Directory structure

```
.
├── README.md                    this file
├── REPRODUCIBILITY.md           per-table reproduction, runtimes, what is and is not included
├── ANONYMITY.md                 what was scrubbed, and the licence situation
├── MANIFEST.md / manifest.json  claim -> artifact -> script -> command mapping
├── requirements.txt
├── tasks/agentic_autoresearch/  pipeline code (construction, retrieval, agents, aggregation)
├── schemas/                     graph schema definitions
├── frozen_artifacts/
│   ├── autoresearch/            the canonical run: episodes, manifests, audits, reports
│   └── academic_benchmarks/     ablation configs (see REPRODUCIBILITY.md for a documented gap)
├── results/
│   ├── paper_tables/            CSV behind every appendix table
│   └── diagnostics/             main-text verification + missing-quantity records
├── examples/                    the paired example trace used in the appendix
└── tests/                       lightweight checks (no downloads, no models)
```

## 3. Environment setup

Python 3.10 or newer. The reproduction path needs only the standard library plus the packages in
`requirements.txt`; `numpy` is the only hard dependency for the statistics.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras (`psycopg2`, `rank_bm25`, `torch`, `vllm`) are needed only to rebuild instances from
a database or to run new inference. Neither is required for anything in Section 4.

## 4. Quick start — regenerate the paper tables (CPU only, < 1 minute total)

From the repository root:

```bash
# 1. controlled interaction study, Task-A  (~21 s)
python -m tasks.agentic_autoresearch.controlled_aggregate \
  --root frozen_artifacts/autoresearch/controlled_task_a

# 2. Citation Set Completion agents  (~6 s)
python -m tasks.agentic_autoresearch.ebc_aggregate \
  --root frozen_artifacts/autoresearch/evidence_bundle_completion/agent_results

# 3. Literature Set Expansion agents  (~2 s)
python -m tasks.agentic_autoresearch.litexp_aggregate \
  --root frozen_artifacts/autoresearch/literature_set_expansion/agent_results

# 4. every appendix table as CSV  (~0.1 s)
python -m tasks.agentic_autoresearch.appendix_tables \
  --root frozen_artifacts/autoresearch

# 5. canonical report + integrity verification  (~0.4 s)
python -m tasks.agentic_autoresearch.graph_autoresearch_report \
  --root frozen_artifacts/autoresearch
```

All five are CPU-only, read only files in this repository, and make no network calls. Measured wall
clock on one core is given above; total under 30 seconds.

Step 5 prints an integrity checklist and exits non-zero if any graded check fails. In this bundle it
reports **13/13 graded checks passed, 1 skipped** when run inside this git repository. The single
skip is honest and expected: the *previous* (natural-action) run is not bundled here, so that check
is not applicable. It is reported as `SKIP` and never counted as a pass. (Outside a git work tree
`git diff --check` is also skipped, giving 12/12 with 2 skipped.)

## 5. Paper table → script → artifact

See `MANIFEST.md` for the full mapping with verification commands. Summary:

| Paper content | Script | Frozen artifact |
|---|---|---|
| Fixed CSC retrieval | `appendix_tables` | `evidence_bundle_completion/retrieval_results.csv` |
| Fixed LSE retrieval | `appendix_tables` | `literature_set_expansion/retrieval_results.csv` |
| Agent correct vs rewired | `ebc_aggregate`, `litexp_aggregate` | `*/agent_results/episodes/*.jsonl` |
| Candidate-access mechanism | `ebc_aggregate` | `*/agent_results/candidate_funnel.csv`, `graph_only_discovery.csv` |
| Matched-pool comparisons | `ebc_aggregate`, `controlled_aggregate` | `*/agent_results/matched_pools/*.jsonl` |
| Temporal / leakage audit | shipped audit outputs | `*/temporal_audit.csv.gz`, `graph_identity_audit.json` |
| Controlled Task-A | `controlled_aggregate` | `controlled_task_a/episodes/*.jsonl` |

## 6. Reproducing the fixed-retrieval tables

The fixed-retrieval arms (BM25, graph expansion, hybrid RRF, and the rewired / random-anchor /
degree-matched-anchor controls) were computed once over the frozen instance sets. Their per-arm
results and paired bootstrap comparisons are shipped directly:

```bash
column -t -s, results/paper_tables/t2_fixed_retrieval.csv
column -t -s, results/paper_tables/t2b_fixed_retrieval_comparisons.csv
```

Recomputing these **from the corpus** requires the ArXiv corpus and graph variants, which are not
redistributable here (Section 9). The per-instance retrieval records are included as
`*/retrieval_per_instance.jsonl.gz` so the aggregate numbers can be independently re-derived without
the corpus.

## 7. Reproducing the agent tables from frozen trajectories

Every agent episode is stored as one JSON line containing the actions emitted, the tool results
returned, the ordered candidates observed, the final answer, and the per-episode metrics. The
aggregators in Section 4 recompute all reported numbers — absolute performance, paired bootstrap
confidence intervals (10,000 resamples), paired permutation tests, and Holm-adjusted p-values —
from those rows alone. Re-running them reproduces the paper values exactly.

## 8. Integrity and leakage audits

```bash
# integrity checklist (identical IDs across paired conditions, no duplicate episode keys,
# graph hashes differ, degree preservation, no out-of-interface tool executed, ...)
python -m tasks.agentic_autoresearch.graph_autoresearch_report --root frozen_artifacts/autoresearch

# main-text value verification (31 checks)
python -c "import json;d=json.load(open('results/diagnostics/maintext_verification.json'));print(d['all_reproduced'], d['n_checked'])"

# leakage-audit summaries
cat frozen_artifacts/autoresearch/evidence_bundle_completion/temporal_audit_summary.json
cat frozen_artifacts/autoresearch/literature_set_expansion/independent_leakage_verification.json
cat frozen_artifacts/autoresearch/literature_set_expansion/graph_identity_audit.json
```

The full per-candidate audit rows are shipped gzipped (`temporal_audit.csv.gz`) because they are
large and no table script reads them.

## 9. Large data that is not included

| Not included | Why | Consequence |
|---|---|---|
| ArXiv corpus (65,328 papers, text + abstracts) | size and redistribution | instance construction cannot be re-run from scratch; all frozen instances and per-instance retrieval records are included |
| Citation graph variants (989,123 edges each) | size | graph hashes are recorded so a rebuilt graph can be checked for identity |
| Model checkpoints | size and licensing | not needed; no inference is required |
| Response caches | bulk model text, not needed to recompute any number | excluded |
| Superseded first-pass artifacts | an early construction bug was fixed and everything rebuilt; the superseded outputs were never used for any reported number | the bug and its measured size are documented in `ANONYMITY.md` and the audit reports |

## 10. Runtime and hardware

Everything in Sections 4, 7 and 8 runs on a single CPU core in well under a minute and needs roughly
300 MB of RAM. No step in the documented reproduction path uses a GPU, downloads anything, or
touches a database. The original agent rollouts used 6 NVIDIA A6000 GPUs for about 4.5 hours; that
cost does **not** need to be repaid to verify the reported tables.

## 11. Licence

MIT, with an anonymized copyright holder for the review period (`Copyright (c) 2026 Anonymous
Authors`). See `LICENSE` and Section 5 of `ANONYMITY.md`.

## 12. Why inference is not needed

The expensive part of this work is generating agent trajectories. Those are one-off and are bundled
here as append-only JSONL. Every statistic in the paper is a deterministic function of those rows,
so verification is a pure recomputation. The aggregators are the same modules that produced the
paper numbers, not a reimplementation.
