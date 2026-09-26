# Paper-to-artifact manifest

Machine-readable twin: `manifest.json`. **No GPU is required for any row.**

| Claim | Artifact | Script | Runtime | GPU | Status |
|---|---|---|---|---|---|
| Six academic benchmark results | `NOT LOCATABLE in source tree (only experiments/*/results/config.json abl` | `-` | - | No | MISSING - documented in REPRODUCIBILITY.md section 5 |
| Fixed Citation Set Completion retrieval | `frozen_artifacts/autoresearch/evidence_bundle_completion/retrieval_resul` | `tasks.agentic_autoresearch.appendix_tables` | 0.1 s | No | reproduced |
| Fixed Literature Set Expansion retrieval | `frozen_artifacts/autoresearch/literature_set_expansion/retrieval_results` | `tasks.agentic_autoresearch.appendix_tables` | 0.1 s | No | reproduced |
| Agent correct vs rewired, CSC | `frozen_artifacts/autoresearch/evidence_bundle_completion/agent_results/e` | `tasks.agentic_autoresearch.ebc_aggregate` | 5.8 s | No | reproduced |
| Agent correct vs rewired, LSE | `frozen_artifacts/autoresearch/literature_set_expansion/agent_results/epi` | `tasks.agentic_autoresearch.litexp_aggregate` | 1.9 s | No | reproduced |
| Candidate-access mechanism (graph-only discovery) | `…/evidence_bundle_completion/agent_results/{candidate_funnel,graph_only_` | `tasks.agentic_autoresearch.ebc_aggregate` | 5.8 s | No | reproduced |
| Matched-pool replay comparisons | `…/agent_results/matched_pools/*.jsonl (2400 replays)` | `ebc_aggregate / litexp_aggregate / controlled_` | 0.4 s | No | reproduced |
| Controlled Task-A, correct vs rewired | `frozen_artifacts/autoresearch/controlled_task_a/episodes/*/*.jsonl` | `tasks.agentic_autoresearch.controlled_aggregat` | 20.6 s | No | reproduced |
| Temporal / source-edge leakage audit | `…/temporal_audit.csv.gz, temporal_audit_summary.json, graph_identity_aud` | `shipped audit outputs (audits were run at cons` | <1 s | No | reproduced |
| LSE 1200 -> 291 dev / 909 screen split, agents nested in screen | `frozen_artifacts/autoresearch/literature_set_expansion/frozen_ids.jsonl ` | `tests/test_release_integrity.py` | 3.1 s | No | reproduced |
| Main-text value verification (31 checks) | `results/diagnostics/maintext_verification.json` | `tasks.agentic_autoresearch.appendix_tables (ve` | <1 s | No | reproduced |

## Expected values and verification commands

### Six academic benchmark results
- **Paper location:** main-text benchmark table
- **Artifact:** `NOT LOCATABLE in source tree (only experiments/*/results/config.json ablation configs)`
- **Script:** `n/a`
- **Expected:** n/a
- **Verify:** n/a
- **Runtime:** n/a · **GPU:** No · **Status:** MISSING - documented in REPRODUCIBILITY.md section 5

### Fixed Citation Set Completion retrieval
- **Paper location:** appendix complete retrieval table
- **Artifact:** `frozen_artifacts/autoresearch/evidence_bundle_completion/retrieval_results.csv`
- **Script:** `tasks.agentic_autoresearch.appendix_tables`
- **Expected:** results/paper_tables/t2_fixed_retrieval.csv; hybrid_correct R@20=0.6303, hybrid_rewired=0.3396
- **Verify:** `python -m tasks.agentic_autoresearch.appendix_tables --root frozen_artifacts/autoresearch`
- **Runtime:** 0.1 s · **GPU:** No · **Status:** reproduced

### Fixed Literature Set Expansion retrieval
- **Paper location:** appendix complete retrieval table
- **Artifact:** `frozen_artifacts/autoresearch/literature_set_expansion/retrieval_results.csv (subset=screen, n=909)`
- **Script:** `tasks.agentic_autoresearch.appendix_tables`
- **Expected:** hybrid_correct R@50=0.4266, hybrid_rewired=0.2516, diff +0.1750 CI [+0.1644,+0.1858]
- **Verify:** `python -m tasks.agentic_autoresearch.appendix_tables --root frozen_artifacts/autoresearch`
- **Runtime:** 0.1 s · **GPU:** No · **Status:** reproduced

### Agent correct vs rewired, CSC
- **Paper location:** main-text agent table
- **Artifact:** `frozen_artifacts/autoresearch/evidence_bundle_completion/agent_results/episodes/*/*.jsonl`
- **Script:** `tasks.agentic_autoresearch.ebc_aggregate`
- **Expected:** R@10 diff: 7B +0.0889 [+0.0428,+0.1378]; 8B +0.1017 [+0.0528,+0.1522]; 32B +0.1028 [+0.0550,+0.1533]
- **Verify:** `python -m tasks.agentic_autoresearch.ebc_aggregate --root frozen_artifacts/autoresearch/evidence_bundle_completion/agent_results`
- **Runtime:** 5.8 s · **GPU:** No · **Status:** reproduced

### Agent correct vs rewired, LSE
- **Paper location:** main-text agent table
- **Artifact:** `frozen_artifacts/autoresearch/literature_set_expansion/agent_results/episodes/*/*.jsonl`
- **Script:** `tasks.agentic_autoresearch.litexp_aggregate`
- **Expected:** R@50 diff: 7B +0.0660 [+0.0387,+0.0934]; 8B +0.0644 [+0.0462,+0.0843]
- **Verify:** `python -m tasks.agentic_autoresearch.litexp_aggregate --root frozen_artifacts/autoresearch/literature_set_expansion/agent_results`
- **Runtime:** 1.9 s · **GPU:** No · **Status:** reproduced

### Candidate-access mechanism (graph-only discovery)
- **Paper location:** appendix diagnostics table
- **Artifact:** `…/evidence_bundle_completion/agent_results/{candidate_funnel,graph_only_discovery}.csv`
- **Script:** `tasks.agentic_autoresearch.ebc_aggregate`
- **Expected:** Qwen3-8B graph-only targets 0.312 correct vs 0.081 rewired; episodes 0.433 vs 0.127
- **Verify:** `column -t -s, results/paper_tables/t4_access_and_tools.csv`
- **Runtime:** 5.8 s · **GPU:** No · **Status:** reproduced

### Matched-pool replay comparisons
- **Paper location:** appendix agent comparisons table
- **Artifact:** `…/agent_results/matched_pools/*.jsonl (2400 replays)`
- **Script:** `ebc_aggregate / litexp_aggregate / controlled_aggregate`
- **Expected:** no agent beats its own replay after Holm; all pool hashes reproduce
- **Verify:** `python -m tasks.agentic_autoresearch.graph_autoresearch_report --root frozen_artifacts/autoresearch`
- **Runtime:** 0.4 s · **GPU:** No · **Status:** reproduced

### Controlled Task-A, correct vs rewired
- **Paper location:** appendix controlled-interaction section
- **Artifact:** `frozen_artifacts/autoresearch/controlled_task_a/episodes/*/*.jsonl`
- **Script:** `tasks.agentic_autoresearch.controlled_aggregate`
- **Expected:** discovery +0.125 / +0.080 significant; Top-1 not significant after Holm
- **Verify:** `python -m tasks.agentic_autoresearch.controlled_aggregate --root frozen_artifacts/autoresearch/controlled_task_a`
- **Runtime:** 20.6 s · **GPU:** No · **Status:** reproduced

### Temporal / source-edge leakage audit
- **Paper location:** appendix leakage section
- **Artifact:** `…/temporal_audit.csv.gz, temporal_audit_summary.json, graph_identity_audit.json, independent_leakage_verification.json`
- **Script:** `shipped audit outputs (audits were run at construction time)`
- **Expected:** 0 violations; CSC 2,204,525 candidates checked; LSE 15,000 re-audited
- **Verify:** `cat frozen_artifacts/autoresearch/evidence_bundle_completion/temporal_audit_summary.json`
- **Runtime:** <1 s · **GPU:** No · **Status:** reproduced

### LSE 1200 -> 291 dev / 909 screen split, agents nested in screen
- **Paper location:** appendix instance-construction section
- **Artifact:** `frozen_artifacts/autoresearch/literature_set_expansion/frozen_ids.jsonl (per-record split label)`
- **Script:** `tests/test_release_integrity.py`
- **Expected:** splits == {dev:291, screen:909}; all 150 agent ids have split==screen; no instance removed after freezing
- **Verify:** `python tests/test_release_integrity.py`
- **Runtime:** 3.1 s · **GPU:** No · **Status:** reproduced

### Main-text value verification (31 checks)
- **Paper location:** all AutoResearch headline values
- **Artifact:** `results/diagnostics/maintext_verification.json`
- **Script:** `tasks.agentic_autoresearch.appendix_tables (verification recorded at audit time)`
- **Expected:** all_reproduced = true, n_checked = 31
- **Verify:** `python -c "import json;d=json.load(open('results/diagnostics/maintext_verification.json'));print(d['all_reproduced'],d['n_checked'])"`
- **Runtime:** <1 s · **GPU:** No · **Status:** reproduced

## The LSE split, stated explicitly

- The 1,200 frozen LSE instances consist of **291 development** and **909 held-out screen** instances.
- **No frozen instances were removed after splitting** (291 + 909 = 1,200).
- The canonical fixed-retrieval results use the **909-instance screen split**.
- **All 150 agent instances are nested within the screen split** (1,200 ⊃ 909 ⊃ 150).

