# Anonymity record

This release was built for double-blind review. It was assembled in a separate staging directory
from an explicit **allowlist** — nothing was copied unless named in the build script — and then
scanned and sanitized. This file records exactly what was searched for and what was changed.

## 1. Patterns searched

Across every staged file's **contents and filename**:

- personal names, initials, and account names
- email addresses (`[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`)
- institutional domains and affiliations
- GitHub usernames and repository URLs belonging to the authors
- OpenReview profile IDs, ORCID identifiers, personal websites
- acknowledgments and funding statements
- server, cluster, and host names; internal IP addresses
- user account names and absolute home/data paths (`(/home/|/data/|/mnt/)…`)
- symbolic links pointing outside the release

## 2. What was found and fixed

| Category | Found | Action |
|---|---|---|
| Absolute paths embedding a user account | 31 occurrences in 16 files | rewritten to `$PROJECT_ROOT`, `$DATA_ROOT`, `$RESULT_ROOT`, `$HF_CACHE`, `$LOCAL_MODELS`, `$SCRATCH` |
| Host name of the compute machine | 2 occurrences | replaced with `<host redacted>` |
| Internal IP addresses | 0 after filtering | — |
| Author names | **0** in the staged release | — |
| Email addresses | **0** | — |
| Author-owned GitHub URLs | **0** | — |
| OpenReview / ORCID / personal sites | **0** | — |
| Acknowledgments / funding text | **0** (not included) | — |
| Symbolic links | **0** | — |
| Files > 50 MB | **0** | largest is 8.0 MB |

Re-scanning after sanitization returns no matches for any identity pattern.

Reproduce the scan:

```bash
grep -rIhoE "(/data|/home|/mnt)/[A-Za-z0-9_]+|exx-[A-Za-z0-9-]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}" .
find . -type l -ls
find . -type f -size +50M -ls
```

## 3. What was deliberately **not** replaced

Blanket substitution would have damaged the science, so the following were reviewed in context and
kept:

- **GitHub URLs inside corpus-derived artifacts.** The Literature Set Expansion artifacts contain
  paper titles and abstracts from the evaluation corpus; some mention third-party repositories
  (`github.com/alibaba`, `github.com/amazon-research`, …). These are public scientific content, not
  author identity. They appear only under `frozen_artifacts/autoresearch/literature_set_expansion/`.
- **Model names and Hugging Face snapshot hashes.** Provenance, not identity.
- **arXiv identifiers**, dataset names, and venue names.
- **Run identifiers** such as `graphauto_20260925_035919_027d9a`. These encode a timestamp, not a
  person.

## 4. Git metadata

The release is published with **fresh history**: the original `.git` directory was not copied and no
original commits, branches, tags, or pull requests are preserved. The single release commit uses
neutral metadata set **locally for this repository only**:

```
user.name  = Anonymous Authors
user.email = anonymous@invalid.example
```

The author's global git configuration was not modified. Verify with:

```bash
git log --format=fuller --all
```

## 5. Licence

The release ships an **MIT licence with an anonymized copyright holder**:

```
Copyright (c) 2026 Anonymous Authors
```

This was an explicit author decision. The source repository's MIT licence names an individual, which
would have broken anonymity; the terms are unchanged and only the holder line is anonymized. **After
de-anonymization the copyright line should be restored to the real holder(s)** — the grant itself
(MIT) needs no change.

## 6. Residual risk

Two things reviewers could still infer, neither of which we can remove without damaging the artifact:

- **Timestamps.** Run identifiers and file dates reveal when the experiments were run.
- **Corpus composition.** The evaluation instances are drawn from public arXiv papers with a stated
  cutoff, which bounds the submission window.

Neither identifies an author, an institution, or a machine.
