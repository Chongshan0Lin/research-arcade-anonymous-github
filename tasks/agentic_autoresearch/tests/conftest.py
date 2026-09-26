"""Five deterministic toy instances (3 RT-Loc + 2 literature-evidence) used by the unit tests."""
import pytest

from tasks.agentic_autoresearch.environment import BM25, LiteratureEvidenceEnv, RTLocEnv
from tasks.agentic_autoresearch.build_graph_variants import PERTURB
from tasks.agentic_autoresearch.temporal_store import TemporalObject, TemporalStore

AS_OF = "2024-11-20 00:00:00"


def _toy_graph(inst_id, gold):
    nodes = {"P": dict(type="paper", text="")}
    texts = ["1 INTRODUCTION: we study graph dropout on citation networks",
             "2 METHOD: we define the effective degree and prove Theorem 1",
             "3 EXPERIMENTS: results on Cora and Citeseer are in Table 1",
             "4 ABLATION: we vary the dropout rate and report Figure 2",
             "5 CONCLUSION: future work covers heterophilic graphs"]
    edges = []
    for i, t in enumerate(texts, 1):
        nodes[f"p{i}"] = dict(type="paragraph", text=t, idx=i)
        edges.append(["P", f"p{i}", "has_paragraph"])
        if i > 1:
            edges.append([f"p{i-1}", f"p{i}", "next"])
    nodes["lTable|1"] = dict(type="label", text="Table 1")
    edges.append(["p3", "lTable|1", "para_mentions"])
    nodes["nr1"] = dict(type="note", ntype="official_review", time="2024-11-01 00:00:00",
                        text="The experiments in Table 1 are weak; please add a baseline comparison.")
    nodes["nr2"] = dict(type="note", ntype="author_response", time="2024-11-10 00:00:00",
                        text="We added the baseline comparison to the experiments section.")
    nodes["nfuture"] = dict(type="note", ntype="official_review", time="2025-01-01 00:00:00",
                            text="POST-CUTOFF NOTE THAT MUST NEVER BE OBSERVED")
    edges += [["nr1", "P", "reply_to"], ["nr2", "nr1", "reply_to"], ["nr1", "lTable|1", "note_mentions"],
              ["nfuture", "P", "reply_to"]]
    return dict(inst_id=inst_id, paper=f"paper_{inst_id}", nodes=nodes, edges=edges, gold=gold)


@pytest.fixture
def rtloc_instances():
    return [dict(instance_id=f"T{i}", task="rtloc", as_of=AS_OF, source_paper_id=f"paper_T{i}",
                 split="test", schema_version="1.0", gold_paragraph_idx=g,
                 gold_paragraph_ids=[f"p{x}" for x in g], n_paragraphs=5, n_input_notes=2,
                 query_text="Table 1 baseline comparison")
            for i, g in enumerate([[3], [3, 4], [2]], start=1)]


@pytest.fixture
def rtloc_env(rtloc_instances):
    import random
    base = {i["instance_id"]: _toy_graph(i["instance_id"], i["gold_paragraph_idx"])
            for i in rtloc_instances}
    graphs = {}
    for v in ("full", "flat", "untyped", "rewire", "type_shuffle"):
        graphs[v] = {k: dict(g, edges=PERTURB[v]([list(e) for e in g["edges"]], random.Random(0)))
                     for k, g in base.items()}
    store = TemporalStore()
    for g in base.values():
        for nid, n in g["nodes"].items():
            store.add(TemporalObject(nid, n["type"], n.get("time", "2000-01-01")))
    return RTLocEnv(graphs, store)


@pytest.fixture
def lit_instances():
    return [dict(instance_id="L1", task="literature_evidence", as_of=AS_OF, source_paper_id="2401.00001",
                 split="test", schema_version="1.0", gold_target_id="1609.02907",
                 masked_text="We build on graph convolutional networks [MASKED_CITATION] for node "
                             "classification on citation graphs.",
                 query_text="graph convolutional networks semi-supervised node classification"),
            dict(instance_id="L2", task="literature_evidence", as_of=AS_OF, source_paper_id="2401.00002",
                 split="test", schema_version="1.0", gold_target_id="1706.03762",
                 masked_text="The transformer architecture [MASKED_CITATION] uses self-attention.",
                 query_text="attention is all you need transformer self-attention")]


@pytest.fixture
def lit_env():
    import random
    papers = {
        "1609.02907": dict(title="Semi-Supervised Classification with Graph Convolutional Networks",
                           abstract="We present a scalable approach for semi-supervised learning on "
                                    "graph-structured data with convolutional networks.", date="2016-09-09"),
        "1706.03762": dict(title="Attention Is All You Need",
                           abstract="We propose the Transformer, based solely on attention mechanisms.",
                           date="2017-06-12"),
        "2401.00001": dict(title="A paper that cites GCN", abstract="node classification", date="2024-01-01"),
        "2401.00002": dict(title="A paper that cites Transformers", abstract="self attention", date="2024-01-02"),
        "2501.99999": dict(title="A future paper about graph convolutional networks and attention",
                           abstract="this must never be returned", date="2025-01-15"),
    }
    store = TemporalStore()
    for pid, p in papers.items():
        store.add(TemporalObject(f"paper:{pid}", "paper", p["date"]))
    edges = [["paper:2401.00001", "paper:1609.02907", "cites"],
             ["paper:2401.00002", "paper:1706.03762", "cites"],
             ["paper:1609.02907", "paper:1706.03762", "cites"],
             ["paper:2501.99999", "paper:1609.02907", "cites"]]
    adj = {}
    for v in ("full", "flat", "untyped", "rewire", "type_shuffle"):
        ev = PERTURB[v]([list(e) for e in edges], random.Random(0))
        d = {}
        for s, t, r in ev:
            d.setdefault(s, []).append((t, r, "out"))
            d.setdefault(t, []).append((s, r, "in"))
        adj[v] = d
    bm = BM25({pid: f"{p['title']} {p['abstract']}" for pid, p in papers.items()})
    return LiteratureEvidenceEnv(papers, bm, adj, store)
