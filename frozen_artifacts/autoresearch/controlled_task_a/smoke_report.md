# controlled_interaction_v1 smoke gate

Protocol: mandatory minimum research phase per condition ({'flat_controlled_agent': {'search': 1}, 'correct_graph_controlled_agent': {'search': 1, 'traverse': 1}, 'rewired_graph_controlled_agent': {'search': 1, 'traverse': 1}, 'compact_hybrid_controlled_agent': {'hybrid_search': 1}}).

Episodes: 160  |  **PASS**

| check | result | detail |
|---|---|---|
| 20/20 terminal rows per condition/model | PASS | {'qwen2.5-7b-instruct/compact_hybrid_controlled_agent': 20, 'qwen2.5-7b-instruct/correct_graph_controlled_agent': 20, 'qwen2.5-7b-instruct/flat_controlled_agent': 20, 'qwen2.5-7b-instruct/rewired_graph_controlled_agent': 20, 'qwen3-8b/compact_hybrid_controlled_agent': 20, 'qwen3-8b/correct_graph_controlled_agent': 20, 'qwen3-8b/flat_controlled_agent': 20, 'qwen3-8b/rewired_graph_controlled_agent': 20} |
| [qwen2.5-7b-instruct/compact_hybrid_controlled_agent] >=90% episodes with a successful tool call | PASS | 100% |
| [qwen2.5-7b-instruct/compact_hybrid_controlled_agent] mean successful tool calls >= 2.0 | PASS | 5.55 |
| [qwen2.5-7b-instruct/correct_graph_controlled_agent] >=90% episodes with a successful tool call | PASS | 95% |
| [qwen2.5-7b-instruct/correct_graph_controlled_agent] mean successful tool calls >= 2.0 | PASS | 5.90 |
| [qwen2.5-7b-instruct/flat_controlled_agent] >=90% episodes with a successful tool call | PASS | 100% |
| [qwen2.5-7b-instruct/flat_controlled_agent] mean successful tool calls >= 2.0 | PASS | 4.55 |
| [qwen2.5-7b-instruct/rewired_graph_controlled_agent] >=90% episodes with a successful tool call | PASS | 95% |
| [qwen2.5-7b-instruct/rewired_graph_controlled_agent] mean successful tool calls >= 2.0 | PASS | 5.70 |
| [qwen3-8b/compact_hybrid_controlled_agent] >=90% episodes with a successful tool call | PASS | 100% |
| [qwen3-8b/compact_hybrid_controlled_agent] mean successful tool calls >= 2.0 | PASS | 4.95 |
| [qwen3-8b/correct_graph_controlled_agent] >=90% episodes with a successful tool call | PASS | 100% |
| [qwen3-8b/correct_graph_controlled_agent] mean successful tool calls >= 2.0 | PASS | 3.50 |
| [qwen3-8b/flat_controlled_agent] >=90% episodes with a successful tool call | PASS | 100% |
| [qwen3-8b/flat_controlled_agent] mean successful tool calls >= 2.0 | PASS | 4.50 |
| [qwen3-8b/rewired_graph_controlled_agent] >=90% episodes with a successful tool call | PASS | 100% |
| [qwen3-8b/rewired_graph_controlled_agent] mean successful tool calls >= 2.0 | PASS | 4.20 |
| [qwen2.5-7b-instruct/compact_hybrid_controlled_agent] hybrid_search uptake >= 90% | PASS | 100% |
| [qwen2.5-7b-instruct/correct_graph_controlled_agent] search uptake >= 90% | PASS | 95% |
| [qwen2.5-7b-instruct/correct_graph_controlled_agent] traverse uptake >= 90% | PASS | 95% |
| [qwen2.5-7b-instruct/flat_controlled_agent] search uptake >= 90% | PASS | 100% |
| [qwen2.5-7b-instruct/rewired_graph_controlled_agent] search uptake >= 90% | PASS | 95% |
| [qwen2.5-7b-instruct/rewired_graph_controlled_agent] traverse uptake >= 90% | PASS | 95% |
| [qwen3-8b/compact_hybrid_controlled_agent] hybrid_search uptake >= 90% | PASS | 100% |
| [qwen3-8b/correct_graph_controlled_agent] search uptake >= 90% | PASS | 100% |
| [qwen3-8b/correct_graph_controlled_agent] traverse uptake >= 90% | PASS | 100% |
| [qwen3-8b/flat_controlled_agent] search uptake >= 90% | PASS | 100% |
| [qwen3-8b/rewired_graph_controlled_agent] search uptake >= 90% | PASS | 100% |
| [qwen3-8b/rewired_graph_controlled_agent] traverse uptake >= 90% | PASS | 100% |
| action-parser validity >= 95% | PASS | 96.9% |
| tool observations appear in subsequent context | PASS | 0 episodes with lost observations |
| [qwen2.5-7b-instruct] correct/rewired observed pools not universally identical | PASS | 19/20 instances differ |
| [qwen3-8b] correct/rewired observed pools not universally identical | PASS | 20/20 instances differ |
| correct/rewired graph hashes differ | PASS | {'correct_graph_controlled_agent': {'c33f0afd190858f3'}, 'rewired_graph_controlled_agent': {'0232ea9a6a878502'}} |
| correct/rewired degree summaries match | PASS | {'correct_graph_controlled_agent': {'1ee5994e832b330b'}, 'rewired_graph_controlled_agent': {'1ee5994e832b330b'}} |
| no graph information reaches flat condition | PASS | 0 violations |
| correct/rewired prompt+schema hashes identical per instance | PASS | 20 distinct |
| cited evidence joins to observed records | PASS | 0 episodes cite unobserved ids |
| zero temporal-leakage violations | PASS | 0 violations |
| budgets enforced and truncation logged | PASS | 0 budget overruns |
| matched_pool_hash reproduces observed pool | PASS | recomputed from raw rows |
