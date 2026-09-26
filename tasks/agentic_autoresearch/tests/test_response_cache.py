"""The determinism invariant: identical transcripts must reuse identical model responses."""
from tasks.agentic_autoresearch.policies import ResponseCache, VLLMPolicy


class FakePolicy(VLLMPolicy):
    """Counts remote calls and returns a different answer every time it actually hits the server."""

    def __init__(self, cache):
        super().__init__("m", base_url="http://127.0.0.1:1/v1", cache=cache)
        self.calls = 0

    def _chat_remote(self, messages):
        self.calls += 1
        return f"response #{self.calls}"


def test_identical_transcripts_reuse_identical_responses(tmp_path):
    cache = ResponseCache(tmp_path / "cache.jsonl")
    msgs = [{"role": "user", "content": "hello"}]
    a, b = FakePolicy(cache), FakePolicy(cache)
    first = a._chat(msgs)
    second = b._chat([dict(m) for m in msgs])       # same transcript, different policy instance
    assert first == second
    assert a.calls == 1 and b.calls == 0            # second call served from cache
    assert cache.hits == 1 and cache.misses == 1


def test_different_transcripts_are_not_shared(tmp_path):
    cache = ResponseCache(tmp_path / "cache.jsonl")
    p = FakePolicy(cache)
    assert p._chat([{"role": "user", "content": "a"}]) != p._chat([{"role": "user", "content": "b"}])
    assert p.calls == 2


def test_cache_persists_across_processes(tmp_path):
    path = tmp_path / "cache.jsonl"
    p1 = FakePolicy(ResponseCache(path))
    v1 = p1._chat([{"role": "user", "content": "x"}])
    p2 = FakePolicy(ResponseCache(path))            # reloaded from disk
    assert p2._chat([{"role": "user", "content": "x"}]) == v1
    assert p2.calls == 0
