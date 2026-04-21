"""Unit tests for signal and version tracing."""

from pfe_core.observability.trace import (
    SignalTrace,
    VersionTrace,
    TraceNode,
    TraceStore,
    trace_signal,
    trace_version,
    record_signal_node,
    append_signal_to_version,
)


def test_signal_trace_add_node():
    st = SignalTrace(signal_id="sig-1")
    st.add_node("collect", "ok", {"foo": "bar"})
    assert len(st.nodes) == 1
    assert st.nodes[0].node == "collect"
    assert st.nodes[0].status == "ok"
    assert st.nodes[0].metadata == {"foo": "bar"}


def test_signal_trace_to_dict():
    st = SignalTrace(signal_id="sig-1")
    st.add_node("train", "ok")
    d = st.to_dict()
    assert d["signal_id"] == "sig-1"
    assert len(d["nodes"]) == 1
    assert d["nodes"][0]["node"] == "train"


def test_version_trace_to_dict():
    st = SignalTrace(signal_id="sig-1")
    st.add_node("collect", "ok")
    vt = VersionTrace(version="v1")
    vt.signal_traces.append(st)
    d = vt.to_dict()
    assert d["version"] == "v1"
    assert d["signal_count"] == 1


def test_trace_store_roundtrip(tmp_path):
    store = TraceStore(store_dir=tmp_path)
    st = SignalTrace(signal_id="sig-2")
    st.add_node("quality_score", "passed")
    store.save_signal_trace(st)

    loaded = store.load_signal_trace("sig-2")
    assert loaded is not None
    assert loaded.signal_id == "sig-2"
    assert len(loaded.nodes) == 1
    assert loaded.nodes[0].node == "quality_score"


def test_trace_store_version_roundtrip(tmp_path):
    store = TraceStore(store_dir=tmp_path)
    vt = VersionTrace(version="v2")
    st = SignalTrace(signal_id="sig-3")
    st.add_node("eval", "ok")
    vt.signal_traces.append(st)
    store.save_version_trace(vt)

    loaded = store.load_version_trace("v2")
    assert loaded is not None
    assert loaded.version == "v2"
    assert len(loaded.signal_traces) == 1
    assert loaded.signal_traces[0].nodes[0].node == "eval"


def test_trace_signal_creates_new():
    st = trace_signal("sig-new")
    assert st.signal_id == "sig-new"
    assert st.nodes == []


def test_trace_version_creates_new():
    vt = trace_version("v-new")
    assert vt.version == "v-new"
    assert vt.signal_traces == []


def test_record_signal_node_persists(tmp_path):
    store = TraceStore(store_dir=tmp_path)
    st = record_signal_node("sig-4", "promote", "completed", {"note": "done"})
    assert st.nodes[-1].node == "promote"


def test_append_signal_to_version(tmp_path):
    store = TraceStore(store_dir=tmp_path)
    vt = append_signal_to_version("v3", "sig-5")
    assert any(t.signal_id == "sig-5" for t in vt.signal_traces)


def test_trace_store_list_recent(tmp_path):
    store = TraceStore(store_dir=tmp_path)
    for i in range(3):
        st = SignalTrace(signal_id=f"sig-{i}")
        st.add_node("collect", "ok")
        store.save_signal_trace(st)
    recent = store.list_recent_signal_ids(limit=2)
    assert len(recent) == 2
