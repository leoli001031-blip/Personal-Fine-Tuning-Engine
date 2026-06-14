"""Optional doctor audit summary formatting."""

from __future__ import annotations


def _format_doctor_pii_compliance(workspace: str | None = None) -> str | None:
    del workspace
    try:
        from pfe_core.data_policy import audit_pii_exposure
        from pfe_core.storage import list_samples
    except Exception:
        return None
    try:
        samples = list_samples(limit=100)
        report = audit_pii_exposure(samples)
        if report.pii_detected_count == 0:
            return "pii compliance: clean"
        return f"pii compliance: detected={report.pii_detected_count} severity={report.severity} types={sorted(report.pii_types_found.keys())}"
    except Exception:
        return None


def _format_doctor_training_audit(workspace: str | None = None) -> str | None:
    del workspace
    try:
        from pfe_core.storage import list_samples
        from pfe_core.trainer.training_auditor import TrainingAuditor
    except Exception:
        return None
    try:
        samples = list_samples(limit=100)
        auditor = TrainingAuditor()
        report = auditor.audit(samples)
        if report.severity == "low" and not report.blocked:
            return "training audit: clean"
        return f"training audit: severity={report.severity} blocked={report.blocked} reasons={report.reasons}"
    except Exception:
        return None


def _format_doctor_signal_chain_integrity(workspace: str | None = None) -> str | None:
    del workspace
    try:
        from pfe_core.observability.trace import TraceStore
    except Exception:
        return None
    try:
        store = TraceStore()
        recent_ids = store.list_recent_signal_ids(limit=5)
        if not recent_ids:
            return "signal chain: no recent traces"
        complete_count = 0
        for signal_id in recent_ids:
            trace = store.load_signal_trace(signal_id)
            if trace is not None and trace.nodes:
                node_names = {node.node for node in trace.nodes}
                if "collect" in node_names:
                    complete_count += 1
        return f"signal chain: recent={len(recent_ids)} traced={complete_count}"
    except Exception:
        return None


__all__ = [
    "_format_doctor_pii_compliance",
    "_format_doctor_signal_chain_integrity",
    "_format_doctor_training_audit",
]
