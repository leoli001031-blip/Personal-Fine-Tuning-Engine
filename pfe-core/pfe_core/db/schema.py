"""SQLite schema for PFE local persistence."""

from __future__ import annotations

SCHEMA_VERSION = 3
DEFAULT_WORKSPACE = "user_default"
SIGNALS_TABLE = "signals"
SAMPLES_TABLE = "samples"
ADAPTER_VERSIONS_TABLE = "adapter_versions"

SIGNALS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    source_event_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    parent_event_id TEXT,
    source_event_ids TEXT NOT NULL DEFAULT '[]',
    event_chain_ids TEXT NOT NULL DEFAULT '[]',
    adapter_version TEXT,
    event_type TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    context TEXT,
    model_output TEXT,
    user_input TEXT,
    action_detail TEXT,
    user_action TEXT,
    lineage TEXT,
    metadata TEXT,
    processed INTEGER NOT NULL DEFAULT 0
);
""".strip()

SAMPLES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    sample_type TEXT NOT NULL,
    instruction TEXT NOT NULL,
    chosen TEXT NOT NULL,
    rejected TEXT,
    score REAL NOT NULL,
    source TEXT NOT NULL,
    source_event_ids TEXT,
    source_adapter_version TEXT,
    created_at DATETIME NOT NULL,
    used_in_version TEXT,
    metadata TEXT
);
""".strip()

ADAPTER_VERSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS adapter_versions (
    version TEXT PRIMARY KEY,
    workspace TEXT NOT NULL DEFAULT 'user_default',
    base_model TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    num_samples INTEGER NOT NULL DEFAULT 0,
    artifact_format TEXT NOT NULL,
    adapter_dir TEXT NOT NULL DEFAULT '',
    manifest_path TEXT,
    artifact_path TEXT,
    training_config TEXT,
    eval_report TEXT,
    metrics TEXT,
    promoted_at DATETIME,
    archived_at DATETIME,
    metadata TEXT
);
""".strip()

SIGNALS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(event_type);",
    "CREATE INDEX IF NOT EXISTS idx_signals_session ON signals(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_signals_request ON signals(request_id);",
    "CREATE INDEX IF NOT EXISTS idx_signals_adapter ON signals(adapter_version);",
]

SAMPLES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_samples_type ON samples(sample_type);",
    "CREATE INDEX IF NOT EXISTS idx_samples_score ON samples(score);",
    "CREATE INDEX IF NOT EXISTS idx_samples_used ON samples(used_in_version);",
    "CREATE INDEX IF NOT EXISTS idx_samples_source_adapter ON samples(source_adapter_version);",
]

ADAPTER_VERSIONS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_adapters_workspace ON adapter_versions(workspace);",
    "CREATE INDEX IF NOT EXISTS idx_adapters_state ON adapter_versions(state);",
    "CREATE INDEX IF NOT EXISTS idx_adapters_created_at ON adapter_versions(created_at);",
]


def adapter_versions_column_definitions() -> list[str]:
    return [
        "workspace TEXT NOT NULL DEFAULT 'user_default'",
        "num_samples INTEGER NOT NULL DEFAULT 0",
        "adapter_dir TEXT NOT NULL DEFAULT ''",
        "manifest_path TEXT",
        "artifact_path TEXT",
        "metrics TEXT",
        "promoted_at DATETIME",
        "archived_at DATETIME",
        "metadata TEXT",
    ]


def signals_column_definitions() -> list[str]:
    return [
        "parent_event_id TEXT",
        "source_event_ids TEXT NOT NULL DEFAULT '[]'",
        "event_chain_ids TEXT NOT NULL DEFAULT '[]'",
        "user_input TEXT",
        "action_detail TEXT",
        "lineage TEXT",
    ]


def build_schema_statements() -> list[str]:
    return [
        SIGNALS_TABLE_SQL,
        SAMPLES_TABLE_SQL,
        ADAPTER_VERSIONS_TABLE_SQL,
        *SIGNALS_INDEXES,
        *SAMPLES_INDEXES,
        *ADAPTER_VERSIONS_INDEXES,
    ]
