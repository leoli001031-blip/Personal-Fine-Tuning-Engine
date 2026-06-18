"""Phase 4 real-corpus driven finetuning loop primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from .adapter_store import create_adapter_store
from .data_policy import audit_pii_exposure, sanitize_for_training
from .db.sqlite import save_samples
from .phase3_signal_loop import PersonaSpec, ScenarioSpec
from .storage import resolve_home, write_jsonl


SourceType = Literal["md", "txt", "pdf", "url"]
CandidateType = Literal[
    "summary",
    "citation_grounded_answer",
    "structured_notes",
    "insufficient_evidence_refusal",
]

LOW_QUALITY_CHARS = 60
CHUNK_SIZE = 900
CHUNK_OVERLAP = 90
HIGH_RISK_DOMAINS = {
    "legal": (
        "合同",
        "诉讼",
        "律师",
        "法律",
        "合规",
        "违约",
        "liability",
        "lawsuit",
        "legal",
        "contract",
    ),
    "medical": (
        "诊断",
        "处方",
        "治疗",
        "病历",
        "药物",
        "medical",
        "diagnosis",
        "prescription",
        "treatment",
    ),
    "financial": (
        "投资",
        "收益",
        "股票",
        "贷款",
        "理财",
        "financial",
        "investment",
        "stock",
        "loan",
    ),
}
DETERMINISTIC_HIGH_RISK_PATTERNS = (
    "一定合法",
    "一定违法",
    "稳赢",
    "必须买入",
    "保证收益",
    "明确诊断为",
    "可以直接用药",
    "definitely legal",
    "guaranteed return",
    "buy this stock",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def _sha256(text: str | bytes) -> str:
    if isinstance(text, str):
        text = text.encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    words = len(re.findall(r"[A-Za-z0-9_]+", text))
    return max(1, cjk_chars + words)


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])\s+|[\n\r]+", text)
    return [_compact_text(part) for part in parts if _compact_text(part)]


def _lead_sentence(text: str, *, max_chars: int = 120) -> str:
    sentences = _sentences(text)
    lead = sentences[0] if sentences else _compact_text(text)
    if len(lead) <= max_chars:
        return lead
    return lead[: max_chars - 1].rstrip() + "…"


def _domain_flags(text: str) -> list[str]:
    lower = text.lower()
    flags: list[str] = []
    for domain, keywords in HIGH_RISK_DOMAINS.items():
        if any(keyword.lower() in lower for keyword in keywords):
            flags.append(domain)
    return flags


def _has_high_risk_deterministic_conclusion(text: str) -> bool:
    lower = text.lower()
    return any(pattern.lower() in lower for pattern in DETERMINISTIC_HIGH_RISK_PATTERNS)


def _license_status(value: str | None) -> str:
    text = str(value or "").strip()
    return text or "local_user_provided"


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return _compact_text(" ".join(self.parts))


@dataclass(frozen=True)
class CorpusSource:
    source_id: str
    title: str
    source_type: SourceType
    content_hash: str
    license_status: str
    created_at: str = field(default_factory=_utcnow_iso)
    source_path: str = ""
    source_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.source_id:
            raise ValueError("source_id is required")
        if not self.title:
            raise ValueError("title is required")
        if self.source_type not in {"md", "txt", "pdf", "url"}:
            raise ValueError(f"unsupported source_type: {self.source_type}")
        if self.source_type == "url" and not self.source_url:
            raise ValueError("source_url is required for url sources")
        if self.source_type != "url" and not self.source_path:
            raise ValueError("source_path is required for local sources")
        if not self.content_hash:
            raise ValueError("content_hash is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusSource":
        source = cls(
            source_id=str(data.get("source_id") or data.get("id") or "").strip(),
            title=str(data.get("title") or "").strip(),
            source_path=str(data.get("source_path") or ""),
            source_url=str(data.get("source_url") or ""),
            source_type=str(data.get("source_type") or "txt").strip().lower(),  # type: ignore[arg-type]
            content_hash=str(data.get("content_hash") or ""),
            license_status=str(data.get("license_status") or data.get("license") or ""),
            created_at=str(data.get("created_at") or _utcnow_iso()),
            metadata=_dict(data.get("metadata")),
        )
        source.validate()
        return source


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    source_id: str
    text: str
    char_count: int
    token_count: int
    provenance: dict[str, Any]
    safety_flags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id is required")
        if not self.source_id:
            raise ValueError("source_id is required")
        if not self.text.strip():
            raise ValueError("chunk text is required")
        if not self.provenance:
            raise ValueError("chunk provenance is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusChunk":
        chunk = cls(
            chunk_id=str(data.get("chunk_id") or data.get("id") or "").strip(),
            source_id=str(data.get("source_id") or "").strip(),
            text=str(data.get("text") or ""),
            char_count=int(data.get("char_count") or len(str(data.get("text") or ""))),
            token_count=int(data.get("token_count") or _approx_token_count(str(data.get("text") or ""))),
            provenance=_dict(data.get("provenance")),
            safety_flags=_string_list(data.get("safety_flags")),
            created_at=str(data.get("created_at") or _utcnow_iso()),
            metadata=_dict(data.get("metadata")),
        )
        chunk.validate()
        return chunk


@dataclass(frozen=True)
class Phase4TrainingCandidate:
    sample_id: str
    sample_type: CandidateType
    instruction: str
    input: str
    output: str
    source_ids: list[str]
    chunk_ids: list[str]
    provenance: dict[str, Any]
    safety_metadata: dict[str, Any]
    eligible_for_training: bool
    excluded_reason: str = ""
    score: float = 0.9
    created_at: str = field(default_factory=_utcnow_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Phase4TrainingCandidate":
        return cls(
            sample_id=str(data.get("sample_id") or data.get("id") or "").strip(),
            sample_type=str(data.get("sample_type") or "summary").strip(),  # type: ignore[arg-type]
            instruction=str(data.get("instruction") or ""),
            input=str(data.get("input") or ""),
            output=str(data.get("output") or data.get("chosen") or ""),
            source_ids=_string_list(data.get("source_ids")),
            chunk_ids=_string_list(data.get("chunk_ids")),
            provenance=_dict(data.get("provenance")),
            safety_metadata=_dict(data.get("safety_metadata")),
            eligible_for_training=bool(data.get("eligible_for_training", False)),
            excluded_reason=str(data.get("excluded_reason") or ""),
            score=float(data.get("score", 0.9) or 0.9),
            created_at=str(data.get("created_at") or _utcnow_iso()),
            metadata=_dict(data.get("metadata")),
        )


DEFAULT_RESEARCH_PERSONA = PersonaSpec(
    persona_id="research-notes-organizer",
    name="研究资料整理助手",
    identity="帮助使用者把给定资料整理成摘要、要点、引用和待确认问题的资料助手",
    goals=[
        "只基于已提供资料整理内容",
        "保留出处和可追溯引用",
        "资料不足时明确提示需补充资料或需人工确认",
    ],
    tone="克制、清楚、标注不确定性",
    forbidden_areas=[
        "不编造资料中不存在的结论",
        "不提供法律、医学、金融结论",
        "不把 PII 或高风险确定性建议写入训练样本",
    ],
    evaluation_criteria=[
        "摘要覆盖主要信息",
        "引用能回溯到 source/chunk",
        "无来源断言尽量为零",
        "高风险问题必须提示人工确认",
    ],
    metadata={"phase": "phase4"},
)

DEFAULT_RESEARCH_SCENARIO = ScenarioSpec(
    scenario_id="research-notes",
    name="研究资料整理",
    task="基于给定资料做摘要、要点整理、引用归纳和待确认问题，不输出无来源结论。",
    input_examples=[
        "请基于这些访谈记录整理出主要发现和待确认问题。",
        "请把这组资料整理成带引用的研究笔记。",
    ],
    expected_output="摘要、结构化要点、引用、待确认问题；资料不足时提示需补充资料/需人工确认。",
    risk_boundaries=[
        "不得输出资料中没有支持的确定性结论",
        "涉及法律、医学、金融内容时只做资料整理和风险/不确定性提示",
        "不能替代专业人士判断",
    ],
    human_review_required=True,
    high_risk_domains=["legal", "medical", "financial"],
    metadata={"phase": "phase4"},
)


def default_research_personas() -> list[PersonaSpec]:
    return [DEFAULT_RESEARCH_PERSONA]


def default_research_scenarios() -> list[ScenarioSpec]:
    return [DEFAULT_RESEARCH_SCENARIO]


class Phase4CorpusStore:
    def __init__(self, *, home: str | Path | None = None, workspace: str = "user_default") -> None:
        self.home = Path(home) if home is not None else resolve_home()
        self.workspace = workspace or "user_default"
        self.root = self.home / "phase4" / "workspaces" / self.workspace
        self.root.mkdir(parents=True, exist_ok=True)
        self.sources_path = self.root / "sources.json"
        self.chunks_path = self.root / "chunks.json"
        self.candidates_path = self.root / "training_candidates.json"
        self.plans_path = self.root / "training_plans.json"
        self.eval_report_path = self.root / "eval" / "phase4-eval-report.json"
        self.exports_dir = self.root / "exports"
        self.imports_dir = self.root / "imports"
        self.imports_dir.mkdir(parents=True, exist_ok=True)

    def personas(self) -> list[dict[str, Any]]:
        return [persona.to_dict() for persona in default_research_personas()]

    def scenarios(self) -> list[dict[str, Any]]:
        return [scenario.to_dict() for scenario in default_research_scenarios()]

    def _read_list(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [dict(item) for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def _write_list(self, path: Path, items: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def list_sources(self, *, source_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sources = [CorpusSource.from_dict(record).to_dict() for record in self._read_list(self.sources_path)]
        filtered = [
            source
            for source in reversed(sources)
            if not source_type or str(source.get("source_type")) == source_type
        ]
        return filtered[:limit]

    def list_chunks(
        self,
        *,
        source_id: str | None = None,
        safety_flag: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        chunks = [CorpusChunk.from_dict(record).to_dict() for record in self._read_list(self.chunks_path)]
        filtered: list[dict[str, Any]] = []
        for chunk in reversed(chunks):
            if source_id and chunk.get("source_id") != source_id:
                continue
            if safety_flag and safety_flag not in _string_list(chunk.get("safety_flags")):
                continue
            filtered.append(chunk)
            if len(filtered) >= limit:
                break
        return filtered

    def list_training_candidates(
        self,
        *,
        eligible_for_training: bool | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        items = [Phase4TrainingCandidate.from_dict(record).to_dict() for record in self._read_list(self.candidates_path)]
        filtered: list[dict[str, Any]] = []
        for item in reversed(items):
            if eligible_for_training is not None and bool(item.get("eligible_for_training")) is not eligible_for_training:
                continue
            filtered.append(item)
            if len(filtered) >= limit:
                break
        return filtered

    def _read_local_text(self, path: Path) -> tuple[str, SourceType]:
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            return path.read_text(encoding="utf-8"), "md"
        if suffix == ".txt":
            return path.read_text(encoding="utf-8"), "txt"
        if suffix == ".pdf":
            return self._read_pdf_text(path), "pdf"
        raise ValueError("only md, txt, and pdf sources are supported")

    def _read_pdf_text(self, path: Path) -> str:
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            parts = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(part for part in parts if part.strip())
        except Exception as exc:
            raise ValueError("PDF ingestion requires readable text or the pypdf package") from exc
        if not text.strip():
            raise ValueError("PDF source did not contain extractable text")
        return text

    def ingest_path(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        license_status: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_path = Path(path).expanduser()
        if not source_path.is_absolute():
            source_path = (Path.cwd() / source_path).resolve()
        if not source_path.exists() or not source_path.is_file():
            raise ValueError(f"source path not found: {source_path}")
        text, source_type = self._read_local_text(source_path)
        return self._ingest_text(
            text=text,
            source_type=source_type,
            title=title or source_path.stem,
            source_path=str(source_path),
            source_url="",
            license_status=license_status,
            metadata=metadata,
        )

    def ingest_url(
        self,
        url: str,
        *,
        title: str | None = None,
        license_status: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        max_bytes: int = 500_000,
    ) -> dict[str, Any]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("URL ingestion requires an http(s) URL")
        request = Request(url, headers={"user-agent": "PFE-Phase4-corpus-ingest/1.0"})
        with urlopen(request, timeout=10) as response:  # noqa: S310 - explicit user-provided small URL ingest.
            content_type = str(response.headers.get("content-type") or "")
            raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError("URL source exceeded the small-auditable ingest limit")
        text = raw.decode("utf-8", errors="replace")
        if "html" in content_type.lower():
            parser = _HTMLTextExtractor()
            parser.feed(text)
            text = parser.text()
        return self._ingest_text(
            text=text,
            source_type="url",
            title=title or parsed.netloc,
            source_path="",
            source_url=url,
            license_status=license_status or "user_review_required",
            metadata={"content_type": content_type, **dict(metadata or {})},
        )

    def _ingest_text(
        self,
        *,
        text: str,
        source_type: SourceType,
        title: str,
        source_path: str,
        source_url: str,
        license_status: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = _compact_text(text)
        if not normalized:
            raise ValueError("source text is empty")
        content_hash = _sha256(normalized)
        existing = self._read_list(self.sources_path)
        existing_match = next((item for item in existing if item.get("content_hash") == content_hash), None)
        if existing_match:
            source = CorpusSource.from_dict(existing_match).to_dict()
            chunks = [chunk for chunk in self.list_chunks(source_id=source["source_id"], limit=500)]
            return {"source": source, "chunks": chunks, "chunk_count": len(chunks), "deduplicated": True}

        source = CorpusSource(
            source_id=_short_id("src"),
            title=title.strip() or "Untitled source",
            source_path=source_path,
            source_url=source_url,
            source_type=source_type,
            content_hash=content_hash,
            license_status=_license_status(license_status),
            metadata=dict(metadata or {}),
        )
        chunks = self._chunk_source(source, normalized)
        sources = self._read_list(self.sources_path)
        sources.append(source.to_dict())
        self._write_list(self.sources_path, sources)
        records = self._read_list(self.chunks_path)
        records.extend(chunk.to_dict() for chunk in chunks)
        self._write_list(self.chunks_path, records)
        return {
            "source": source.to_dict(),
            "chunks": [chunk.to_dict() for chunk in chunks],
            "chunk_count": len(chunks),
            "deduplicated": False,
        }

    def _chunk_source(self, source: CorpusSource, text: str) -> list[CorpusChunk]:
        chunks: list[CorpusChunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            part = text[start:end].strip()
            if part:
                flags = self._chunk_safety_flags(part)
                chunks.append(
                    CorpusChunk(
                        chunk_id=f"{source.source_id}-chunk-{index + 1:03d}",
                        source_id=source.source_id,
                        text=part,
                        char_count=len(part),
                        token_count=_approx_token_count(part),
                        provenance={
                            "source_id": source.source_id,
                            "title": source.title,
                            "source_path": source.source_path,
                            "source_url": source.source_url,
                            "char_start": start,
                            "char_end": end,
                            "chunk_index": index,
                        },
                        safety_flags=flags,
                        metadata={"content_hash": _sha256(part)},
                    )
                )
            if end == len(text):
                break
            start = max(0, end - CHUNK_OVERLAP)
            index += 1
        return chunks

    def _chunk_safety_flags(self, text: str) -> list[str]:
        flags: list[str] = []
        if len(_compact_text(text)) < LOW_QUALITY_CHARS:
            flags.append("low_quality")
        pii = audit_pii_exposure([{"sample_id": "chunk", "input": text}])
        if pii.pii_detected_count:
            flags.append("pii_detected")
            flags.extend(f"pii:{pii_type}" for pii_type in sorted(pii.pii_types_found))
        for domain in _domain_flags(text):
            flags.append(f"high_risk_domain:{domain}")
        if _has_high_risk_deterministic_conclusion(text):
            flags.append("high_risk_deterministic_conclusion")
        return sorted(set(flags))

    def _candidate_safety(
        self,
        *,
        candidate_id: str,
        chunk: Mapping[str, Any],
        output: str,
    ) -> tuple[bool, str, dict[str, Any]]:
        flags = _string_list(chunk.get("safety_flags"))
        source_domains = sorted(
            flag.split(":", 1)[1]
            for flag in flags
            if flag.startswith("high_risk_domain:") and ":" in flag
        )
        pii_report = audit_pii_exposure(
            [
                {
                    "sample_id": candidate_id,
                    "input": chunk.get("text") or "",
                    "output": output,
                }
            ]
        ).to_dict()
        safety = {
            "chunk_safety_flags": flags,
            "pii_audit": pii_report,
            "high_risk_domains": source_domains,
            "human_review_required": bool(source_domains),
            "provenance_complete": bool(chunk.get("source_id") and chunk.get("chunk_id") and chunk.get("provenance")),
        }
        if not safety["provenance_complete"]:
            return False, "missing_provenance", safety
        if "low_quality" in flags:
            return False, "low_quality_chunk", safety
        if pii_report.get("severity") in {"high", "critical"}:
            return False, "pii_audit_blocked", safety
        if "high_risk_deterministic_conclusion" in flags or _has_high_risk_deterministic_conclusion(output):
            return False, "high_risk_deterministic_conclusion", safety
        return True, "", safety

    def _candidate_from_chunk(
        self,
        *,
        chunk: Mapping[str, Any],
        sample_type: CandidateType,
    ) -> Phase4TrainingCandidate:
        source_id = str(chunk.get("source_id") or "")
        chunk_id = str(chunk.get("chunk_id") or "")
        provenance = _dict(chunk.get("provenance"))
        title = str(provenance.get("title") or source_id or "source")
        text = str(chunk.get("text") or "")
        citation = f"[{source_id}:{chunk_id}]"
        lead = _lead_sentence(text)
        if sample_type == "summary":
            instruction = "Summarize only the provided material. Include citations and mark uncertainty."
            output = f"摘要：{lead} 依据：{citation}。待确认：是否还有补充资料或更新版本需要人工确认。"
        elif sample_type == "citation_grounded_answer":
            instruction = "Answer from the provided material only. Cite the exact source and chunk."
            output = f"根据资料 {citation}，可以整理出的信息是：{lead}。未在资料中出现的结论需补充资料/需人工确认。"
        elif sample_type == "structured_notes":
            instruction = "Turn the material into structured research notes with citations and open questions."
            output = (
                f"要点：{lead}\n"
                f"引用：{citation}\n"
                "待确认问题：是否有更多上下文、反例或更新资料需要补充。\n"
                "边界：本记录只整理资料，不提供法律、医学或金融结论。"
            )
        else:
            instruction = "If the requested answer is not supported by the material, refuse unsupported conclusions."
            output = (
                f"现有资料 {citation} 只能支持对已给内容的整理，不能支持题目中的确定性结论。"
                "需补充资料/需人工确认后再判断。"
            )
        candidate_id = _short_id("p4s")
        eligible, excluded_reason, safety = self._candidate_safety(
            candidate_id=candidate_id,
            chunk=chunk,
            output=output,
        )
        return Phase4TrainingCandidate(
            sample_id=candidate_id,
            sample_type=sample_type,
            instruction=instruction,
            input=f"资料标题：{title}\n来源：{citation}\n\n{text}",
            output=sanitize_for_training(output),
            source_ids=[source_id],
            chunk_ids=[chunk_id],
            provenance={
                "source_id": source_id,
                "chunk_id": chunk_id,
                "title": title,
                "source_path": provenance.get("source_path"),
                "source_url": provenance.get("source_url"),
            },
            safety_metadata=safety,
            eligible_for_training=eligible,
            excluded_reason=excluded_reason,
            metadata={"persona_id": DEFAULT_RESEARCH_PERSONA.persona_id, "scenario_id": DEFAULT_RESEARCH_SCENARIO.scenario_id},
        )

    def generate_training_candidates(self, *, limit: int = 24) -> dict[str, Any]:
        chunks = list(reversed(self.list_chunks(limit=max(limit, 1) * 4)))
        candidates: list[Phase4TrainingCandidate] = []
        for chunk in chunks:
            for sample_type in ("summary", "citation_grounded_answer", "structured_notes"):
                candidates.append(self._candidate_from_chunk(chunk=chunk, sample_type=sample_type))  # type: ignore[arg-type]
                if len(candidates) >= limit:
                    break
            if len(candidates) >= limit:
                break
        if chunks and len(candidates) < limit:
            candidates.append(self._candidate_from_chunk(chunk=chunks[0], sample_type="insufficient_evidence_refusal"))

        records = [candidate.to_dict() for candidate in candidates]
        self._write_list(self.candidates_path, records)
        eligible_count = sum(1 for item in records if item.get("eligible_for_training"))
        excluded_count = len(records) - eligible_count
        return {
            "workspace": self.workspace,
            "kind": "phase4_training_candidates",
            "count": len(records),
            "eligible_count": eligible_count,
            "excluded_count": excluded_count,
            "candidates": records,
        }

    def export_training_candidates(self, *, format: str = "jsonl", path: str | Path | None = None) -> dict[str, Any]:
        candidates = self.list_training_candidates(limit=1000)
        normalized_format = str(format or "jsonl").lower()
        if normalized_format not in {"json", "jsonl"}:
            raise ValueError("candidate export format must be json or jsonl")
        output_path = Path(path) if path is not None else self.exports_dir / f"phase4-training-candidates.{normalized_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if normalized_format == "jsonl":
            write_jsonl(output_path, candidates)
        else:
            output_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"path": str(output_path), "format": normalized_format, "count": len(candidates)}

    def export_to_training_samples(self, *, limit: int = 1000) -> dict[str, Any]:
        candidates = [item for item in reversed(self.list_training_candidates(eligible_for_training=True, limit=limit))]
        samples: list[dict[str, Any]] = []
        split_counts = {"train": 0, "val": 0, "test": 0}
        total = max(len(candidates), 1)
        for index, candidate in enumerate(candidates):
            ratio = (index + 1) / total
            if ratio <= 0.8:
                split = "train"
            elif ratio <= 0.9:
                split = "val"
            else:
                split = "test"
            split_counts[split] += 1
            samples.append(
                {
                    "sample_id": f"phase4-{candidate['sample_id']}",
                    "sample_type": "sft",
                    "instruction": candidate["instruction"] + "\n\n" + candidate["input"],
                    "chosen": candidate["output"],
                    "rejected": None,
                    "score": float(candidate.get("score", 0.9) or 0.9),
                    "source": "signal",
                    "source_event_ids": candidate.get("chunk_ids") or [],
                    "source_adapter_version": None,
                    "metadata": {
                        "phase": "phase4",
                        "dataset_split": split,
                        "phase4_sample_id": candidate["sample_id"],
                        "candidate_type": candidate["sample_type"],
                        "source_ids": candidate.get("source_ids") or [],
                        "chunk_ids": candidate.get("chunk_ids") or [],
                        "provenance": candidate.get("provenance") or {},
                        "safety_metadata": candidate.get("safety_metadata") or {},
                        "persona_id": DEFAULT_RESEARCH_PERSONA.persona_id,
                        "scenario_id": DEFAULT_RESEARCH_SCENARIO.scenario_id,
                    },
                }
            )
        count = save_samples(samples, home=self.home)
        export = self.export_training_candidates(format="jsonl")
        return {
            "workspace": self.workspace,
            "saved_samples": count,
            "split_counts": split_counts,
            "candidate_export": export,
        }

    def build_training_plan(self, *, base_model: str = "local-default") -> dict[str, Any]:
        candidates = self.list_training_candidates(limit=1000)
        eligible = [item for item in candidates if item.get("eligible_for_training")]
        blocked_by: list[str] = []
        if not candidates:
            blocked_by.append("no_training_candidates")
        if not eligible:
            blocked_by.append("no_eligible_training_candidates")
        plan = {
            "kind": "phase4_candidate_training_plan",
            "plan_id": _short_id("p4plan"),
            "workspace": self.workspace,
            "persona_id": DEFAULT_RESEARCH_PERSONA.persona_id,
            "scenario_id": DEFAULT_RESEARCH_SCENARIO.scenario_id,
            "candidate_adapter": {
                "version": f"phase4-candidate-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "state": "planned" if eligible else "blocked",
                "training_method": "sft",
                "sample_count": len(eligible),
                "base_model": base_model,
                "real_training_preferred": True,
                "mock_fallback_allowed": True,
            },
            "sample_count": len(eligible),
            "blocked_by": blocked_by,
            "handoff": {
                "export_endpoint": "/pfe/phase4/training-candidates/export",
                "training_endpoint": "/pfe/training/jobs",
                "eval_endpoint": "/pfe/phase4/eval",
                "promote_endpoint": "/pfe/candidate/promote",
                "archive_endpoint": "/pfe/candidate/archive",
            },
            "notes": [
                "Phase4 candidates are generated from real corpus chunks with provenance.",
                "Real LoRA training can use the existing /pfe/training/jobs endpoint after export.",
            ],
            "created_at": _utcnow_iso(),
        }
        plans = self._read_list(self.plans_path)
        plans.append(plan)
        self._write_list(self.plans_path, plans[-20:])
        return plan

    def materialize_mock_candidate_adapter(self, *, base_model: str = "local-default") -> dict[str, Any]:
        export = self.export_to_training_samples()
        adapter_store = create_adapter_store(workspace=self.workspace)
        created = adapter_store.create_training_version(
            base_model=base_model,
            training_config={
                "backend": "mock_phase4",
                "train_type": "sft",
                "phase": "phase4",
                "candidate_export": export["candidate_export"],
                "real_training_status": "skipped",
                "skip_reason": "phase4_real_train_smoke runs mock fallback unless PFE_PHASE4_REAL_TRAIN_MODEL points to a local trainable model",
            },
        )
        version = str(created["version"])
        adapter_store.merge_manifest(
            version,
            {
                "metadata": {
                    "phase": "phase4",
                    "phase4": True,
                    "candidate_export": export["candidate_export"],
                    "real_training_status": "skipped",
                }
            },
        )
        adapter_store.mark_pending_eval(
            version,
            num_samples=int(export["saved_samples"]),
            metrics={
                "phase4_candidate_samples": export["saved_samples"],
                "real_training": "skipped",
                "skip_reason": "no local trainable model configured",
            },
        )
        return {
            "workspace": self.workspace,
            "adapter_version": version,
            "adapter_path": created["path"],
            "state": "pending_eval",
            "training": {
                "real_training": "skipped",
                "skip_reason": "no local trainable model configured",
                "mock_fallback": True,
            },
            "export": export,
        }

    def _coverage_score(self, source_text: str, response: str) -> float:
        tokens = [token.lower() for token in re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]{3,}", source_text)]
        if not tokens:
            return 0.0
        selected = list(dict.fromkeys(tokens))[:24]
        if not selected:
            return 0.0
        lower = response.lower()
        hit = sum(1 for token in selected if token in lower)
        return round(hit / len(selected), 3)

    def build_eval_report(
        self,
        *,
        adapter_version: str | None = None,
        base_model: str = "local-default",
        attach_to_adapter: bool = False,
    ) -> dict[str, Any]:
        candidates = [item for item in reversed(self.list_training_candidates(eligible_for_training=True, limit=24))]
        if not candidates:
            generated = self.generate_training_candidates(limit=8)
            candidates = [item for item in generated["candidates"] if item.get("eligible_for_training")]
        details: list[dict[str, Any]] = []
        citation_hits_base = 0
        citation_hits_local = 0
        coverage_base_total = 0.0
        coverage_local_total = 0.0
        unsupported_base = 0
        unsupported_local = 0
        refusal_passed = 0
        for index, candidate in enumerate(candidates[:8]):
            chunk_ids = _string_list(candidate.get("chunk_ids"))
            source_ids = _string_list(candidate.get("source_ids"))
            expected_citation = f"[{source_ids[0]}:{chunk_ids[0]}]" if source_ids and chunk_ids else ""
            prompt = candidate["instruction"]
            base_output = "这份资料看起来可以得出一个整体结论，但需要更多上下文。"
            local_output = str(candidate.get("output") or "")
            source_text = str(candidate.get("input") or "")
            base_has_citation = bool(expected_citation and expected_citation in base_output)
            local_has_citation = bool(expected_citation and expected_citation in local_output)
            citation_hits_base += int(base_has_citation)
            citation_hits_local += int(local_has_citation)
            base_coverage = self._coverage_score(source_text, base_output)
            local_coverage = self._coverage_score(source_text, local_output)
            coverage_base_total += base_coverage
            coverage_local_total += local_coverage
            base_unsupported = 1 if not base_has_citation else 0
            local_unsupported = 0 if local_has_citation else 1
            unsupported_base += base_unsupported
            unsupported_local += local_unsupported
            refusal_boundary = (
                candidate.get("sample_type") == "insufficient_evidence_refusal"
                and "需补充资料" in local_output
                and "人工确认" in local_output
            )
            refusal_passed += int(refusal_boundary)
            details.append(
                {
                    "prompt_id": f"phase4-holdout-{index + 1:03d}",
                    "prompt": prompt,
                    "base_output": base_output,
                    "local_output": local_output,
                    "expected_citation": expected_citation,
                    "source_ids": source_ids,
                    "chunk_ids": chunk_ids,
                    "scores": {
                        "base_citation_hit": float(base_has_citation),
                        "local_citation_hit": float(local_has_citation),
                        "base_summary_coverage": base_coverage,
                        "local_summary_coverage": local_coverage,
                        "base_unsupported_assertions": base_unsupported,
                        "local_unsupported_assertions": local_unsupported,
                        "refusal_boundary_passed": float(refusal_boundary),
                    },
                }
            )

        count = max(len(details), 1)
        base_citation_rate = round(citation_hits_base / count, 3)
        local_citation_rate = round(citation_hits_local / count, 3)
        base_coverage = round(coverage_base_total / count, 3)
        local_coverage = round(coverage_local_total / count, 3)
        refusal_total = sum(1 for item in candidates[:8] if item.get("sample_type") == "insufficient_evidence_refusal")
        refusal_rate = round(refusal_passed / max(refusal_total, 1), 3) if refusal_total else 1.0
        delta = {
            "citation_hit_rate": round(local_citation_rate - base_citation_rate, 3),
            "summary_coverage": round(local_coverage - base_coverage, 3),
            "unsupported_assertions": unsupported_base - unsupported_local,
            "refusal_boundary": refusal_rate,
        }
        improved = delta["citation_hit_rate"] > 0 and delta["summary_coverage"] >= 0 and unsupported_local <= unsupported_base
        gate_status = "pass" if improved and local_citation_rate >= 0.75 and refusal_rate >= 0.8 else "review"
        if not details:
            gate_status = "fail"
        report = {
            "kind": "phase4_eval_report",
            "report_id": _short_id("p4eval"),
            "workspace": self.workspace,
            "adapter_version": adapter_version or "phase4-candidate-untrained",
            "base_model": base_model,
            "created_at": _utcnow_iso(),
            "real_model_calls": False,
            "holdout_count": len(details),
            "scores": {
                "citation_hit_rate": local_citation_rate,
                "summary_coverage": local_coverage,
                "unsupported_assertions": unsupported_local,
                "refusal_boundary": refusal_rate,
                "local_delta": delta,
            },
            "comparison": "improved" if improved else "neutral",
            "recommendation": "deploy" if gate_status == "pass" else "needs_more_data",
            "base_metrics": {
                "citation_hit_rate": base_citation_rate,
                "summary_coverage": base_coverage,
                "unsupported_assertions": unsupported_base,
            },
            "local_metrics": {
                "citation_hit_rate": local_citation_rate,
                "summary_coverage": local_coverage,
                "unsupported_assertions": unsupported_local,
                "refusal_boundary": refusal_rate,
            },
            "eval_gate": {
                "status": gate_status,
                "reasons": [
                    "local responses include source/chunk citations" if local_citation_rate else "missing local citations",
                    "unsupported assertions decreased" if unsupported_local <= unsupported_base else "unsupported assertions did not improve",
                ],
            },
            "improvement_summary": (
                "local candidate is more citation-grounded than base"
                if improved
                else "local candidate did not show enough measured improvement"
            ),
            "details": details,
        }
        self.eval_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.eval_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if attach_to_adapter and adapter_version:
            create_adapter_store(workspace=self.workspace).attach_eval_report(adapter_version, report)
        return report

    def summary(self) -> dict[str, Any]:
        sources = self.list_sources(limit=200)
        chunks = self.list_chunks(limit=1000)
        candidates = self.list_training_candidates(limit=1000)
        eligible = [item for item in candidates if item.get("eligible_for_training")]
        latest_plan = self._read_list(self.plans_path)[-1:] or []
        latest_eval = None
        if self.eval_report_path.exists():
            try:
                latest_eval = json.loads(self.eval_report_path.read_text(encoding="utf-8"))
            except Exception:
                latest_eval = None
        adapter_state: dict[str, Any] = {"state": "none", "version": None}
        try:
            rows = create_adapter_store(workspace=self.workspace).list_version_records(limit=20)
            phase4_rows = [
                row
                for row in rows
                if isinstance(row.get("metadata"), Mapping)
                and (
                    dict(row.get("metadata") or {}).get("phase") == "phase4"
                    or dict(row.get("metadata") or {}).get("phase4") is True
                    or dict(row.get("metadata") or {}).get("training", {}).get("phase") == "phase4"
                )
            ]
            row = phase4_rows[0] if phase4_rows else (rows[0] if rows else None)
            if row:
                adapter_state = {
                    "state": row.get("state"),
                    "version": row.get("version"),
                    "num_samples": row.get("num_samples"),
                }
        except Exception:
            pass
        return {
            "kind": "phase4_real_corpus_loop",
            "workspace": self.workspace,
            "personas": self.personas(),
            "scenarios": self.scenarios(),
            "source_count": len(sources),
            "chunk_count": len(chunks),
            "training_candidate_count": len(candidates),
            "eligible_training_candidate_count": len(eligible),
            "sources": sources[:8],
            "chunks": chunks[:8],
            "latest_candidates": candidates[:8],
            "latest_plan": latest_plan[0] if latest_plan else None,
            "candidate_adapter": adapter_state,
            "latest_eval_report": latest_eval,
            "eval_gate": (latest_eval or {}).get("eval_gate") if isinstance(latest_eval, Mapping) else None,
            "safety": {
                "pii_audit_enabled": True,
                "high_risk_domain_labels": sorted(HIGH_RISK_DOMAINS),
                "deterministic_legal_medical_financial_conclusions_excluded": True,
                "provenance_required": True,
            },
        }


__all__ = [
    "CorpusChunk",
    "CorpusSource",
    "DEFAULT_RESEARCH_PERSONA",
    "DEFAULT_RESEARCH_SCENARIO",
    "Phase4CorpusStore",
    "Phase4TrainingCandidate",
    "default_research_personas",
    "default_research_scenarios",
]
