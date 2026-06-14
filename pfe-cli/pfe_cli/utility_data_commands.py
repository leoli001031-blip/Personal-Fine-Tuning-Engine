"""Data management utility commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer


def register_data_commands(app: typer.Typer) -> None:
    data_app = typer.Typer(help="Data management and privacy compliance.")
    app.add_typer(data_app, name="data")

    @data_app.command("pii-check")
    def data_pii_check(
        fix: bool = typer.Option(False, "--fix", help="Auto-anonymize detected PII."),
        sample_file: Optional[str] = typer.Option(None, "--file", help="Specific sample file to check."),
        json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    ) -> None:
        """Check training samples for PII (Personally Identifiable Information)."""

        from pfe_core.anonymizer import AnonymizationConfig, Anonymizer
        from pfe_core.config import PFEConfig
        from pfe_core.pii_detector import PIIDetector

        config = PFEConfig.load()
        detector = PIIDetector(sensitivity="medium")
        anonymizer = Anonymizer(AnonymizationConfig(strategy="mask")) if fix else None

        samples_to_check = []
        if sample_file:
            file_path = Path(sample_file)
            if file_path.exists():
                samples_to_check.append(file_path)
        else:
            samples_dir = Path(config.workspace) / "training_samples"
            if samples_dir.exists():
                samples_to_check = list(samples_dir.glob("**/*.jsonl"))

        if not samples_to_check:
            typer.echo("No sample files found to check.", err=True)
            raise typer.Exit(code=1)

        results = []
        total_files = 0
        total_samples = 0
        samples_with_pii = 0
        total_findings = 0

        for file_path in samples_to_check:
            total_files += 1
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        sample = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    total_samples += 1
                    fields_to_check = ["instruction", "input", "output", "conversation"]
                    file_findings = []

                    for field in fields_to_check:
                        if field not in sample or not isinstance(sample[field], str):
                            continue
                        detection = detector.detect(sample[field])
                        if not detection.has_pii:
                            continue
                        file_findings.append(
                            {
                                "field": field,
                                "types": [pii_type.value for pii_type in detection.pii_types_found],
                                "count": len(detection.findings),
                            }
                        )
                        total_findings += len(detection.findings)

                        if anonymizer is not None:
                            sample[field] = anonymizer.anonymize(sample[field], detection)

                    if file_findings:
                        samples_with_pii += 1
                        results.append(
                            {
                                "file": str(file_path),
                                "sample_index": total_samples - 1,
                                "findings": file_findings,
                            }
                        )

        summary = {
            "total_files": total_files,
            "total_samples": total_samples,
            "samples_with_pii": samples_with_pii,
            "total_findings": total_findings,
            "files_checked": [str(path) for path in samples_to_check],
        }

        if json_output:
            typer.echo(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
        else:
            typer.echo("PII Check Results:")
            typer.echo(f"  Files checked: {total_files}")
            typer.echo(f"  Total samples: {total_samples}")
            typer.echo(f"  Samples with PII: {samples_with_pii}")
            typer.echo(f"  Total PII findings: {total_findings}")

            if results and not fix:
                typer.echo("\nRun with --fix to auto-anonymize detected PII.")
            elif fix:
                typer.echo("\nAnonymization applied to detected PII.")


__all__ = ["register_data_commands"]
