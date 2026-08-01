"""Verify that the delivery package is complete, frozen and credential-safe."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "PACKAGE_MANIFEST.json"

REQUIRED_PATHS = [
    "cli.py",
    "config.py",
    ".env.example",
    "agents/intention_agent.py",
    "agents/orchestration_agent.py",
    "context/memory_manager.py",
    ".claude/skills/ask-question/script/agent.py",
    ".claude/skills/plan-trip/script/agent.py",
    ".claude/skills/preference/script/agent.py",
    ".claude/skills/memory-query/script/agent.py",
    ".claude/skills/query-info/script/agent.py",
    ".claude/skills/event-collection/script/agent.py",
    "data/models/bge-small-zh-v1.5/model.safetensors",
    "evaluation/datasets/intent_eval.holdout.v2.jsonl",
    "evaluation/datasets/rag_eval.formal.v1.jsonl",
    "evaluation/datasets/latency_orchestration.formal.v1.jsonl",
    "evaluation/reports/final_technical_evaluation.html",
    "evaluation/reports/technical_evaluation_v1.manifest.json",
    "tests/test_core_offline.py",
    "tests/test_orchestration_error_propagation.py",
]

FROZEN_HASHES = {
    "evaluation/datasets/intent_eval.holdout.v2.jsonl":
        "3c89c80ab5fbd92d28269571b3f4e7adb8699f8c3164561f86c29350cd0ba526",
    "evaluation/datasets/rag_eval.formal.v1.jsonl":
        "248e590044d1643fdad54b884db54eaa6e8d5404579fc3abd9bd74647e4e7fe9",
    "evaluation/datasets/latency_orchestration.formal.v1.jsonl":
        "ccd36766214fe7536fe57a8c31838e286e886938eaeb87bc767f63cf8f975db5",
    "evaluation/reports/final_technical_evaluation.html":
        "818e8044fd58405230adc4d2c0cf2ad98a567ff24eda640016cc9116f74154a0",
}

EXCLUDED_PARTS = {".venv", "__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".ini", ".toml",
    ".yaml", ".yml", ".example", ".sql", ".gitignore",
}
KEY_PATTERN = re.compile(r"\b[0-9a-fA-F]{32}\.[A-Za-z0-9]{12,}\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST_PATH:
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def scan_secrets(files: list[Path]) -> list[str]:
    findings = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if KEY_PATTERN.search(text):
            findings.append(path.relative_to(ROOT).as_posix())
        for line in text.splitlines():
            if line.startswith("ZHIPUAI_API_KEY="):
                value = line.split("=", 1)[1].strip()
                placeholder_markers = ("replace", "your", "你的", "<", ">")
                if value and not any(
                    marker.lower() in value.lower()
                    for marker in placeholder_markers
                ):
                    findings.append(path.relative_to(ROOT).as_posix())
    return sorted(set(findings))


def verify(write_manifest: bool) -> int:
    missing = [item for item in REQUIRED_PATHS if not (ROOT / item).is_file()]
    hash_errors = []
    for relative, expected in FROZEN_HASHES.items():
        path = ROOT / relative
        if path.is_file() and sha256(path) != expected:
            hash_errors.append(relative)

    files = package_files()
    secret_findings = scan_secrets(files)

    if write_manifest and not missing and not hash_errors and not secret_findings:
        entries = [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ]
        manifest = {
            "package": "差旅出行助手_项目交付包_V1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "file_count": len(entries),
            "total_size_bytes": sum(item["size_bytes"] for item in entries),
            "excluded": [
                ".env and API credentials",
                ".venv and Python caches",
                "personal memory files",
                "temporary logs, PID and progress files",
                "legacy smoke and draft reports",
            ],
            "files": entries,
        }
        MANIFEST_PATH.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Required files: {'PASS' if not missing else 'FAIL'}")
    print(f"Frozen hashes: {'PASS' if not hash_errors else 'FAIL'}")
    print(f"Credential scan: {'PASS' if not secret_findings else 'FAIL'}")
    print(f"Package files: {len(files)}")
    print(f"Package size: {sum(path.stat().st_size for path in files) / 1024 / 1024:.2f} MB")
    if missing:
        print("Missing:", ", ".join(missing))
    if hash_errors:
        print("Hash mismatch:", ", ".join(hash_errors))
    if secret_findings:
        print("Possible credentials:", ", ".join(secret_findings))
    return 1 if missing or hash_errors or secret_findings else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write PACKAGE_MANIFEST.json after all checks pass.",
    )
    args = parser.parse_args()
    raise SystemExit(verify(args.write_manifest))
