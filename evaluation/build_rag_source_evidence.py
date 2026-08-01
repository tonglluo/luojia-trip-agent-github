"""Build line-level source evidence for a RAG evaluation candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import load_jsonl, utc_now_iso, write_json


KNOWLEDGE_DIR = (
    PROJECT_ROOT
    / ".claude"
    / "skills"
    / "ask-question"
    / "data"
    / "documents"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="evaluation/datasets/rag_eval.formal.candidate.jsonl",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/rag_formal_source_evidence.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="evaluation/reports/rag_formal_source_evidence.md",
    )
    return parser.parse_args()


def normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold())


def locate_point(
    point: str,
    question: str,
    key_points: list[str],
    sources: list[str],
    source_lines: dict[str, list[str]],
) -> dict[str, Any] | None:
    target = normalize(point)
    normalized_question = normalize(question)
    question_bigrams = {
        normalized_question[index:index + 2]
        for index in range(max(0, len(normalized_question) - 1))
    }
    normalized_points = [normalize(item) for item in key_points]
    candidates: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    for source_index, source in enumerate(sources):
        lines = source_lines[source]
        for window_size in (1, 2, 3):
            for start in range(len(lines) - window_size + 1):
                window = lines[start:start + window_size]
                normalized_window = normalize("\n".join(window))
                if target in normalized_window:
                    co_located_points = sum(
                        candidate in normalized_window
                        for candidate in normalized_points
                        if candidate
                    )
                    question_overlap = sum(
                        bigram in normalized_window
                        for bigram in question_bigrams
                    )
                    evidence = {
                        "source": source,
                        "line_start": start + 1,
                        "line_end": start + window_size,
                        "co_located_point_count": co_located_points,
                        "question_bigram_overlap": question_overlap,
                        "snippet": "\n".join(
                            line.strip() for line in window if line.strip()
                        ),
                    }
                    score = (
                        co_located_points,
                        question_overlap,
                        -window_size,
                        -source_index,
                    )
                    candidates.append((score, evidence))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def build(
    rows: list[dict[str, Any]],
    *,
    dataset_path: Path,
) -> dict[str, Any]:
    source_lines = {
        path.name: path.read_text(encoding="utf-8").splitlines()
        for path in KNOWLEDGE_DIR.glob("*.txt")
    }
    evidence_rows = []
    missing = []
    weak_evidence = []

    for row in rows:
        point_evidence = []
        for point in row["answer_key_points"]:
            evidence = locate_point(
                point,
                row["question"],
                row["answer_key_points"],
                row["expected_sources"],
                source_lines,
            )
            item = {
                "key_point": point,
                "evidence": evidence,
            }
            point_evidence.append(item)
            if evidence is None:
                missing.append({
                    "id": row["id"],
                    "key_point": point,
                })
            elif (
                evidence["co_located_point_count"] == 1
                and evidence["question_bigram_overlap"] == 0
            ):
                weak_evidence.append({
                    "id": row["id"],
                    "key_point": point,
                    "evidence": evidence,
                })
        evidence_rows.append({
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "answerable": row["answerable"],
            "expected_sources": row["expected_sources"],
            "point_evidence": point_evidence,
            "unanswerable_rationale": row.get("annotation_note"),
        })

    answerable_rows = [row for row in rows if row["answerable"]]
    total_points = sum(
        len(row["answer_key_points"]) for row in answerable_rows
    )
    return {
        "generated_at": utc_now_iso(),
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_sha256": hashlib.sha256(
            dataset_path.read_bytes(),
        ).hexdigest(),
        "sample_count": len(rows),
        "answerable_count": len(answerable_rows),
        "unanswerable_count": sum(
            not row["answerable"] for row in rows
        ),
        "answer_key_point_count": total_points,
        "located_key_point_count": total_points - len(missing),
        "line_level_evidence_coverage": round(
            (total_points - len(missing)) / total_points
            if total_points
            else 0.0,
            4,
        ),
        "missing_evidence_count": len(missing),
        "missing_evidence": missing,
        "weak_evidence_count": len(weak_evidence),
        "weak_evidence": weak_evidence,
        "rows": evidence_rows,
    }


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# RAG 正式评测集来源证据",
        "",
        f"- 样本数：{result['sample_count']}",
        f"- 可回答 / 不可回答：{result['answerable_count']} / "
        f"{result['unanswerable_count']}",
        f"- 答案要点：{result['answer_key_point_count']}",
        f"- 行级证据覆盖率：{result['line_level_evidence_coverage']:.2%}",
        f"- 缺失证据：{result['missing_evidence_count']}",
        f"- 弱关联证据：{result['weak_evidence_count']}",
        "",
        "## 可回答问题",
        "",
        "|ID|问题|答案要点|来源证据|",
        "|---|---|---|---|",
    ]
    for row in result["rows"]:
        if not row["answerable"]:
            continue
        evidence_parts = []
        for item in row["point_evidence"]:
            evidence = item["evidence"]
            if evidence is None:
                evidence_parts.append(
                    f"{item['key_point']} → **未定位**",
                )
            else:
                line_label = (
                    str(evidence["line_start"])
                    if evidence["line_start"] == evidence["line_end"]
                    else (
                        f"{evidence['line_start']}-"
                        f"{evidence['line_end']}"
                    )
                )
                evidence_parts.append(
                    f"{item['key_point']} → "
                    f"`{evidence['source']}:{line_label}` "
                    f"{markdown_escape(evidence['snippet'])}"
                )
        lines.append(
            f"|{row['id']}|{markdown_escape(row['question'])}|"
            f"{'；'.join(item['key_point'] for item in row['point_evidence'])}|"
            f"{'<br>'.join(evidence_parts)}|"
        )

    lines.extend([
        "",
        "## 不可回答问题",
        "",
        "|ID|问题|拒答依据|",
        "|---|---|---|",
    ])
    for row in result["rows"]:
        if row["answerable"]:
            continue
        lines.append(
            f"|{row['id']}|{markdown_escape(row['question'])}|"
            f"{markdown_escape(row['unanswerable_rationale'] or '')}|"
        )

    lines.extend([
        "",
        "## 使用说明",
        "",
        "- 行号对应当前八份知识文档；文档变更后必须重新生成证据。",
        "- 行级字符串证据证明答案要点可追溯，但不能替代对问题语义的人工复核。",
        "- 不可回答题的依据是“请求的具体制度或数值不存在”，不是简单关键词缺失。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_path = PROJECT_ROOT / args.dataset
    rows = load_jsonl(dataset_path)
    result = build(rows, dataset_path=dataset_path)
    output_path = write_json(PROJECT_ROOT / args.output, result)
    markdown_path = PROJECT_ROOT / args.markdown_output
    write_markdown(markdown_path, result)
    print(json.dumps({
        key: result[key]
        for key in (
            "sample_count",
            "answer_key_point_count",
            "located_key_point_count",
            "line_level_evidence_coverage",
            "missing_evidence_count",
            "weak_evidence_count",
        )
    }, ensure_ascii=False, indent=2))
    print(f"Saved evidence: {output_path}")
    print(f"Saved markdown: {markdown_path}")
    return 0 if not result["missing_evidence"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
