"""LangGraph-only command-line entrypoint for batch diacritization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langgraph_pipeline import build_langgraph_pipeline, run_one_batch
from runtime import PROJECT_ROOT


def _resolve_input_files(argv: list[str]) -> list[Path]:
    """Resolve the optional input-file argument without framework coupling."""
    inputs_dir = PROJECT_ROOT / "dataset" / "inputs"
    if len(argv) > 1:
        arg = argv[1]
        if arg == "--all":
            return sorted(inputs_dir.glob("*.jsonl"))
        supplied = Path(arg)
        if supplied.is_absolute() or supplied.exists():
            return [supplied]
        candidate = inputs_dir / supplied.name
        if candidate.exists():
            return [candidate]
        raise FileNotFoundError(f"Input file not found: {arg}")
    return sorted(inputs_dir.glob("*.jsonl"))


def _batches_by_meter(input_path: Path) -> dict[str, list[dict]]:
    batches: dict[str, list[dict]] = {}
    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                verse = {
                    "verse_id": raw["verse_id"],
                    "sadr": raw["sadr"],
                    "ajuz": raw.get("ajuz", ""),
                }
            except KeyError as exc:
                raise ValueError(f"{input_path}:{line_number} is missing {exc.args[0]!r}") from exc
            batches.setdefault(raw.get("meter", "taweel"), []).append(verse)
    return batches


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv
    try:
        input_files = _resolve_input_files(argv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[-] {exc}")
        return

    if not input_files:
        print(f"[-] No .jsonl files found in {PROJECT_ROOT / 'dataset' / 'inputs'}")
        return

    graph, checkpoint_conn, _ = build_langgraph_pipeline(use_checkpointer=True)
    try:
        for input_path in input_files:
            for meter, verses in _batches_by_meter(input_path).items():
                if not verses:
                    continue
                print(f"[*] Processing {len(verses)} verse(s), meter={meter!r}, input={input_path.name}")
                # A LangGraph-only namespace prevents accidental reuse of a
                # checkpoint created by the retired DeepAgents architecture.
                thread_id = f"lg:{input_path.stem}:{meter}"
                run_one_batch(graph, verses, meter, thread_id)
    finally:
        if checkpoint_conn is not None:
            checkpoint_conn.close()


if __name__ == "__main__":
    main()
