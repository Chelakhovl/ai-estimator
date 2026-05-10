from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def load_project_intake_knowledge() -> str:
    repo_root = _repo_root()
    instruction_path = repo_root / "estimator_files" / "PROJECT_INSTRUCTIONS_PASTE_INTO_CLAUDE.md"
    knowledge_dir = repo_root / "estimator_files" / "knowledge"
    chunks = [f"# {instruction_path.name}\n{instruction_path.read_text(encoding='utf-8')}"]
    if knowledge_dir.exists():
        for path in sorted(knowledge_dir.iterdir()):
            if path.is_file():
                chunks.append(f"# {path.name}\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(chunks)

