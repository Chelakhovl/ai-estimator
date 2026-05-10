from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def load_eval_cases(fixtures_dir: Path) -> list[dict]:
    cases: list[dict] = []
    for fixture_path in sorted(fixtures_dir.glob("*.json")):
        cases.extend(json.loads(fixture_path.read_text()))
    return cases


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if os.getenv("AI_EVAL_FORCE_LLM", "").lower() not in {"1", "true", "yes"}:
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["OPENAI_MODEL"] = ""

    sys.path.insert(0, str(root))

    from importlib import reload

    import app.config as app_config
    import app.services.llm_client as app_llm_client
    import app.services.matcher as app_matcher
    from app.schemas import PreviewRequest
    from app.services.matcher import generate_preview

    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    fixtures_dir = root / "app" / "tests" / "fixtures"
    cases = load_eval_cases(fixtures_dir)

    passed = 0
    for case in cases:
        request = PreviewRequest(
            quote_guid=f"eval-{case['name']}",
            prompt=case["prompt"],
            candidate_rows=case["candidate_rows"],
        )
        response = generate_preview(request)
        matched_ids = [row.INSIDEQUOTESGUID for row in response.matched_rows]
        warning_text = " ".join(assumption.text for assumption in response.assumptions if assumption.severity == "warning")

        checks: list[tuple[str, bool]] = [
            ("matched_ids", set(matched_ids) == set(case["expected_matched_ids"])),
            (
                "unmatched_count",
                len(response.unmatched_items) == case.get("expected_unmatched_count", len(response.unmatched_items)),
            ),
            (
                "review_count",
                response.review_summary.review_count == case.get("expected_review_count", response.review_summary.review_count),
            ),
        ]
        for fragment in case.get("expected_warning_contains", []):
            checks.append((f"warning:{fragment}", fragment in warning_text))

        case_passed = all(result for _, result in checks)
        if case_passed:
            passed += 1
            print(f"PASS {case['name']}")
            continue

        print(f"FAIL {case['name']}")
        print(f"  matched_ids:   actual={matched_ids} expected={case['expected_matched_ids']}")
        print(f"  unmatched:     actual={len(response.unmatched_items)} expected={case.get('expected_unmatched_count')}")
        print(f"  review_count:  actual={response.review_summary.review_count} expected={case.get('expected_review_count')}")
        print(f"  warnings:      {warning_text}")

    print(f"\nSummary: {passed}/{len(cases)} cases passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
