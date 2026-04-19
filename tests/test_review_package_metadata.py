from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_release_metadata_files_exist() -> None:
    assert (ROOT / "CITATION.cff").exists()
    assert (ROOT / "LICENSE").exists()
    assert (ROOT / "REPRODUCIBILITY.md").exists()


def test_review_facing_files_do_not_contain_local_workstation_paths() -> None:
    review_files = [
        ROOT / "README.md",
        ROOT / "benchmark" / "README.md",
        ROOT / "benchmark" / "PUBLIC_EHT_VALIDATION.md",
        ROOT / "benchmark" / "REVIEW_SNAPSHOT.md",
        ROOT / "benchmark" / "EXPECTED_OUTPUTS.md",
        ROOT / "paper" / "visual_asset_manifest.json",
        *sorted((ROOT / "paper" / "figures").glob("*.selection.json")),
    ]
    for path in review_files:
        text = path.read_text(encoding="utf-8")
        assert "/Users/stelioszacharioudakis" not in text, path.as_posix()


def test_manuscript_and_package_metadata_are_clean() -> None:
    manuscript = (ROOT / "paper" / "manuscript.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "REPLACE_WITH_COAUTHOR_EMAIL" not in manuscript
    assert "OpenAI Codex" not in pyproject
    assert "small, Colab-first research prototype" not in (ROOT / "README.md").read_text(encoding="utf-8")
