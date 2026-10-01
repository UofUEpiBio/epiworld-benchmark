"""Every relative link in the rendered documentation resolves: the file
exists and, for a link with an anchor, the target has a heading with that
GitHub anchor."""

from __future__ import annotations

from pathlib import Path
import re

from run import ROOT

DOCUMENTS = [
    ROOT / "README.md", ROOT / "setup.md", ROOT / "scenarios.md",
    *sorted((ROOT / "docs").glob("*.md")), *sorted(ROOT.glob("scenario_*/README.md")),
]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)


def github_anchor(heading: str) -> str:
    """The anchor GitHub gives a Markdown heading."""
    text = re.sub(r"`|\*|_(?=\w)|(?<=\w)_", "", heading.strip().lower())
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    return {github_anchor(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+?)\s*#*$", text, re.MULTILINE)}


def test_github_anchors() -> None:
    assert github_anchor("Why the two fastest engines differ") == "why-the-two-fastest-engines-differ"
    assert github_anchor("Watts–Strogatz") == "wattsstrogatz"
    assert github_anchor("`run()` and setup") == "run-and-setup"


def test_rendered_documents_have_no_broken_relative_links() -> None:
    broken = []
    for document in DOCUMENTS:
        if not document.is_file():
            broken.append(f"{document.relative_to(ROOT)}: not rendered")
            continue
        text = FENCE.sub("", document.read_text(encoding="utf-8"))
        for target in LINK.findall(text):
            if re.match(r"[a-z]+:", target):
                continue
            path, _, anchor = target.partition("#")
            resolved = (document.parent / path).resolve() if path else document
            if not resolved.exists():
                broken.append(f"{document.relative_to(ROOT)}: {target} (no such file)")
            elif anchor and resolved.suffix == ".md" and anchor not in anchors(resolved):
                broken.append(f"{document.relative_to(ROOT)}: {target} (no such heading)")
    assert not broken, "\n".join(broken)
