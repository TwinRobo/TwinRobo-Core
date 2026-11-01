"""MkDocs hook: list the catalog's IDs on the specs page, read from ``twinrobo/catalog``.

Each documentation version therefore lists exactly what its code can load.
"""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "twinrobo" / "catalog"
REPO = "https://github.com/TwinRobo/TwinRobo-Core"
BROWSER = "https://twinrobo.github.io/catalog/"  # the camera browser; #cam=<id> opens a camera
REF = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")  # the branch or tag being built
if os.environ.get("READTHEDOCS_VERSION_TYPE") == "external":  # a pull request: the id is its number
    REF = os.environ["READTHEDOCS_GIT_COMMIT_HASH"]


def camera_shown(path: Path, entry: dict) -> str:
    """The catalog camera the website shows for an entry: a module shows its left eye."""
    eye = (entry.get("eyes") or {}).get("left", {}).get("spec")
    return (
        (path.parent / eye).resolve().relative_to(CATALOG).parent.as_posix() if eye else entry["id"]
    )


def table(filename: str, live: bool = True) -> str:
    """The catalog's entries of one kind; ``live`` links each to the website's camera browser."""
    head, rule, rows = "| ID | Name | File |", "|---|---|---|", []
    if live:
        head, rule = head + " |", rule + "---|"
    for path in sorted(CATALOG.rglob(filename)):
        entry = yaml.safe_load(path.read_text())
        name = f"{entry.get('manufacturer', '')} {entry.get('product', '')}".strip()
        link = f"{REPO}/blob/{REF}/{path.relative_to(ROOT).as_posix()}"
        row = f"| `{entry['id']}` | {name} | [spec]({link}) |"
        if live:
            row += f" [Try it live]({BROWSER}#cam={camera_shown(path, entry)}){{ .tr-try }} |"
        rows.append(row)
    return "\n".join([head, rule, *rows])


def on_page_markdown(markdown, page, config, files):
    for marker, filename, live in (
        ("{{ catalog_cameras }}", "camera.yaml", True),
        ("{{ catalog_modules }}", "module.yaml", True),
        ("{{ catalog_lidars }}", "lidar.yaml", False),
    ):
        if marker in markdown:
            markdown = markdown.replace(marker, table(filename, live))
    return markdown
