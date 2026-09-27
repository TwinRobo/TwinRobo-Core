"""MkDocs hook: make repo-relative Markdown links work on the site.

The pages are written for GitHub, where ``README.md`` links ``docs/rendering.md``
and ``docs/cameras.md`` links ``../CONTRIBUTING.md``. On the site those become
site pages; links to anything else in the repo (code, ``LICENSE``, issue
templates) become GitHub URLs.
"""

import posixpath
import re
from urllib.parse import urljoin

REPO = "https://github.com/TwinRobo/TwinRobo-Core"
SITE_PAGES = {"README.md": "index.md", "CONTRIBUTING.md": "contributing.md"}
LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def _repo_path(src_uri: str) -> str:
    """Where a site page's Markdown lives in the repo."""
    inverse = {v: k for k, v in SITE_PAGES.items()}
    return inverse.get(src_uri, f"docs/{src_uri}")


def on_page_markdown(markdown, page, config, files):
    src = page.file.src_uri
    origin = _repo_path(src)
    here = posixpath.dirname(src)

    def fix(m):
        target = m.group(2)
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            return m.group(0)
        path, _, frag = target.partition("#")
        url = urljoin(f"{REPO}/blob/main/{origin}", path)
        prefix = f"{REPO}/blob/main/"
        if url.startswith(prefix):
            repo_file = url[len(prefix) :]
            site = SITE_PAGES.get(repo_file)
            if site is None and repo_file.startswith("docs/"):
                site = repo_file[len("docs/") :]
            if site is not None:
                rel = posixpath.relpath(site, here or ".")
                return f"{m.group(1)}{rel}{'#' + frag if frag else ''}{m.group(3)}"
        if not path:
            return m.group(0)
        return f"{m.group(1)}{url}{'#' + frag if frag else ''}{m.group(3)}"

    return LINK.sub(fix, markdown)
