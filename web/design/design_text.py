"""Write the text of each exported design screen, one string per line in document order.

    python web/design/design_text.py web/design/exports

Reads every <id>.html that Pencil exported and writes <id>.txt beside it. The text files are
what a build reads for verbatim copy when the design file itself is not open.
"""

from __future__ import annotations

import html
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("style", "script"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("style", "script"):
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        text = re.sub(r"\s+", " ", html.unescape(data)).strip()
        if text and not self._skip:
            self.lines.append(text)


def main() -> None:
    folder = Path(sys.argv[1])
    for page in sorted(folder.glob("*.html")):
        p = _Text()
        p.feed(page.read_text())
        page.with_suffix(".txt").write_text("\n".join(p.lines) + "\n")
        print(page.stem, len(p.lines), "lines")


if __name__ == "__main__":
    main()
