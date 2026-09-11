"""Spike. List a Google Drive folder recursively so we can see what a dataset holds.

Throwaway probe, not part of the pipeline. It reads and never writes or downloads.

Credentials come from an OAuth desktop client. The first run opens a browser once and caches
the token, later runs reuse it.

    python playground/trio/gdrive_inventory.py <folder-url-or-id>

Paths are overridable with GDRIVE_CLIENT_SECRET and GDRIVE_TOKEN. Neither file belongs in the
repo, the defaults live under ~/.config/3dgnome.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CONFIG = Path.home() / ".config" / "3dgnome"
CLIENT_SECRET = Path(os.environ.get("GDRIVE_CLIENT_SECRET", CONFIG / "gdrive_client_secret.json"))
TOKEN = Path(os.environ.get("GDRIVE_TOKEN", CONFIG / "gdrive_token.json"))
FOLDER_MIME = "application/vnd.google-apps.folder"

# Every field the walk reads. md5Checksum is absent on Google-native documents and on folders.
FIELDS = "nextPageToken, files(id, name, mimeType, size, md5Checksum, modifiedTime, shortcutDetails)"


def folder_id(s: str) -> str:
    """Accept either a bare folder id or any of the Drive URL shapes that carry one."""
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", s) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", s)
    return m.group(1) if m else s.strip()


def copy_to_clipboard(text: str) -> bool:
    """Put text on the macOS clipboard. False when pbcopy is missing or fails."""
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def service(port: int = 0) -> Any:
    creds = None
    if TOKEN.is_file():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        if not CLIENT_SECRET.is_file():
            sys.exit(
                f"no OAuth client secret at {CLIENT_SECRET}\n"
                "Download the desktop-app client JSON from the Google Cloud console and put it\n"
                "there, or point GDRIVE_CLIENT_SECRET at it."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES)

        # run_local_server builds the consent URL internally and then blocks on the loopback
        # server, so the URL is grabbed on its way out. Copying has to happen here rather than
        # afterwards, because afterwards is after the block. Rebuilding the URL separately would
        # not work, its state parameter has to be the one the waiting server expects.
        issue = flow.authorization_url

        def capture(*a: Any, **kw: Any) -> tuple[str, str]:
            url, state = issue(*a, **kw)
            where = "copied to clipboard" if copy_to_clipboard(url) else "copy it by hand"
            print(
                f"\n[gdrive] open this in the browser signed into the account that can see the\n"
                f"         folder. Any browser on this machine works, the redirect comes back to\n"
                f"         loopback here. The link is {where}.\n\n{url}\n",
                file=sys.stderr,
                flush=True,
            )
            return url, state

        flow.authorization_url = capture  # type: ignore[method-assign]
        creds = flow.run_local_server(port=port, open_browser=False, authorization_prompt_message="")
        TOKEN.parent.mkdir(parents=True, exist_ok=True)
        TOKEN.write_text(creds.to_json())
        TOKEN.chmod(0o600)
        print(f"[gdrive] token cached at {TOKEN}", file=sys.stderr)
    return build("drive", "v3", credentials=creds)


def children(svc: Any, parent: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    page = None
    while True:
        resp = (
            svc.files()
            .list(
                q=f"'{parent}' in parents and trashed = false",
                fields=FIELDS,
                pageSize=1000,
                pageToken=page,
                orderBy="folder,name",
                # Set so a folder living on a shared drive lists like any other.
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        out.extend(resp.get("files", []))
        page = resp.get("nextPageToken")
        if not page:
            return out


def walk(svc: Any, node: str, prefix: str = "", rows: list[dict[str, Any]] | None = None,
         depth: int = 0) -> list[dict[str, Any]]:
    rows = [] if rows is None else rows
    for f in children(svc, node):
        # A shortcut points at a file held elsewhere. Follow the target so the listing shows what
        # the folder actually gives access to.
        target = f.get("shortcutDetails", {}).get("targetId")
        mime = f.get("shortcutDetails", {}).get("targetMimeType", f["mimeType"])
        path = f"{prefix}/{f['name']}"
        if mime == FOLDER_MIME:
            print(f"{'  ' * depth}{f['name']}/")
            walk(svc, target or f["id"], path, rows, depth + 1)
            continue
        size = int(f["size"]) if "size" in f else None
        human = f"{size / 1e6:10.2f} MB" if size is not None else "  native doc"
        print(f"{'  ' * depth}{f['name']:<52} {human}  {mime}")
        rows.append(
            {
                "path": path,
                "id": target or f["id"],
                "name": f["name"],
                "mimeType": mime,
                "size": size,
                "md5": f.get("md5Checksum"),
                "modified": f.get("modifiedTime"),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="Drive folder URL or bare id")
    ap.add_argument("--json", help="also write the flat file list here")
    ap.add_argument(
        "--port",
        type=int,
        default=0,
        help="fixed loopback port for the OAuth redirect. 0 picks a free one. Set it when the "
        "client needs a redirect URI registered up front.",
    )
    args = ap.parse_args()

    rows = walk(service(args.port), folder_id(args.folder))

    total = sum(r["size"] or 0 for r in rows)
    print(f"\n{len(rows)} files, {total / 1e9:.2f} GB total", file=sys.stderr)
    ext = Counter("".join(Path(r["name"]).suffixes[-2:]) or "(none)" for r in rows)
    for suffix, n in ext.most_common():
        by = sum(r["size"] or 0 for r in rows if "".join(Path(r["name"]).suffixes[-2:]) == suffix)
        print(f"  {suffix:<20} {n:4d} files  {by / 1e9:8.2f} GB", file=sys.stderr)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"[gdrive] wrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
