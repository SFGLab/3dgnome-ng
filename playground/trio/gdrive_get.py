"""Spike. Download one Drive file by id, exporting Google-native docs to text.

    python playground/trio/gdrive_get.py <file-id> <dest>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdrive_inventory import service  # noqa: E402

EXPORT = {
    "application/vnd.google-apps.document": ("text/plain", ".txt"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
}


def main() -> None:
    file_id, dest = sys.argv[1], Path(sys.argv[2])
    svc = service()
    meta = svc.files().get(fileId=file_id, fields="name, mimeType", supportsAllDrives=True).execute()
    mime = meta["mimeType"]
    if mime in EXPORT:
        data = svc.files().export(fileId=file_id, mimeType=EXPORT[mime][0]).execute()
    else:
        data = svc.files().get_media(fileId=file_id, supportsAllDrives=True).execute()
    dest.write_bytes(data)
    print(f"{meta['name']} -> {dest} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
