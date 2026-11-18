"""The changes the site's source needs for current library releases, applied to a copy.

    python apply_port.py <copy of the source>

Each change names the text it replaces and how often it occurs, and stops if the source differs.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def edit(path: str, old: str, new: str, count: int) -> None:
    f = root / path
    s = f.read_text()
    if s.count(old) != count:
        raise SystemExit(f"{path}: {old!r} occurs {s.count(old)} times, expected {count}")
    f.write_text(s.replace(old, new))
    print(f"{path}: {count} x {old!r}")


# numpy 1.24 removed the aliases of the builtin types.
edit("app/pipeline/calc_chiapet_data_stats.py", "dtype=np.int)", "dtype=int)", 1)
edit("app/pipeline/calc_chiapet_data_stats.py", "dtype=np.float)", "dtype=float)", 1)
# Python 3.9 moved gcd to math. cmp came from the future package, which the site needs for
# nothing else.
edit("external_scripts/genome_lib.py", "from fractions import gcd", "from math import gcd", 1)
CMP = "def cmp(a, b):\n    return (a > b) - (a < b)\n"
edit("app/views_help.py", "from past.builtins import cmp\n", CMP, 1)
edit("app/modelsVisualiser.py", "from past.builtins import cmp\n", CMP, 1)
# Flask-WTF 1.0 dropped the name Form for its form class.
edit("app/forms.py", "from flask_wtf import Form", "from flask_wtf import FlaskForm as Form", 1)
# Flask-SQLAlchemy 3 reaches the database inside an application context alone, and the pipeline
# is a script of its own.
edit("chromatin_pipeline.py", "from app import db\n", "from app import app, db\n\napp.app_context().push()\n", 1)
# Jinja 3.1 no longer exports Markup, which is markupsafe's.
edit("app/views.py", "from jinja2 import Markup", "from markupsafe import Markup", 1)
# Flask 2 renamed send_file's argument.
edit("app/views.py", "attachment_filename=", "download_name=", 2)
# sqlalchemy-migrate does not support SQLAlchemy 2 and Python 3.12 has no imp module. The site
# has two tables and create_all makes them where they are missing.
(root / "db_create.py").write_text(
    "#!/usr/bin/env python\nfrom app import app, db\n\nwith app.app_context():\n    db.create_all()\n"
)
print("db_create.py: rewritten")
s = (root / "start-container.sh").read_text()
for old, new, n in (
    ("service mysql start", "service mariadb start", 1),
    ("python db_migrate.py\n", "", 1),
    (" --log-level=debug --reload", "", 1),
):
    if s.count(old) != n:
        raise SystemExit(f"start-container.sh: {old!r} occurs {s.count(old)} times, expected {n}")
    s = s.replace(old, new)
(root / "start-container.sh").write_text(s)
print("start-container.sh: mariadb service, no migrate step, gunicorn without reload and debug")
