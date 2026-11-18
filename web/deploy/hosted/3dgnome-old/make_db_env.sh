#!/bin/bash
# Writes db.env for MariaDB's image from the database address in the site's config, without
# printing the password.   make_db_env.sh <copy of the source> > db.env
python3 - "$1" <<'PY'
import re, sys
s = open(sys.argv[1] + "/config.py.dist_local").read()
user, pw, name = re.search(r"mysql\+pymysql://([^:]+):([^@]+)@[^/]+/(\w+)", s).groups()
print(f"MARIADB_DATABASE={name}\nMARIADB_USER={user}\nMARIADB_PASSWORD={pw}")
PY
