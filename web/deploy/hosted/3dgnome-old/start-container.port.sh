#!/bin/bash
# The site's start script without the database, which runs in a container of its own. It waits
# for that database, makes the tables that are missing, starts the queue and the web server.
# The queue reads lines added from now on. The script it replaces read the file's last ten
# lines too, which ran the last ten requests again at every start.
python - <<'PY'
import time
import sqlalchemy
import config

engine = sqlalchemy.create_engine(config.SQLALCHEMY_DATABASE_URI)
for _ in range(180):
    try:
        engine.connect().close()
        break
    except Exception:
        time.sleep(1)
PY
python db_create.py
touch tasks_file
tail -n 0 -f tasks_file | parallel -j 4 --timeout 86400 --joblog gnu_log &
exec gunicorn -w 5 -b 0.0.0.0:8000 app:app
