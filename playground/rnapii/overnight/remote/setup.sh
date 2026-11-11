set -e
cd ~/3dgnome-web-smoke
python3.12 -m venv venv
venv/bin/pip -q install numpy==2.4.4 numba==0.65.1 scipy==1.17.1 "jax[cuda12]==0.10.1"
venv/bin/pip -q install -r repo/web/requirements.txt
venv/bin/pip -q install --no-deps -e repo
venv/bin/python -c "import jax, numba, pandas, fastapi; print(jax.devices(), numba.__version__, pandas.__version__)"
echo SETUP DONE
