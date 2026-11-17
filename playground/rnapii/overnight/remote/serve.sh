cd ~/3dgnome-web-smoke/repo
export PYTHONPATH=web:web/vendor DATA_DIR=$HOME/3dgnome-web-smoke/site-data RESULTS_DIR=$HOME/3dgnome-web-smoke/results2 QUEUE=thread ENGINE_PYTHON=$HOME/3dgnome-web-smoke/venv/bin/python ENGINE_ROOT=$HOME/3dgnome-web-smoke/repo GNOME3D_JAX_CACHE=$HOME/3dgnome-web-smoke/jax-cache
exec ../venv/bin/uvicorn gnome3d_web.app:create_app --factory --host 127.0.0.1 --port 8765
