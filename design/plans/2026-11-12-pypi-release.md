# gnome3d-ng on PyPI, the release checklist

State on 2026-11-12. Nothing is published.

## Done

- The name `gnome3d-ng` is free on PyPI (404 on 2026-11-11), as are `gnome3d` and `3dgnome`.
- `pip wheel .` builds `gnome3d_ng-0.2.0-py3-none-any.whl`, about 210 kB, holding only the
  `gnome3d` package and its metadata. The reference, the data, validation and playground stay
  out, by `[tool.setuptools.packages.find] include = ["gnome3d*"]`.
- scipy added to the runtime dependencies. The production arcs solver is scipy's L-BFGS-B, so
  an install without it failed at the first solve. A clean venv installing only the wheel
  imports the solver, the stitch and the CLI, and `gnome3d-ng --help` runs.
- The website calls the same CLI the package installs, so the site and the release run one engine.

## Decisions that are yours

1. **Licence.** The repository has no LICENSE file, and PyPI shows none. The vendored rules in
   `web/vendor/personal3d/rules` are BSD; the reference 3dgnome has its own terms. Pick one
   before anything is published.
2. **Version.** 0.2.0 now. 1.0.0 would say the engine is closed, which is the plan.
3. **The README.** It opens with "a very experimental playground project". PyPI shows it as
   the project page, so it wants a short user facing version: install, one example config, the
   CLI, the JAX extra for GPUs, a pointer to SETTINGS.md and the citation.
4. **Who publishes.** A PyPI account or a lab organisation, and an API token or trusted
   publishing from GitHub Actions.

## Then

```bash
python -m pip install build twine
python -m build                      # sdist and wheel in dist/
twine check dist/*
twine upload --repository testpypi dist/*   # a dry run on TestPyPI first
pip install -i https://test.pypi.org/simple/ gnome3d-ng[jax]
twine upload dist/*
```
