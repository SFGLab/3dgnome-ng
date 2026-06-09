"""3dgnome validation harness.

Proves that reconstructed structures "make sense" using the *resolution-independent*
validation toolkit the original 3D-GNOME papers used (Szałaj et al. 2016; Tang et al.
2015) — not MERFISH/250-kb imaging, which is too coarse for our native-resolution
models. See ``docs/validation.md`` for the methodology and which metric implements
which check (V1 self-consistency, V2 scaling laws, V3 ensemble reproducibility, plus the
excluded-volume / confinement overlap test that motivated our divergence).

Everything here runs through the *public* gnome3d API (``Settings``, ``ContactData``,
``simulate``) — exactly what a user doing modelling would call — and reads only public
output (``BeadOut``) plus the user's own input contacts.
"""
