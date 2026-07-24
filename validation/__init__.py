"""3dgnome validation harness.

Checks that reconstructed structures make sense using the resolution-independent validation toolkit
the original 3D-GNOME papers used, Szałaj et al. 2016 and Tang et al. 2015. See docs/validation.md
for the methodology and which metric implements which check. Self-consistency, scaling laws,
ensemble diversity, and the excluded-volume and confinement overlap test that motivated our
divergence.

Everything here runs through the public gnome3d API, Settings, ContactData, and simulate, which is
what a user doing modelling would call. It reads only public output, BeadOut, plus the user's own
input contacts.
"""
