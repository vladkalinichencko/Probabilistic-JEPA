# Variational JEPA – Adding Uncertainty to Visual Representations

**[Read the report (PDF)](report/report.pdf)**

JEPA models predict latent representations rather than pixels, but a standard
predictor returns one answer even when an image admits several plausible
completions. This project asks what changes when that prediction becomes a
distribution: it compares mixture, autoregressive, flow-matching, and diffusion
heads on the same visual JEPA backbone, probing uncertainty without giving up
semantic representation learning.
