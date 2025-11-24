# Probabilistic JEPA

Probabilistic JEPA augments the ViT-based I-JEPA self-supervised vision backbone with probabilistic heads—Mixture Density Network, RNADE-style autoregressive MLP, flow-matching RealNVP flow, and a latent diffusion predictor—trained on Tiny ImageNet-100 so the model outputs sampleable distributions for masked image tokens, capturing multimodal uncertainty while retaining the teacher–student masking pipeline and achieving representation quality comparable to the deterministic baseline.

### How to run the demo?

1. Open the console in the project folder
2. Type in `. .venv_streamlit/bin/activate`
3. Type in `streamlit run demo.py --server.headless true --server.port 8501`