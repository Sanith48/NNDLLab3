# CIFAR-10 Classification — Lab 3

This repository contains a Jupyter notebook that trains a feedforward neural network (and a small CNN) on CIFAR-10, saves checkpoints and training plots, and includes a small NumPy backprop demo.

Artifacts produced by the notebook:
- `checkpoints/` — saved model checkpoints (e.g. `FFN-ReLU-full50_best.h5`)
- `plots/` — training loss/accuracy PNGs

Streamlit demo
----------------
A small Streamlit app is provided to demo the trained model.

Files:
- `streamlit_app.py` — Streamlit demo. It attempts to load the model at `checkpoints/FFN-ReLU-full50_best.h5`.

How to run the Streamlit demo
-----------------------------
1. Ensure dependencies are installed. It's recommended to use a virtualenv or conda environment.

   powershell example:

   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt
   ```

2. Start the Streamlit app from the project root:

   ```powershell
   streamlit run streamlit_app.py
   ```

3. If you don't have a checkpoint at `checkpoints/FFN-ReLU-full50_best.h5`, run the notebook `Sanith_148_Lab3.ipynb` (or `Sanith_148_Lab3_clean.ipynb` if present) to train and save the model, or place a compatible Keras `.h5` model at that path.

Notes
-----
- The feedforward model used in the notebook expects 32x32 RGB inputs scaled to [0,1]. Uploaded images will be resized to 32x32 before prediction.
- The Streamlit app uses the saved Keras model; if you trained a different architecture (e.g., CNN), make sure the checkpoint is compatible with the demo or adapt `streamlit_app.py` accordingly.

Contact
-------
For any issues, open a ticket or edit the notebook directly.
