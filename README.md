# Regression-Modelling-for-Hospital-Bed-Usage-Prediction
# Deep Learning Portfolio

A six-question portfolio covering classical machine learning,
deep neural networks, and generative models — applied to healthcare,
aerospace, computer vision, and natural language tasks.

## Contents

- `deep.pdf` — Full written report with analysis and justifications
- `deep.ipynb` — Jupyter notebook containing all six question implementations
- `predict_decay.py` — Standalone inference script for orbital decay prediction (Q4)
- `predict_product.py` — Standalone inference script for dice product classification (Q5)
- `scaler.pkl` — Saved StandardScaler for Q4 inputs
- `weights.pkl` — Q4 neural network weights
- `weights.pth` — Q5 PyTorch model weights
- `best_model.pth` — Q5 best validation accuracy checkpoint

## Questions and Results

### Q1 — Regularised Linear Regression
Hospital bed days prediction using OLS, Ridge, Lasso, and ElasticNet.
Ridge selected as best model — Test RMSE 1.50, Test R² 0.68.

### Q2 — Decision Trees with PCA
Exam data classification. Peak accuracy 84.83% at depth 3.
PCA analysis with k=1, 2, 8 components showing dimensionality effects.

### Q3 — Word Embeddings
GloVe embeddings applied to Pride and Prejudice corpus.
Analysis of sentiment similarity and contextual neighbours.

### Q4 — Neural Network Regression
Satellite orbital decay time prediction.
MLP with 6 inputs → 256 → 128 → 64 → 32 → 1 output.
Log-transformed target, BatchNorm, Dropout, L1 loss.
Test MAE 72.72 days.

### Q5 — Convolutional Neural Network
Dice product image classification, 217 classes.
4 conv blocks (64→128→256→512), adaptive pooling, 3 FC layers.
4.9 million parameters, 97.27% validation accuracy.

### Q6 — Conditional Variational Autoencoder
Character glyph generation for 5 classes (c, n, s, v, z).
Encoder-decoder with 64-dim latent space and class conditioning.
Generates conditional samples and supports latent interpolation.

## Tech Stack

- **Language:** Python 3.12
- **ML Frameworks:** PyTorch, scikit-learn, Gensim
- **Data:** NumPy, Pandas
- **Visualisation:** Matplotlib

## Running the Code

```bash
# Install dependencies
pip install torch torchvision scikit-learn pandas numpy matplotlib gensim joblib pillow

# Run inference scripts
python predict_decay.py
python predict_product.py
```
