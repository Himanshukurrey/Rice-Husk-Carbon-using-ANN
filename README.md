# ANN Model for Adsorption of Dyes on Rice Husk Carbon

This project implements an Artificial Neural Network (ANN) model using PyTorch to predict the adsorption behavior of various dyes on rice husk carbon adsorbent.

## Paper Reference

Based on: **"Artificial Neural Network (ANN) Modeling for Adsorption of Dyes on Rice Husk Carbon"**
- Published in: International Journal of Computer Applications (0975 – 8887)
- Volume 41 – No.4, March 2012

## Overview

The ANN model predicts two key outputs:
1. **Equilibrium concentration** (qe in mg/gm) - Amount of adsorbate adsorbed per unit amount of adsorbent
2. **% Adsorption** - Percentage of dye removed from solution

### Input Parameters

The model takes three inputs:
- **Adsorbate coding**: Numerical code representing the type of dye
  - 30 = Bromocresol Red
  - 60 = Alizarin Red
  - 80 = Malachite Green
  - 90 = Methylene Blue
- **Initial concentration** (mg/lit): Starting concentration of the dye in solution
- **Adsorbent amount** (gm): Amount of rice husk carbon used

### Model Architecture

```
Input Layer (3 neurons)
    ↓
Hidden Layer 1 (5 neurons, Sigmoid activation)
    ↓
Hidden Layer 2 (5 neurons, Sigmoid activation)
    ↓
Output Layer (2 neurons)
```

**Training Parameters:**
- Learning rate: 0.3
- Momentum factors: 0.75 (primary), 0.01 (weight decay)
- Training epochs: 50,000
- Optimizer: SGD with momentum

## Installation

### 1. Create Conda Environment

```bash
conda create -n rice python=3.10 -y
conda activate rice
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch numpy matplotlib scikit-learn pandas PyPDF2
```

## Usage

### Training the Model

Run the main training script:

```bash
conda activate rice
python ann_adsorption_model.py
```

This will:
- Load the training and test data from the paper
- Train the ANN model for 50,000 epochs
- Evaluate performance on both training and test sets
- Generate visualization plots
- Save the trained model to `ann_model.pth`
- Save result plots to `ann_results.png`

### Making Predictions

Use the prediction script to make predictions with the trained model:

```bash
conda activate rice
python predict_adsorption.py
```

The script provides:
- **Example predictions** for common scenarios
- **Interactive mode** where you can input custom values

#### Example Usage in Code

```python
from predict_adsorption import load_model, predict

# Load the trained model
model, scaler_X, scaler_y = load_model()

# Make a prediction
# Predict for Malachite Green (80) with 500 mg/lit initial concentration
# and 2.5 gm adsorbent
qe, percent_ads = predict(
    adsorbate_coding=80,
    initial_concentration=500,
    adsorbent_amount=2.5,
    model=model,
    scaler_X=scaler_X,
    scaler_y=scaler_y
)

print(f"Equilibrium Concentration: {qe:.4f} mg/gm")
print(f"% Adsorption: {percent_ads:.2f}%")
```

## Dataset

The model uses data from the original paper:

### Training Data (18 samples)
- Bromocresol Red: 4 samples
- Alizarin Red: 4 samples
- Malachite Green: 6 samples
- Methylene Blue: 4 samples

### Test Data (4 samples)
- One sample from each dye type

## Results

The model achieves excellent accuracy with:
- **Training RMSE**: ~0.025 (as reported in paper)
- **Test RMSE**: ~0.024 (as reported in paper)
- Very low relative errors across all predictions

### Visualization

The training script generates comprehensive plots showing:
1. Actual vs Predicted values for equilibrium concentration (training & test)
2. Actual vs Predicted values for % adsorption (training & test)
3. Relative error plots for both outputs (training & test)
4. Training loss curve

## Files

- `ann_adsorption_model.py` - Main training script
- `predict_adsorption.py` - Prediction script for trained model
- `requirements.txt` - Python dependencies
- `DOC-20251129-WA0006.pdf` - Original research paper
- `ann_model.pth` - Trained model (generated after training)
- `ann_results.png` - Visualization plots (generated after training)

## Key Features

✅ **Accurate Predictions**: Closely matches actual experimental values  
✅ **Multi-Adsorbate Support**: Single model handles multiple dye types  
✅ **Easy to Use**: Simple prediction interface  
✅ **Well Documented**: Clear code with comprehensive comments  
✅ **Visualization**: Detailed plots for analysis  
✅ **Reproducible**: Fixed random seeds for consistent results

## Rice Husk Carbon Properties

According to the paper:
- **Yield**: 50%
- **Surface Area**: 208.637 ± 3.4941 m²/gm (BET analysis)

## Future Extensions

The model can be extended to:
- Include more adsorbate types
- Incorporate additional input parameters (pH, temperature, contact time)
- Use different adsorbent materials
- Implement ensemble methods for improved accuracy

## License

This implementation is for educational and research purposes.

## Citation

If you use this implementation, please cite the original paper:
```
Pandharipande S L & Badhe Y P (2012). 
"Artificial Neural Network (ANN) Modeling for Adsorption of Dyes on Rice Husk Carbon"
International Journal of Computer Applications, Volume 41, No.4, March 2012
```

## Contact

For questions or issues, please refer to the original paper or contact the repository maintainer.
