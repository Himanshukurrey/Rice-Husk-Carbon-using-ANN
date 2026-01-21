"""
Prediction script for the trained ANN model
Use this to predict adsorption for new data points
"""

import torch
import numpy as np
from ann_adsorption_model import AdsorptionANN


def load_model(model_path='/home/himan/Rice_Husk_Carbon/ann_model.pth'):
    """Load the trained model and scalers"""
    checkpoint = torch.load(model_path, weights_only=False)
    
    model = AdsorptionANN(input_size=3, hidden1_size=5, hidden2_size=5, output_size=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    
    return model, scaler_X, scaler_y


def predict(adsorbate_coding, initial_concentration, adsorbent_amount, 
            model, scaler_X, scaler_y):
    """
    Make prediction for a single data point
    
    Args:
        adsorbate_coding: 30 (Bromocresol Red), 60 (Alizarin Red), 
                         80 (Malachite Green), 90 (Methylene Blue)
        initial_concentration: Initial concentration in mg/lit
        adsorbent_amount: Amount of adsorbent in gm
        model: Trained PyTorch model
        scaler_X: Input scaler
        scaler_y: Output scaler
    
    Returns:
        equilibrium_concentration: qe in mg/gm
        percent_adsorption: % adsorption
    """
    # Prepare input
    X = np.array([[adsorbate_coding, initial_concentration, adsorbent_amount]], 
                 dtype=np.float32)
    
    # Normalize
    X_scaled = scaler_X.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    # Predict
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()
    
    # Denormalize
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    equilibrium_concentration = y_pred[0, 0]
    percent_adsorption = y_pred[0, 1]
    
    return equilibrium_concentration, percent_adsorption


def main():
    """Example usage"""
    print("=" * 80)
    print("Adsorption Prediction System")
    print("=" * 80)
    
    # Load model
    print("\nLoading trained model...")
    model, scaler_X, scaler_y = load_model()
    print("Model loaded successfully!")
    
    # Adsorbate codes
    adsorbates = {
        30: "Bromocresol Red",
        60: "Alizarin Red",
        80: "Malachite Green",
        90: "Methylene Blue"
    }
    
    print("\n" + "=" * 80)
    print("Example Predictions")
    print("=" * 80)
    
    # Example predictions
    test_cases = [
        (30, 1000, 5, "Bromocresol Red"),
        (60, 100, 5, "Alizarin Red"),
        (80, 500, 2.5, "Malachite Green"),
        (90, 500, 2, "Methylene Blue"),
    ]
    
    for adsorbate_code, init_conc, adsorbent, name in test_cases:
        qe, percent_ads = predict(adsorbate_code, init_conc, adsorbent, 
                                  model, scaler_X, scaler_y)
        
        print(f"\nAdsorbate: {name}")
        print(f"Initial Concentration: {init_conc} mg/lit")
        print(f"Adsorbent Amount: {adsorbent} gm")
        print(f"Predicted Equilibrium Concentration (qe): {qe:.4f} mg/gm")
        print(f"Predicted % Adsorption: {percent_ads:.2f}%")
        print("-" * 80)
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("Interactive Prediction Mode")
    print("=" * 80)
    print("\nAdsorbate Codes:")
    for code, name in adsorbates.items():
        print(f"  {code}: {name}")
    
    try:
        while True:
            print("\n" + "-" * 80)
            adsorbate_code = int(input("\nEnter adsorbate code (30/60/80/90) or 0 to exit: "))
            
            if adsorbate_code == 0:
                print("Exiting...")
                break
            
            if adsorbate_code not in adsorbates:
                print("Invalid adsorbate code! Please use 30, 60, 80, or 90.")
                continue
            
            init_conc = float(input("Enter initial concentration (mg/lit): "))
            adsorbent = float(input("Enter adsorbent amount (gm): "))
            
            qe, percent_ads = predict(adsorbate_code, init_conc, adsorbent,
                                     model, scaler_X, scaler_y)
            
            print(f"\n{'Results:':<40}")
            print(f"{'Adsorbate:':<40} {adsorbates[adsorbate_code]}")
            print(f"{'Equilibrium Concentration (qe):':<40} {qe:.4f} mg/gm")
            print(f"{'% Adsorption:':<40} {percent_ads:.2f}%")
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
