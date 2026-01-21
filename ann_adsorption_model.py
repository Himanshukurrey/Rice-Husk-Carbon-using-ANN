"""
ANN Model for Adsorption of Dyes on Rice Husk Carbon
Based on the paper: International Journal of Computer Applications (0975 – 8887)
Volume 41 – No.4, March 2012

This implementation uses PyTorch to predict:
1. Equilibrium concentration of adsorbates
2. % adsorption of dyes

Input Parameters:
- Adsorbate coding (30=Bromocresol Red, 60=Alizarin Red, 80=Malachite Green, 90=Methylene Blue)
- Initial concentration (mg/lit)
- Amount of adsorbent (gm)

Output Parameters:
- Equilibrium adsorption per unit adsorbent (qe in mg/gm)
- % adsorption
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class AdsorptionANN(nn.Module):
    """
    Artificial Neural Network for Adsorption Prediction
    Architecture: 3 inputs -> 5 hidden -> 5 hidden -> 2 outputs
    """
    
    def __init__(self, input_size=3, hidden1_size=5, hidden2_size=5, output_size=2):
        super(AdsorptionANN, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        
        # Activation function (using sigmoid as per traditional ANN)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through the network"""
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)  # No activation on output layer for regression
        return x


def create_training_data():
    """
    Create training dataset from Table 3 of the paper
    Returns: numpy arrays of inputs and outputs
    """
    training_data = [
        # Bromocresol Red (coding: 30)
        [30, 1000, 2, 39, 78],
        [30, 1000, 4, 21.5, 86],
        [30, 1000, 6, 14.91667, 89.5],
        [30, 1000, 10, 9.375, 93.75],
        
        # Alizarin Red (coding: 60)
        [60, 100, 2, 3.85, 77],
        [60, 100, 4, 2.188, 87.5],
        [60, 100, 6, 1.576, 94.58],
        [60, 100, 10, 0.988, 98.75],
        
        # Malachite Green (coding: 80)
        [80, 500, 0.5, 98.75, 98.75],
        [80, 500, 1, 49.583, 99.17],
        [80, 500, 1.5, 33.056, 99.17],
        [80, 500, 2, 24.844, 99.38],
        [80, 500, 3, 16.597, 99.58],
        [80, 500, 4, 12.448, 99.58],
        
        # Methylene Blue (coding: 90)
        [90, 500, 1, 48.464, 96.93],
        [90, 500, 2, 24.974, 99.9],
        [90, 500, 2.5, 19.99, 99.95],
        [90, 500, 3, 16.667, 100],
    ]
    
    data = np.array(training_data, dtype=np.float32)
    X_train = data[:, :3]  # First 3 columns: inputs
    y_train = data[:, 3:]  # Last 2 columns: outputs
    
    return X_train, y_train


def create_test_data():
    """
    Create test dataset from Table 4 of the paper
    Returns: numpy arrays of inputs and outputs
    """
    test_data = [
        [30, 1000, 8, 11.5625, 92.5],
        [60, 100, 8, 1.224, 97.92],
        [80, 500, 2.5, 19.875, 99.38],
        [90, 500, 1.5, 33.021, 99.06],
    ]
    
    data = np.array(test_data, dtype=np.float32)
    X_test = data[:, :3]
    y_test = data[:, 3:]
    
    return X_test, y_test


def train_model(model, X_train, y_train, epochs=50000, learning_rate=0.3, 
                momentum1=0.75, momentum2=0.01):
    """
    Train the ANN model
    
    Args:
        model: PyTorch model
        X_train: Training inputs
        y_train: Training outputs
        epochs: Number of training iterations
        learning_rate: Learning rate for optimizer
        momentum1: First momentum factor
        momentum2: Second momentum factor (weight decay)
    """
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                          momentum=momentum1, weight_decay=momentum2)
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 5000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return losses


def evaluate_model(model, X, y, scaler_X, scaler_y, dataset_name=""):
    """
    Evaluate model performance and calculate metrics
    
    Args:
        model: Trained PyTorch model
        X: Input data (normalized)
        y: True output data (normalized)
        scaler_X: Scaler for inputs
        scaler_y: Scaler for outputs
        dataset_name: Name of dataset for printing
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()
    
    # Denormalize predictions and actual values
    y_actual = scaler_y.inverse_transform(y.numpy())
    y_pred = scaler_y.inverse_transform(predictions)
    
    # Calculate metrics
    mse = np.mean((y_actual - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate relative errors
    relative_errors = np.abs((y_actual - y_pred) / (y_actual + 1e-10)) * 100
    
    print(f"\n{dataset_name} Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE: {mse:.4f}")
    print("\nActual vs Predicted Values:")
    print("=" * 80)
    print(f"{'qe (actual)':<15} {'qe (predicted)':<15} {'% ads (actual)':<15} {'% ads (predicted)':<15}")
    print("=" * 80)
    
    for i in range(len(y_actual)):
        print(f"{y_actual[i, 0]:<15.4f} {y_pred[i, 0]:<15.4f} "
              f"{y_actual[i, 1]:<15.4f} {y_pred[i, 1]:<15.4f}")
    
    return y_actual, y_pred, relative_errors


def plot_results(y_actual_train, y_pred_train, y_actual_test, y_pred_test,
                 rel_error_train, rel_error_test, losses):
    """
    Create visualization plots similar to the paper
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Training - Equilibrium Concentration
    ax1 = plt.subplot(3, 3, 1)
    indices_train = np.arange(len(y_actual_train))
    ax1.plot(indices_train, y_actual_train[:, 0], 'bo-', label='Actual', markersize=8)
    ax1.plot(indices_train, y_pred_train[:, 0], 'rs-', label='Predicted', markersize=8)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Equilibrium Concentration (qe)')
    ax1.set_title('Training: Equilibrium Concentration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training - % Adsorption
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(indices_train, y_actual_train[:, 1], 'bo-', label='Actual', markersize=8)
    ax2.plot(indices_train, y_pred_train[:, 1], 'rs-', label='Predicted', markersize=8)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('% Adsorption')
    ax2.set_title('Training: % Adsorption')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Test - Equilibrium Concentration
    ax3 = plt.subplot(3, 3, 3)
    indices_test = np.arange(len(y_actual_test))
    ax3.plot(indices_test, y_actual_test[:, 0], 'bo-', label='Actual', markersize=8)
    ax3.plot(indices_test, y_pred_test[:, 0], 'rs-', label='Predicted', markersize=8)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Equilibrium Concentration (qe)')
    ax3.set_title('Test: Equilibrium Concentration')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Test - % Adsorption
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(indices_test, y_actual_test[:, 1], 'bo-', label='Actual', markersize=8)
    ax4.plot(indices_test, y_pred_test[:, 1], 'rs-', label='Predicted', markersize=8)
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('% Adsorption')
    ax4.set_title('Test: % Adsorption')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Relative Error - Training qe
    ax5 = plt.subplot(3, 3, 5)
    ax5.bar(indices_train, rel_error_train[:, 0], color='steelblue', alpha=0.7)
    ax5.set_xlabel('Sample Index')
    ax5.set_ylabel('% Relative Error')
    ax5.set_title('Training: Relative Error (qe)')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Relative Error - Training % adsorption
    ax6 = plt.subplot(3, 3, 6)
    ax6.bar(indices_train, rel_error_train[:, 1], color='coral', alpha=0.7)
    ax6.set_xlabel('Sample Index')
    ax6.set_ylabel('% Relative Error')
    ax6.set_title('Training: Relative Error (% Adsorption)')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Relative Error - Test qe
    ax7 = plt.subplot(3, 3, 7)
    ax7.bar(indices_test, rel_error_test[:, 0], color='steelblue', alpha=0.7)
    ax7.set_xlabel('Sample Index')
    ax7.set_ylabel('% Relative Error')
    ax7.set_title('Test: Relative Error (qe)')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Relative Error - Test % adsorption
    ax8 = plt.subplot(3, 3, 8)
    ax8.bar(indices_test, rel_error_test[:, 1], color='coral', alpha=0.7)
    ax8.set_xlabel('Sample Index')
    ax8.set_ylabel('% Relative Error')
    ax8.set_title('Test: Relative Error (% Adsorption)')
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Training Loss Curve
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(losses, color='green', linewidth=1)
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('MSE Loss')
    ax9.set_title('Training Loss Curve')
    ax9.set_yscale('log')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/himan/Rice_Husk_Carbon/ann_results.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved to: /home/himan/Rice_Husk_Carbon/ann_results.png")
    plt.show()


def main():
    """Main execution function"""
    print("=" * 80)
    print("ANN Model for Adsorption of Dyes on Rice Husk Carbon")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading Data...")
    X_train_raw, y_train_raw = create_training_data()
    X_test_raw, y_test_raw = create_test_data()
    
    print(f"Training samples: {len(X_train_raw)}")
    print(f"Test samples: {len(X_test_raw)}")
    
    # Normalize data
    print("\n2. Normalizing Data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train_scaled)
    y_train = torch.FloatTensor(y_train_scaled)
    X_test = torch.FloatTensor(X_test_scaled)
    y_test = torch.FloatTensor(y_test_scaled)
    
    # Initialize model
    print("\n3. Initializing ANN Model...")
    print("Architecture: 3 inputs -> 5 hidden -> 5 hidden -> 2 outputs")
    model = AdsorptionANN(input_size=3, hidden1_size=5, hidden2_size=5, output_size=2)
    print(model)
    
    # Train model
    print("\n4. Training Model...")
    print("Learning rate: 0.3")
    print("Momentum factors: 0.75, 0.01")
    losses = train_model(model, X_train, y_train, epochs=50000, learning_rate=0.3,
                        momentum1=0.75, momentum2=0.01)
    
    # Evaluate on training data
    print("\n5. Evaluating Model...")
    y_actual_train, y_pred_train, rel_error_train = evaluate_model(
        model, X_train, y_train, scaler_X, scaler_y, "Training Data"
    )
    
    # Evaluate on test data
    y_actual_test, y_pred_test, rel_error_test = evaluate_model(
        model, X_test, y_test, scaler_X, scaler_y, "Test Data"
    )
    
    # Plot results
    print("\n6. Generating Plots...")
    plot_results(y_actual_train, y_pred_train, y_actual_test, y_pred_test,
                 rel_error_train, rel_error_test, losses)
    
    # Save model
    print("\n7. Saving Model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
    }, '/home/himan/Rice_Husk_Carbon/ann_model.pth')
    print("Model saved to: /home/himan/Rice_Husk_Carbon/ann_model.pth")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
