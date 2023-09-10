import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

def train_regression(model, epochs, X_train_scaled, y_train, X_val_tensor, y_val_tensor, lr=0.001, batch_size=64, patience=10):
    """Traing with minibatching"""
    # Create DataLoader for training and validation
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.float32))
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    inputs = torch.tensor(X_train_scaled, dtype=torch.float32)
    targets = torch.tensor(y_train.values, dtype=torch.float32)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            batch_loss = criterion(outputs, targets)
            batch_loss.backward()
            optimizer.step()
        
        # Calculate train and validation loss for epoch
        model.eval()
        with torch.no_grad():
            y_train_pred = model(inputs).squeeze()
            train_loss = criterion(y_train_pred, targets)
            y_val_pred = model(X_val_tensor).squeeze()
            val_loss = criterion(y_val_pred, y_val_tensor)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
        # keep count of training loss for early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            counter = 0
        else:
            counter += 1
            
        # check if early stopping criteria has been met
        if counter >= patience:
            print(f"Early stopping after {epoch}.")
            break

def evaluate_regression(model, X_test_scaled, y_test):
    """Evaluate model performance based on actual and predicted values"""
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test_scaled, dtype=torch.float32)
        test_targets = torch.tensor(y_test.values, dtype=torch.float32)
        test_outputs = model(test_inputs).squeeze()

    mae = nn.L1Loss()
    mae = mae(test_outputs, test_targets)
    mse_o = nn.MSELoss()
    mse = mse_o(test_outputs, test_targets)
    rmse = torch.sqrt(mse_o(test_outputs, test_targets))
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')


def train_regression_(model, epochs, X_train_scaled, y_train, X_val_tensor, y_val_tensor, lr=0.001, patience=10):
    """Train without minibatching"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    counter = 0
    
    inputs = torch.tensor(X_train_scaled, dtype=torch.float32)
    targets = torch.tensor(y_train.values, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        train_loss = criterion(outputs, targets)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor).squeeze()
            val_loss = criterion(y_val_pred, y_val_tensor)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            
        # keep count of training loss for early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            counter = 0
        else:
            counter += 1
            
        # check if early stopping criteria has been met
        if counter >= patience:
            print("Early stopping")
            break