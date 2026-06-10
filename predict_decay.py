import torch
import torch.nn as nn

class DecayPredictionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def predict(parameters):
    device = torch.device("cuda" if parameters.is_cuda else "cpu")
    
    # Scaler parameters from training
    scaler_mean = torch.tensor([393.3193, 91.0090, 1.2350, 0.0250, 161.3209, 2.2485], dtype=torch.float32, device=device)
    scaler_scale = torch.tensor([109.9246, 122.3409, 1.8969, 0.0146, 51.6276, 0.4260], dtype=torch.float32, device=device)
    
    # Standardise inputs
    parameters_norm = (parameters - scaler_mean) / scaler_scale
    
    # Load model
    model = DecayPredictionNetwork().to(device)
    model.load_state_dict(torch.load('weights.pkl', map_location=torch.device(device)))
    model.eval()
    
    with torch.no_grad():
        pred_log = model(parameters_norm)
        predicted_decay_times = torch.expm1(pred_log)
    
    return predicted_decay_times