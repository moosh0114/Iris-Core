import torch
import torch.nn as nn

class VisualWeightModel(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) to predict the visual weight of colors.
    
    Input: A tensor of shape (batch_size, input_features), where input_features include:
          - Pixel Ratio
          - Spatial distribution / Centrality
          - OKLCH Lightness (L)
          - OKLCH Chroma (C)
          - OKLCH Hue (H) as sin(H)
          - OKLCH Hue (H) as cos(H)
    Output: A tensor of shape (batch_size, 1), where each value represents the
            visual weight score (0-1) for the corresponding color.
    """
    def __init__(self, input_features=6, hidden_dim1=16, hidden_dim2=8):
        super(VisualWeightModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid()  # Compress the output to a range between 0 and 1
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        return self.network(x)

if __name__ == '__main__':
    # Test if the model works correctly
    num_colors = 10  # Assuming K-Means extracts 10 colors
    input_dim = 6    # 6 features
    
    # Create a model instance
    model = VisualWeightModel(input_features=input_dim)
    print("Model Architecture:")
    print(model)
    
    # Create a dummy input tensor
    # Represents 10 colors, each with 6 feature values
    dummy_features = torch.randn(num_colors, input_dim)
    print(f"\nInput dummy feature tensor (first 5 rows): Shape={dummy_features.shape}")
    print(dummy_features[:5])
    
    # Set the model to evaluation mode
    model.eval()
    
    # No need to calculate gradients for inference
    with torch.no_grad():
        # Perform prediction
        predicted_weights = model(dummy_features)
    
    print(f"\nModel output visual weights: Shape={predicted_weights.shape}")
    print(predicted_weights)

    assert predicted_weights.shape == (num_colors, 1)
    assert predicted_weights.min() >= 0.0
    assert predicted_weights.max() <= 1.0
    print("\nModel test successful! Output dimensions and range are correct.")
