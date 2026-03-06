import torch
import json
import numpy as np
import math
from pathlib import Path

# Import our defined model from models/model.py
from models.model import VisualWeightModel

def get_color_features_from_image(image_path: str, k_colors: int = 10):
    """
    **This is a placeholder function for development**
    
    In the future, this function will perform the following tasks:
    1. Read the image from image_path.
    2. Use scikit-learn's K-Means to extract `k_colors` main colors from the image.
    3. For each color, calculate its features:
       - Pixel Ratio
       - Spatial distribution / Centrality
       - OKLCH Lightness (L), Chroma (C), and Hue (H)
    4. Convert H to hue_sin and hue_cos.
    5. Return the calculated colors and their 6 features.

    Currently, it only returns a predefined set of dummy data for process testing.
    """
    # [Security] Prevent Path Traversal attacks
    base_dir = Path("dataset").resolve()
    target_path = Path(image_path).resolve()
    
    # Check if the target path is under base_dir (If you don't have a dataset folder locally, temporarily comment out these two lines for testing)
    # if not target_path.is_relative_to(base_dir):
    #     raise ValueError(f"Security Error: Invalid access path '{image_path}'")

    print(f"INFO: Extracting color features from '{target_path}' (using dummy data)...")
    
    # Assumed K-Means output (HEX values of 10 colors)
    dummy_colors_hex = [
        "#1A2B3C", "#C0392B", "#F1C40F", "#2ECC71", "#3498DB",
        "#9B59B6", "#34495E", "#E67E22", "#BDC3C7", "#7F8C8D"
    ]
    
    # Dummy raw features before processing
    # [ratio, centrality, L, C, H]
    raw_features_list = [
        [0.30, 0.8, 20.0, 15.0, 250.0],  # Dark color, centered, blueish
        [0.10, 0.2, 55.0, 80.0, 20.0],   # Small ratio, edge, vibrant red
        [0.05, 0.9, 90.0, 95.0, 85.0],   # Tiny ratio, centered, bright vibrant yellow
        [0.15, 0.6, 60.0, 40.0, 150.0],  # Green
        [0.12, 0.5, 70.0, 50.0, 220.0],  # Blue
        [0.08, 0.7, 45.0, 60.0, 280.0],  # Purple
        [0.07, 0.4, 30.0, 10.0, 40.0],   # Brownish
        [0.05, 0.3, 65.0, 70.0, 35.0],   # Orange
        [0.04, 0.5, 85.0, 5.0, 200.0],   # Light blue/grey
        [0.04, 0.6, 50.0, 8.0, 180.0],   # Greyish green
    ]

    # Process features: normalize and convert Hue
    processed_features = []
    total_ratio = sum(item[0] for item in raw_features_list)

    for ratio, centrality, l, c, h in raw_features_list:
        # Normalize features
        norm_ratio = ratio / total_ratio
        norm_l = l / 100.0
        norm_c = c / 150.0 # Assuming max chroma
        
        # Convert Hue to sin/cos
        h_rad = math.radians(h)
        hue_sin = math.sin(h_rad)
        hue_cos = math.cos(h_rad)
        
        processed_features.append([
            norm_ratio, centrality, norm_l, norm_c, hue_sin, hue_cos
        ])

    dummy_features_tensor = torch.tensor(processed_features, dtype=torch.float32)

    return dummy_colors_hex, dummy_features_tensor


def inference(image_path: str):
    """
    Executes the complete inference pipeline: feature extraction -> model prediction -> sorted output.
    """
    # 1. (Placeholder) Feature Extraction
    colors, features = get_color_features_from_image(image_path)
    
    # 2. Initialize PyTorch Model
    input_dim = features.shape[1]
    model = VisualWeightModel(input_features=input_dim)

    # [Security] Use weights_only=True to prevent Pickle vulnerabilities when loading trained weights
    # try:
    #     model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
    #     print("INFO: Successfully loaded model weights.")
    # except FileNotFoundError:
    #     print("WARNING: 'model_weights.pth' not found. Using random initial weights.")

    model.eval() # Set to evaluation mode

    # 3. Phase 2: Predict Weights
    with torch.no_grad():
        visual_weights = model(features)
    
    # 4. Combine and Sort
    scores = visual_weights.squeeze().numpy()
    
    results = []
    for i, color_hex in enumerate(colors):
        results.append({
            "color_hex": color_hex,
            "visual_weight": round(float(scores[i]), 4),
            "raw_features": {
                "pixel_ratio": round(float(features[i, 0]), 4),
                "centrality": round(float(features[i, 1]), 4),
                "L": round(float(features[i, 2]), 4),
                "C": round(float(features[i, 3]), 4),
                "hue_sin": round(float(features[i, 4]), 4),
                "hue_cos": round(float(features[i, 5]), 4)
            }
        })
        
    # Sort results by visual_weight in descending order
    sorted_results = sorted(results, key=lambda x: x["visual_weight"], reverse=True)
    
    # 5. Output the final JSON
    return json.dumps(sorted_results, indent=2)


if __name__ == "__main__":
    # Simulate running the main pipeline on a dummy image path
    DUMMY_IMAGE_PATH = "dataset/example_image.jpg"
    
    print("="*50)
    print("Executing Iris-Core Inference Pipeline (with Security & Hue features)...")
    print("="*50)
    
    final_ranked_colors = inference(DUMMY_IMAGE_PATH)
    
    print("\n--- Final Ranked Colors (JSON) ---")
    print(final_ranked_colors)
    print("\nPipeline skeleton execution complete.")