import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.io
import pandas as pd

from shiny import render, ui, reactive
from shiny.express import input, render, ui

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn

class SatelliteImgCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.part_block_1 = nn.Sequential(
            # Input: 64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Output: 32x32
        )

        self.part_block_2 = nn.Sequential(
            # Input: 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU() # Output: 32x32
        )

        self.res_block = nn.Sequential(
            # Input: 32x32
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) # Output: 32x32
        )
        self.res_relu = nn.ReLU()

        self.part_block_3 = nn.Sequential(
            # Input: 16x16
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Output: 8x8
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.35),
            nn.Linear(in_features=128, out_features=10)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.part_block_1(x)
        x = self.part_block_2(x)
        block_2 = x

        res_block = self.res_block(x)
        res_block += block_2
        x = self.res_relu(res_block)

        x = self.part_block_3(x)
        x = self.global_pool(x)

        return self.classifier(x)

def load_trained_model():
    m = SatelliteImgCNN() 
    
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    _ = m.load_state_dict(state_dict)
    _ = m.eval()
    return m.to(device)

model = load_trained_model()

label_dict = {
    'PermanentCrop': 0, 
    'Forest': 1, 
    'SeaLake': 2, 
    'Highway': 3, 
    'Residential': 4, 
    'Industrial': 5, 
    'Pasture': 6, 
    'River': 7, 
    'AnnualCrop': 8, 
    'HerbaceousVegetation': 9
}

index_to_class = {index: name for name, index in label_dict.items()}

ui.page_opts(title="Satellite Image Classifier (EuroSAT)", fillable=True)

with ui.sidebar():
    ui.input_file("uploaded_image", "Upload Satellite Image", accept=[".jpg", ".jpeg"])
    ui.input_action_button("run_prediction", "Run Prediction", class_="btn-primary", style="background-color: #285920")

with ui.layout_columns(col_widths=[5, 7]):
    
    with ui.card(full_screen=True):
        ui.card_header("Uploaded Image View")
        
        @render.image
        def display_uploaded_image():
            file_info = input.uploaded_image()
            if file_info is None:
                return None
            return {"src": file_info[0]["datapath"], "width": "100%", "style": "max-width: 300px; aspect-ratio: 1;"}
    
    with ui.card():
        ui.card_header("Probability Distribution Table")
        
        @render.data_frame
        def display_probability_table():
            input.run_prediction()
            
            with reactive.isolate():
                file_info = input.uploaded_image()
                if file_info is None:
                    return pd.DataFrame(columns=["Class Label", "Probability"])
                
                img_path = file_info[0]["datapath"]
                
            image = torchvision.io.read_image(img_path).float() / 255.0
            image_input = image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(image_input)
                probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            data = {
                "Class Label": [index_to_class[i] for i in range(10)],
                "Probability": [f"{p * 100:.4f}%" for p in probabilities]
            }
            df = pd.DataFrame(data)
            return df.sort_values(by="Probability", ascending=False)
        
    with ui.card():
        ui.card_header("Model Prediction Result")
        
        @render.ui
        def display_prediction_box():
            input.run_prediction()
            
            with reactive.isolate():
                file_info = input.uploaded_image()
                if file_info is None:
                    return ui.p("Please upload an image and click 'Run Prediction'.")
                
                img_path = file_info[0]["datapath"]
            
            image = torchvision.io.read_image(img_path).float() / 255.0
            image_input = image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(image_input)
                probabilities = F.softmax(logits, dim=1).squeeze(0)
            
            best_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[best_class_idx].item() * 100
            predicted_label = index_to_class[best_class_idx]
            
            return ui.div(
                ui.h2(f"Class: {predicted_label}", style="color: #1e5128; margin-bottom: 5px;"),
                ui.h3(f"Confidence: {confidence:.2f}%", style="color: #4e9f3d; margin-top: 0;")
            )