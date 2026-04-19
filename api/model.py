from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model() -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
    )
    return model


class OralCancerPredictor:
    def __init__(self, model_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        tensor = self._process_image(image)
        logits = self.model(tensor).squeeze(-1)
        probability = float(torch.sigmoid(logits).item())
        predicted_label = "Cancer" if probability >= 0.5 else "Normal"

        return {
            "predicted_label": predicted_label,
            "probability_cancer": probability,
            "probability_normal": float(1.0 - probability),
            "confidence": float(probability if probability >= 0.5 else 1.0 - probability),
        }


MODEL_PATH = Path(__file__).resolve().parent.parent / "modelos" / "mobilenet_con_aug" / "best.pt"
PREDICTOR = OralCancerPredictor(MODEL_PATH)


def predict_oral_cancer(image: Image.Image) -> dict:
    return PREDICTOR.predict(image)
