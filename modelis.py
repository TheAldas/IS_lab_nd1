from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from PIL import Image

model_name = "Jacques7103/Food-Recognition"

image_processor = AutoImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Nuotraukos paemimas ir apdorjimas
image = Image.open("./duotos_nuotraukos/apple_pie.png").convert("RGB")
inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits  # [batch_size, num_classes]
probs = torch.nn.functional.softmax(logits, dim=-1)
predicted_class_idx = torch.argmax(probs, dim=-1).item()

# Indeksu keitimas i tekstinius label'ius
labels = model.config.id2label
predicted_label = labels[predicted_class_idx]

print(f"Spejimas: {predicted_label}, tikimybe: {probs[0,predicted_class_idx]:.4f}")
