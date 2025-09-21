import os

from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from PIL import Image, ImageDraw, ImageFont

model_name = "google/vit-base-patch16-224"

image_processor = AutoImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

inp_dir = "./savos_nuotraukos/"
out_dir = "./rezultatai_savi_duomenys/"
os.makedirs(out_dir, exist_ok=True) # Jei nebutu sukurto aplanko, kad nemestu klaidos

font = ImageFont.load_default()

for fname in os.listdir(inp_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # Nuotraukos paemimas ir apdorjimas
    image = Image.open(os.path.join(inp_dir, fname)).convert("RGB")
    
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

    # Rezultatu surasymas i nuotrauka ir issaugojimas nuotraukos i isejimo aplanka
    draw = ImageDraw.Draw(image)
    text = f"{predicted_label}: {probs[0,predicted_class_idx]:.4f}"
    draw.rectangle([10, 10, 10 + len(text)*10, 30], fill="black")
    draw.text((15, 15), text, fill="white", font=font)

    image.save(os.path.join(out_dir, fname))
