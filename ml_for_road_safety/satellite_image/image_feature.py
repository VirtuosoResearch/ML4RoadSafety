# %%
import torch
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image

model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)

image_path = "/home/michael/project/ML4RoadSafety/ml_for_road_safety/image_sample_MD.png"
image = Image.open(image_path).convert("RGB")

inputs = feature_extractor(images=image, return_tensors="pt")

print("inputs.shape: ",inputs['pixel_values'].shape)

with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # shape: (1, 197, 768)
    cls_embedding = last_hidden_state[0, 0] # [CLS]
    print(last_hidden_state.shape)
print("CLS embedding shape:", cls_embedding.shape)
