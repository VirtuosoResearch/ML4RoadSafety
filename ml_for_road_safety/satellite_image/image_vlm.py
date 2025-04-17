# %%
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_path = "/home/michael/project/ML4RoadSafety/ml_for_road_safety/satellite_image/image_sample_MD.png"
raw_image = Image.open(image_path).convert('RGB')

print("raw_image.shape: ",raw_image.size)

inputs = processor(images=raw_image, return_tensors="pt")

print("inputs.shape: ",inputs['pixel_values'].shape)

with torch.no_grad():
    output = model.generate(**inputs)

print("output",output)
caption = processor.decode(output[0], skip_special_tokens=True)
print("Output: ", caption)

# %%
from transformers import OFATokenizer, OFAModel
from PIL import Image
import torch
from torchvision import transforms

tokenizer = OFATokenizer.from_pretrained("OFA-Sys/OFA-base")
model = OFAModel.from_pretrained("OFA-Sys/OFA-base").eval()

image = Image.open("/home/michael/project/ML4RoadSafety/ml_for_road_safety/satellite_image/image_sample_MD.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
image_tensor = transform(image).unsqueeze(0)

question = "What is in this image?"
inputs = tokenizer(question, return_tensors="pt")
inputs['pixel_values'] = image_tensor

with torch.no_grad():
    generated = model.generate(**inputs)
    answer = tokenizer.decode(generated[0], skip_special_tokens=True)

print("Answer:", answer)

# %%
from transformers import OFATokenizer, OFAModel
from PIL import Image
import torch
from torchvision import transforms

tokenizer = OFATokenizer.from_pretrained("OFA-Sys/OFA-base")
model = OFAModel.from_pretrained("OFA-Sys/OFA-base").eval()

image = Image.open("image_sample_MD.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
image_tensor = transform(image).unsqueeze(0)

question = "What is in this image?"
inputs = tokenizer(question, return_tensors="pt")
inputs['pixel_values'] = image_tensor

with torch.no_grad():
    generated = model.generate(**inputs)
    answer = tokenizer.decode(generated[0], skip_special_tokens=True)

print("Answer:", answer)