import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from huggingface_hub import login

# Log in to Hugging Face using your token
token = os.getenv('HUGGING_FACE_TOKEN')
login(token=token)

model_id = "Salesforce/blip2-opt-6.7b-coco"
processor = Blip2Processor.from_pretrained(model_id, use_auth_token=token)
model = Blip2ForConditionalGeneration.from_pretrained(model_id, use_auth_token=token)

image_folder = "Inputs"

for image_name in os.listdir(image_folder):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path)

        inputs = processor(image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        print(f"Caption for {image_name}: {caption}")