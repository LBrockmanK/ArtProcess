import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
# from huggingface_hub import login

# Log in to Hugging Face using your token
token = os.getenv('HUGGING_FACE_TOKEN')
# login(token=token)

model_id = "Salesforce/blip2-opt-6.7b-coco"
processor = Blip2Processor.from_pretrained(model_id, use_auth_token=token)
model = Blip2ForConditionalGeneration.from_pretrained(model_id, use_auth_token=token)

image_folder = "Inputs"
tags = ["Male", "Female", "Animal", "Blue"]  # List of tags

# Create a prompt with available tags
prompt = f"Question: Which of these tags apply to the image [{', '.join(tags)}]? Answer: ["
print(f"Prompt: {prompt}")

for image_name in os.listdir(image_folder):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path)

        inputs = processor(images=image, text=prompt, return_tensors="pt")

        outputs = model.generate(**inputs)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        print(f"Tags for {image_name}: {generated_text}")
