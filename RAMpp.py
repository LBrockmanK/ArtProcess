'''
 * The Recognize Anything Plus Model (RAM++) inference on unseen classes
 * Written by Xinyu Huang
'''
import numpy as np
import torch
import os
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram_openset as inference
from ram import get_transform
from ram.utils import build_openset_llm_label_embedding
from torch import nn
import json

def process_images_in_folder(folder_path, model, transform, device):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            image = transform(Image.open(image_path)).unsqueeze(0).to(device)
            res = inference(image, model)
            print(f"Image Tags for {filename}: ", res)

if __name__ == "__main__":
    image_folder = 'Inputs'  # Folder containing images
    pretrained_model_path = 'C:/Users/cpnbe/Documents/image_classification_models/ram_plus_swin_large_14m.pth'
    llm_tag_des_path = 'Categories.json'
    image_size = 384

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform(image_size=image_size)

    # Load model
    model = ram_plus(pretrained=pretrained_model_path, image_size=image_size, vit='swin_l')

    # Build tag embedding
    print('Building tag embedding:')
    with open(llm_tag_des_path, 'rb') as fo:
        llm_tag_des = json.load(fo)
    openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)

    model.tag_list = np.array(openset_categories)
    model.label_embed = nn.Parameter(openset_label_embedding.float())
    model.num_class = len(openset_categories)
    model.class_threshold = torch.ones(model.num_class) * 0.67
    model.eval()
    model = model.to(device)

    # Process images in the specified folder
    process_images_in_folder(image_folder, model, transform, device)
