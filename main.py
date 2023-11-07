import os
import shutil
import json
from openai import OpenAI
from PIL import Image
import piexif

# Set your OpenAI API key
openai_api_key = 'your-api-key'
client = OpenAI(api_key=openai_api_key)

# Function to call OpenAI API and get image tags
def classify_image(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Provide a list of tags describing this image."},
                    {"type": "data", "data": image_data},
                ],
            }
        ],
        max_tokens=60
    )
    
    # Assuming that the response will be a list of tags in text format
    tags_text = response.choices[0].message['content']['text']
    tags = tags_text.split(', ')  # Split tags by comma and space
    return tags

# Function to embed tags into image metadata
def embed_metadata(image_path, tags):
    image = Image.open(image_path)
    exif_dict = piexif.load(image.info.get('exif', b''))
    
    # Convert tags list to JSON string for embedding
    user_comment = json.dumps(tags)
    exif_dict['Exif'][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(user_comment)
    
    exif_bytes = piexif.dump(exif_dict)
    image.save(image_path, exif=exif_bytes)

# Function to move image to the Outputs folder
def move_to_output(image_path):
    output_folder = os.path.join('Outputs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    shutil.move(image_path, os.path.join(output_folder, os.path.basename(image_path)))

# Main processing loop for images in the Inputs folder
def process_images(input_folder, output_folder):
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            try:
                tags = classify_image(image_path)
                embed_metadata(image_path, tags)
                move_to_output(image_path)
                print(f"Processed and moved: {image_name}")
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

# Define the root, input, and output folders
root_folder = os.getcwd()
input_folder = os.path.join(root_folder, 'Inputs')
output_folder = os.path.join(root_folder, 'Outputs')

# Process all images in the Inputs folder
process_images(input_folder, output_folder)
