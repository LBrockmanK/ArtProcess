from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import os
import shutil
import json
from PIL import Image
import piexif
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

def is_ngrok_running():
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        return True
    except:
        return False

# TODO: ngrok is temporary for development, but it would be good to get this properly working with path, subprocess seems to fail. Fix this or find alternative method for https


def start_ngrok():
    if not is_ngrok_running():
        subprocess.Popen(
            "start cmd /k \"C:\\Program Files\\ngrok\\ngrok.exe\" http 8000", shell=True)
        time.sleep(2)


def get_ngrok_url():
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        data = response.json()
        public_url = data["tunnels"][0]["public_url"]
        return public_url
    except Exception as e:
        print(f"An error occurred while fetching ngrok URL: {e}")
        return None


def get_public_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json')
        ip = response.json()['ip']
        return ip
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to embed tags into image metadata
def embed_metadata(image_path, tags):
    image = Image.open(image_path)
    exif_dict = piexif.load(image.info.get('exif', b''))
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

@app.get("/getImage")
def get_image():
    input_folder = os.path.join(os.getcwd(), 'Inputs')
    try:
        for image_name in os.listdir(input_folder):
            image_path = os.path.join(input_folder, image_name)
            if os.path.isfile(image_path):
                return FileResponse(image_path)
        raise HTTPException(status_code=404, detail="No images found in the Inputs folder")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TagList(BaseModel):
    tags: list[str]

@app.post("/embedTags")
async def embed_tags(tags: TagList, file: UploadFile = File(...)):
    try:
        input_folder = os.path.join(os.getcwd(), 'Inputs')
        input_image_path = os.path.join(input_folder, file.filename)

        # Save the uploaded file to the Inputs folder
        with open(input_image_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Embed metadata and move the file to the Outputs folder
        embed_metadata(input_image_path, tags.tags)
        move_to_output(input_image_path)

        return {"status": f"Tags embedded and {file.filename} moved to Outputs folder"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
