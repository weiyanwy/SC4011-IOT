from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()
# Serve static files like CSS, JS, images from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load pre-trained MNIST model
model = tf.keras.models.load_model("mnist_model.h5")

def preprocess_image(image: Image.Image):
    image = image.convert("L").resize((28, 28))  # Convert to grayscale & resize
    img_array = np.array(image) / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model
    return img_array

@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    processed_image = preprocess_image(image)
    
    prediction = model.predict(processed_image)
    digit = np.argmax(prediction)
    
    return {"digit": int(digit)}

@app.get("/test/")
async def test():
    return 1