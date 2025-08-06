from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from dotenv import load_dotenv
from flask_cors import CORS
from google.cloud import storage
import io

load_dotenv()

app = Flask(__name__)
CORS(app)

# L'environnement des variables
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

# Charger le modèle de segmentation
model = tf.keras.models.load_model("vgg16_unet_final.h5")

# Taille d'entrée du modèle
INPUT_SIZE = (256, 256)

# Initialiser le client Cloud Storage
storage_client = storage.Client()
bucket_name = "segmentation-image-bucket1"
bucket = storage_client.bucket(bucket_name)

@app.route('/')
def home():
    return "Bienvenue sur l'API de Segmentation d'image"

@app.route('/images', methods=['GET'])
def get_image_ids():
    blobs = storage_client.list_blobs(bucket_name, prefix="test_images/")
    image_ids = [blob.name.split('/')[-1].split('.')[0] for blob in blobs if blob.name.endswith('.png')]
    return jsonify({"image_ids": image_ids})

@app.route('/predict/<image_id>', methods=['GET'])
def predict(image_id):
    blob = bucket.blob(f"test_images/{image_id}.png")
    if not blob.exists():
        return jsonify({"error": "Image not found"}), 404

    # Télécharger l'image en mémoire
    image_data = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_data))
    image = image.resize(INPUT_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Faire la prédiction
    prediction = model.predict(image)
    predicted_mask = np.argmax(prediction, axis=-1)[0]

    # Sauvegarder le masque prédit dans Cloud Storage
    mask_image = Image.fromarray((predicted_mask * 255 / 7).astype(np.uint8))
    mask_buffer = io.BytesIO()
    mask_image.save(mask_buffer, format='PNG')
    mask_blob = bucket.blob(f"predicted_masks/{image_id}_mask.png")
    mask_blob.upload_from_string(mask_buffer.getvalue(), content_type='image/png')

    return jsonify({
        "message": "Prediction successful",
        "image_url": blob.public_url,
        "mask_url": mask_blob.public_url
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
