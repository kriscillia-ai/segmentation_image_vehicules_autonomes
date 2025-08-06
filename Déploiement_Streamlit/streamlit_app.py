import streamlit as st
import requests
from PIL import Image
import io
import json
from google.cloud import storage

# URL de l'API Flask
API_URL = "https://projet-seg-img-v2-498304223612.europe-west12.run.app"

# Initialiser le client Cloud Storage
storage_client = storage.Client()
bucket_name = "segmentation-image-bucket1"
bucket = storage_client.bucket(bucket_name)

st.title("Image Segmentation App")

# Récupérer la liste des ID d'images disponibles
try:
    response = requests.get(f"{API_URL}/images")
    if response.status_code == 200:
        image_ids = response.json()["image_ids"]
    else:
        st.error(f"API request failed with status code: {response.status_code}")
        st.stop()
except requests.exceptions.RequestException as e:
    st.error(f"Error connecting to API: {e}")
    st.stop()
except json.JSONDecodeError:
    st.error("Invalid JSON response from API")
    st.error(f"Response content: {response.text}")
    st.stop()

# Sélectionner un ID d'image
if image_ids:
    selected_id = st.selectbox("Select an image ID", image_ids)

    if selected_id:
        # Faire la prédiction avec l'image du dossier test_images
        try:
            response = requests.get(f"{API_URL}/predict/{selected_id}")

            if response.status_code == 200:
                result = response.json()

                # Extraire la partie commune de l'ID
                common_id = selected_id.split("_leftImg8bit")[0]

                # Construire les chemins vers l'image réelle et le masque réel dans Cloud Storage
                real_image_blob = bucket.blob(f"test_images/{selected_id}.png")
                real_mask_blob = bucket.blob(f"test_masks/{common_id}_gtFine_labelIds.png")

                # Afficher l'image réelle
                if real_image_blob.exists():
                    real_image_data = real_image_blob.download_as_bytes()
                    real_image = Image.open(io.BytesIO(real_image_data))
                    st.image(real_image, caption="Real Image", use_column_width=True)
                else:
                    st.error(f"Real image not found: {real_image_blob.name}")

                # Afficher le masque réel (si disponible)
                if real_mask_blob.exists():
                    real_mask_data = real_mask_blob.download_as_bytes()
                    real_mask = Image.open(io.BytesIO(real_mask_data))
                    st.image(real_mask, caption="Real Mask", use_column_width=True)
                else:
                    st.error(f"Real mask not found: {real_mask_blob.name}")

                # Afficher le masque prédit
                predicted_mask_url = result["mask_url"]
                predicted_mask_response = requests.get(predicted_mask_url)
                predicted_mask = Image.open(io.BytesIO(predicted_mask_response.content))
                st.image(predicted_mask, caption="Predicted Mask", use_column_width=True)
            else:
                st.error(f"Prediction request failed with status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error during prediction: {e}")
else:
    st.warning("No image IDs available.")