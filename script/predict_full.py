# predict_full.py
from PIL import Image
import torch
import os
from transformers import CLIPProcessor, CLIPModel
# Pastikan 'predict_image' ada di PYTHONPATH atau diimpor dengan benar
from predict_image import predict_image_file 

# ===============================
# Load CLIP model (sekali saja)
# ===============================
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    # Jika gagal, atur ke None, atau tangani sesuai kebutuhan aplikasi
    clip_model, clip_processor = None, None


def check_relevance(text, image_path, threshold=0.3):
    """
    Hitung relevansi teks dan gambar menggunakan CLIP.
    Menerima image_path (string).
    """
    if not clip_model or not clip_processor:
        # Jika model gagal dimuat saat startup
        raise RuntimeError("CLIP model failed to load.")

    # Use 'with' to ensure the file is closed and the lock is released
    with Image.open(image_path) as image:
        image = image.convert("RGB") 
        inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
    
    # Normalisasi embeddings
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    # Normalisasi L2
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    # Hitung Cosine Similarity
    similarity = (text_embeds @ image_embeds.T).item()
    is_relevant = similarity >= threshold
    return similarity, is_relevant

def predict_full(text_response, image_file, threshold=0.3):
    """
    Menggabungkan hasil prediksi text & image, cek relevansi menggunakan CLIP.
    - text_response: dict dari predict_text / API predict_text
    - image_file: werkzeug FileStorage dari request.files['image']
    """
    text = text_response.get("text", "")
    
    if not text or not image_file:
        return {
            "error": "Teks atau file gambar tidak tersedia"
        }

    # 1. Simpan sementara file image
    TMP_DIR = "./tmp"
    os.makedirs(TMP_DIR, exist_ok=True)
    tmp_path = os.path.join(TMP_DIR, image_file.filename)
    
    # Pastikan kursor file dikembalikan ke awal sebelum disimpan
    image_file.seek(0)
    image_file.save(tmp_path)

    # Inisialisasi variabel di luar try block untuk mencegah error 'local variable referenced before assignment'
    image_result = None
    score = None
    is_relevant = False
    final_result = None

    try:
        # 2. Prediksi image (ASUMSI: predict_image_file menerima string path)
        image_result = predict_image_file(tmp_path) 

        # 3. Cek relevansi dengan CLIP
        score, is_relevant = check_relevance(text, tmp_path, threshold=threshold)

        final_result = {
            "text": text_response,
            "image": image_result,
            "relevance_score": score,
            "image_relevant": is_relevant
        }
    except Exception as e:
        # Menangkap error dari predict_image_file atau check_relevance
        final_result = {
            "text": text_response,
            "image": image_result, # Mungkin None atau hasil parsial
            "error": f"CLIP relevance check or image prediction failed: {str(e)}"
        }
    finally:
        # 4. Hapus file sementara
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return final_result