import os
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import joblib


# ==========================
# KONFIGURASI & MUAT MODEL
# ==========================
MODEL_DIR = './image_models'
IMAGE_SIZE = (64, 64)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)

# Membuat folder sementara untuk upload
TMP_DIR = './tmp_images'
os.makedirs(TMP_DIR, exist_ok=True)

# Muat vector size
with open(os.path.join(MODEL_DIR, 'vector_size.txt'), 'r') as f:
    VECTOR_SIZE = int(f.read())

# Muat semua model
SVM_MODEL = joblib.load(os.path.join(MODEL_DIR, 'svm_spam_model.joblib'))
DT_MODEL = joblib.load(os.path.join(MODEL_DIR, 'dt_spam_model.joblib'))
VOTING_MODEL = joblib.load(os.path.join(MODEL_DIR, 'voting_spam_model.joblib'))

# ==========================
# Fungsi ekstraksi HOG
# ==========================
def extract_hog_features(image_path):
    """Baca gambar, ubah ukuran, dan ekstrak fitur HOG."""
    image = imread(image_path, as_gray=True)
    resized_image = resize(image, IMAGE_SIZE, anti_aliasing=True)
    features = hog(resized_image,
                   orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=(2, 2),
                   transform_sqrt=True,
                   feature_vector=True)
    return features

# ==========================
# Fungsi prediksi gambar
# ==========================
def predict_image_file(tmp_path):
    """
    Prediksi gambar menggunakan semua model (SVM, Decision Tree, Voting).
    Parameter:
        image_file : werkzeug FileStorage (dari Flask request.files['image'])
    """
    # Simpan sementara


    try:
        hog_vector = extract_hog_features(tmp_path)
        if hog_vector.shape[0] != VECTOR_SIZE:
            raise ValueError(f"Ukuran vektor fitur ({hog_vector.shape[0]}) tidak cocok dengan pelatihan ({VECTOR_SIZE})")
        hog_vector = hog_vector.reshape(1, -1)

        result = {
            'svm': SVM_MODEL.predict(hog_vector)[0],
            'decision_tree': DT_MODEL.predict(hog_vector)[0],
            'voting': VOTING_MODEL.predict(hog_vector)[0]
        }
    except Exception as e:
        final_result = {
            "text": 'text_response',
            "image": None,
            "error": f"CLIP relevance check failed: {str(e)}"
        }

    return result

