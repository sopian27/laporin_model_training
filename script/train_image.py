import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier # <-- BARU: Import VotingClassifier
from tqdm import tqdm
import joblib 
import glob

# =================================================================
# BAGIAN 1: KONFIGURASI & EKSTRAKSI FITUR
# =================================================================

# --- KONFIGURASI PATH ---
BASE_DIR = '../dataset/SPAM IMAGE dataset' 
FOLDER_MAP = {
    'spam': os.path.join(BASE_DIR, 'SpamImages', 'SpamImages'),
    'non_spam': os.path.join(BASE_DIR, 'NaturalImages', 'NaturalImages')
}

# --- PARAMETER HOG & IMAGE ---
IMAGE_SIZE = (64, 64) 
HOG_ORIENTATIONS = 9 
HOG_PIXELS_PER_CELL = (8, 8)

features = []
labels = []

# --- Fungsi Ekstraksi Fitur HOG (Harus sama persis di kedua script) ---
def extract_hog_features(image_path):
    try:
        image = imread(image_path, as_gray=True)
        resized_image = resize(image, IMAGE_SIZE, anti_aliasing=True)
        hog_features = hog(resized_image, 
                           orientations=HOG_ORIENTATIONS, 
                           pixels_per_cell=HOG_PIXELS_PER_CELL,
                           cells_per_block=(2, 2), 
                           transform_sqrt=True, 
                           feature_vector=True)
        return hog_features
    except Exception:
        return None

print("Memulai ekstraksi fitur HOG dan pelatihan model...")

# --- Pemuatan dan Ekstraksi Data ---
for label, folder_path in FOLDER_MAP.items():
    if not os.path.exists(folder_path):
        print(f"❌ ERROR: Folder '{folder_path}' tidak ditemukan.")
        continue
    
    for filename in tqdm(os.listdir(folder_path), desc=f"Memproses {label}"):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            hog_vector = extract_hog_features(image_path)
            
            if hog_vector is not None:
                features.append(hog_vector)
                labels.append(label)

if not features:
    print("\n❌ Gagal mengekstrak fitur. Program dihentikan.")
    exit()

X = np.array(features)
y = np.array(labels)
print(f"\n✅ Ekstraksi selesai. Total sampel: {len(y)}. Ukuran vektor fitur: {X.shape[1]}")
vector_size = X.shape[1] 

# --- PELATIHAN MODEL ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 2. Pelatihan dan Evaluasi Model SVM ---
svm_model = SVC(kernel='linear', random_state=42, probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
svm_report = classification_report(y_test, y_pred_svm, output_dict=True)
svm_f1_spam = svm_report['spam']['f1-score']
print(f"\n--- Hasil Evaluasi Model SVM ---")
print(f"Akurasi: {accuracy_score(y_test, y_pred_svm):.4f} | F1-Spam: {svm_f1_spam:.4f}")

# --- 3. Pelatihan dan Evaluasi Model Decision Tree ---
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
dt_report = classification_report(y_test, y_pred_dt, output_dict=True)
dt_f1_spam = dt_report['spam']['f1-score']
print(f"\n--- Hasil Evaluasi Model Decision Tree ---")
print(f"Akurasi: {accuracy_score(y_test, y_pred_dt):.4f} | F1-Spam: {dt_f1_spam:.4f}")


# =================================================================
# BAGIAN BARU: PELATIHAN VOTING CLASSIFIER
# =================================================================

# Membuat list pasangan (nama model, objek model)
estimators = [
    ('svm', svm_model),
    ('dt', dt_model)
]

# Inisialisasi dan pelatihan Voting Classifier (Soft Voting)
# Soft Voting menggunakan probabilitas, sehingga SVM harus punya probability=True
voting_model = VotingClassifier(estimators=estimators, voting='soft')
voting_model.fit(X_train, y_train)
y_pred_vote = voting_model.predict(X_test)
vote_report = classification_report(y_test, y_pred_vote, output_dict=True)
vote_f1_spam = vote_report['spam']['f1-score']

print(f"\n--- Hasil Evaluasi Model Voting Classifier (Ensemble) ---")
print(f"Akurasi: {accuracy_score(y_test, y_pred_vote):.4f} | F1-Spam: {vote_f1_spam:.4f}")

# =================================================================
# BAGIAN 4: PENYIMPANAN MODEL
# =================================================================
MODEL_DIR = './image_models'
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    # 1. Simpan Tiga Model
    joblib.dump(svm_model, os.path.join(MODEL_DIR, 'svm_spam_model.joblib'))
    joblib.dump(dt_model, os.path.join(MODEL_DIR, 'dt_spam_model.joblib'))
    joblib.dump(voting_model, os.path.join(MODEL_DIR, 'voting_spam_model.joblib')) # <-- Voting Model Disimpan
    
    # 2. Simpan Ukuran Vektor Fitur
    with open(os.path.join(MODEL_DIR, 'vector_size.txt'), 'w') as f:
        f.write(str(vector_size))
        
    print(f"\n✅ Semua model berhasil disimpan di folder: {MODEL_DIR}")
    print("\nSekarang Anda dapat menjalankan script 'predict_image.py' yang baru.")
    
except Exception as e:
    print(f"❌ Gagal menyimpan model: {e}")