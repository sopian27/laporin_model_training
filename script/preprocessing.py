# =========================================================
# KLASIFIKASI SPAM & PRIORITAS PENGADUAN MASYARAKAT
# Decision Tree & Support Vector Machine
# =========================================================

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1. KONFIGURASI
# =========================
FILE_INPUT = 'dataset/new-augmented_keluhan_masyarakat.xlsx'
SHEET_NAME = 0
RANDOM_STATE = 42

print("Memuat dataset...")

# =========================
# 2. LOAD DATASET
# =========================
df = pd.read_excel(FILE_INPUT, sheet_name=SHEET_NAME)
df.columns = df.columns.str.strip()

# Pastikan kolom penting ada
required_columns = ['Keluhan', 'Topik Keluhan', 'Dinas Terkait']
for col in required_columns:
    if col not in df.columns:
        raise Exception(f"Kolom '{col}' tidak ditemukan dalam dataset!")

# =========================
# 3. PREPROCESSING TEKS
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['keluhan_clean'] = df['Keluhan'].apply(clean_text)
df['topik_clean'] = df['Topik Keluhan'].apply(clean_text)

df['teks_gabungan'] = df['topik_clean'] + ' ' + df['keluhan_clean']

# =========================
# 4. LABELING SPAM (RULE-BASED)
# =========================
SPAM_KEYWORDS = [
    # Promo / Penipuan
    'promo', 'gratis', 'hadiah', 'diskon',
    'pinjaman', 'judi', 'slot', 'deposit',
    'investasi', 'cuan', 'bonus',
    'klik', 'link', 'whatsapp', 'telegram',

    # Test / Coba-coba
    'test', 'testing', 'tes', 'coba',
    'cek', 'percobaan',

    # Kata Ngasal / Tidak Bermakna
    'asdf', 'qwerty', 'zxcv',
    'lorem', 'ipsum',

    # Umum Spam Pendek
    'hai', 'halo', 'bro', 'gan'
]


def assign_spam_label(text):
    text = str(text).lower().strip()

    # 1. Keyword spam
    for keyword in SPAM_KEYWORDS:
        if keyword in text:
            return 1

    # 2. Terlalu pendek (<= 3 kata)
    if len(text.split()) <= 3:
        return 1

    # 3. Karakter berulang (aaaa, zzzz)
    if re.search(r'(.)\1{3,}', text):
        return 1

    # 4. Huruf acak (tidak mengandung vokal)
    if not re.search(r'[aiueo]', text) and len(text) > 6:
        return 1

    # 5. Dominasi karakter non-huruf
    non_alpha_ratio = len(re.findall(r'[^a-z\s]', text)) / max(len(text), 1)
    if non_alpha_ratio > 0.3:
        return 1

    # 6. Banyak karakter acak
    if re.fullmatch(r'[a-z]{6,}', text) and not re.search(r'[aiueo]', text):
        return 1

    return 0


df['is_spam'] = df['teks_gabungan'].apply(assign_spam_label)

print("\nDistribusi Label Spam:")
print(df['is_spam'].value_counts())

# =========================
# 5. LABELING PRIORITAS (RULE-BASED)
# =========================
def assign_priority(row):
    keluhan = row['keluhan_clean']
    topik = row['topik_clean']
    dinas = str(row['Dinas Terkait']).lower()

    high_priority_topics = [
        'kebakaran', 'bencana', 'banjir', 'longsor',
        'kecelakaan', 'kriminal', 'kekerasan',
        'kesehatan', 'odgj', 'tanah'
    ]

    low_priority_keywords = [
        'saran', 'informasi', 'apresiasi',
        'dukungan', 'mohon cek'
    ]

    if any(tp in topik for tp in high_priority_topics):
        return 'Tinggi'

    if any(lw in keluhan for lw in low_priority_keywords):
        return 'Rendah'

    if 'reklame' in topik:
        return 'Rendah'

    if 'multitopik' in topik and 'tidak diketahui' in dinas:
        return 'Rendah'

    return 'Sedang'

# Prioritas hanya untuk Non-Spam
df['prioritas'] = 'Spam'
df.loc[df['is_spam'] == 0, 'prioritas'] = (
    df[df['is_spam'] == 0].apply(assign_priority, axis=1)
)

print("\nDistribusi Prioritas (Non-Spam):")
print(df[df['is_spam'] == 0]['prioritas'].value_counts())

# =========================
# 6. TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2)
)

# =========================
# 7. KLASIFIKASI SPAM (ML)
# =========================
print("\n=== KLASIFIKASI SPAM ===")

X_spam = vectorizer.fit_transform(df['teks_gabungan'])
y_spam = df['is_spam']

Xsp_train, Xsp_test, ysp_train, ysp_test = train_test_split(
    X_spam, y_spam,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_spam
)

# Decision Tree - Spam
dt_spam = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_spam.fit(Xsp_train, ysp_train)

ysp_pred_dt = dt_spam.predict(Xsp_test)
print("\nDecision Tree - Spam")
print("Akurasi:", accuracy_score(ysp_test, ysp_pred_dt))
print(classification_report(ysp_test, ysp_pred_dt))

# SVM - Spam
svm_spam = SVC(kernel='linear', random_state=RANDOM_STATE)
svm_spam.fit(Xsp_train, ysp_train)

ysp_pred_svm = svm_spam.predict(Xsp_test)
print("\nSVM - Spam")
print("Akurasi:", accuracy_score(ysp_test, ysp_pred_svm))
print(classification_report(ysp_test, ysp_pred_svm))

# =========================
# 8. KLASIFIKASI PRIORITAS (NON-SPAM)
# =========================
print("\n=== KLASIFIKASI PRIORITAS (NON-SPAM) ===")

df_nonspam = df[df['is_spam'] == 0]

X_prio = vectorizer.fit_transform(df_nonspam['teks_gabungan'])
y_prio = df_nonspam['prioritas']

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_prio, y_prio,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_prio
)

# Decision Tree - Prioritas
dt_prio = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_prio.fit(Xp_train, yp_train)

yp_pred_dt = dt_prio.predict(Xp_test)
print("\nDecision Tree - Prioritas")
print("Akurasi:", accuracy_score(yp_test, yp_pred_dt))
print(classification_report(yp_test, yp_pred_dt))

# SVM - Prioritas
svm_prio = SVC(kernel='linear', random_state=RANDOM_STATE)
svm_prio.fit(Xp_train, yp_train)

yp_pred_svm = svm_prio.predict(Xp_test)
print("\nSVM - Prioritas")
print("Akurasi:", accuracy_score(yp_test, yp_pred_svm))
print(classification_report(yp_test, yp_pred_svm))

# =========================
# 9. SIMPAN DATASET FINAL
# =========================
OUTPUT_FILE = "output/dataset_keluhan_spam_prioritas_labeled.xlsx"
df.to_excel(OUTPUT_FILE, index=False)

print(f"\n✅ Dataset berlabel berhasil disimpan: {OUTPUT_FILE}")

# --- Data dan prediksi sudah ada ---
# ysp_test, ysp_pred_dt, ysp_pred_svm

print("===== EVALUASI SPAM - DECISION TREE =====")
accuracy = accuracy_score(ysp_test, ysp_pred_dt)
precision = precision_score(ysp_test, ysp_pred_dt)
recall = recall_score(ysp_test, ysp_pred_dt)
f1 = f1_score(ysp_test, ysp_pred_dt)

print(f"Akurasi   : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(ysp_test, ysp_pred_dt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Spam (Decision Tree)")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

# --- SVM ---
print("===== EVALUASI SPAM - SVM =====")
accuracy = accuracy_score(ysp_test, ysp_pred_svm)
precision = precision_score(ysp_test, ysp_pred_svm)
recall = recall_score(ysp_test, ysp_pred_svm)
f1 = f1_score(ysp_test, ysp_pred_svm)

print(f"Akurasi   : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

cm = confusion_matrix(ysp_test, ysp_pred_svm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Spam (SVM)")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()
print("===== EVALUASI PRIORITAS - DECISION TREE =====")

# --- Data dan prediksi sudah ada ---
# yp_test, yp_pred_dt, yp_pred_svm

labels = ['Rendah', 'Sedang', 'Tinggi']  # urutan kelas

print("===== EVALUASI PRIORITAS - DECISION TREE =====")
accuracy = accuracy_score(yp_test, yp_pred_dt)
precision = precision_score(yp_test, yp_pred_dt, average='macro')
recall = recall_score(yp_test, yp_pred_dt, average='macro')
f1 = f1_score(yp_test, yp_pred_dt, average='macro')

print(f"Akurasi   : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(yp_test, yp_pred_dt, labels=labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Prioritas (Decision Tree)")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

# --- SVM ---
print("===== EVALUASI PRIORITAS - SVM =====")
accuracy = accuracy_score(yp_test, yp_pred_svm)
precision = precision_score(yp_test, yp_pred_svm, average='macro')
recall = recall_score(yp_test, yp_pred_svm, average='macro')
f1 = f1_score(yp_test, yp_pred_svm, average='macro')

print(f"Akurasi   : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

cm = confusion_matrix(yp_test, yp_pred_svm, labels=labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Prioritas (SVM)")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()
print("===== EVALUASI PRIORITAS - SVM =====")