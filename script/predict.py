from flask import Flask, request, jsonify
import re
import joblib
import os

# ===============================
# LOAD MODEL
# ===============================
BASE_DIR = "models"

tfidf_spam = joblib.load(os.path.join(BASE_DIR, "tfidf_spam.pkl"))
spam_model = joblib.load(os.path.join(BASE_DIR, "model_svm_spam.pkl"))

tfidf_priority = joblib.load(os.path.join(BASE_DIR, "tfidf_prio.pkl"))
priority_model = joblib.load(os.path.join(BASE_DIR, "model_dt_priority.pkl"))

# ===============================
# CLEANING FUNCTION
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# VALID COMPLAINT KEYWORDS
# (HARD OVERRIDE → NON-SPAM)
# ===============================
VALID_COMPLAINT_KEYWORDS = [
    "jalan", "berlubang", "rusak", "retak", "ambles",
    "lampu", "pju", "mati",
    "banjir", "genangan", "drainase", "selokan",
    "sampah", "bau", "liar",
    "trotoar", "jembatan", "aspal",
    "sekolah", "smp", "sd", "puskesmas",
    "marka", "rambu",
    "longsor", "irigasi", "saluran",
    "fasilitas", "umum", "infrastruktur"
]

def rule_based_valid_complaint(text):
    for k in VALID_COMPLAINT_KEYWORDS:
        if k in text:
            return True
    return False

# ---------------------------
# Spam detection helper
# ---------------------------
promo_keywords = [
    # Promosi & Diskon
    "promo", "promosi", "diskon", "murah", "sale", "obral", "banting harga",
    "berhadiah", "gratis", "hadiah", "potongan harga", "terbatas", 
    "promo besar", "pemenang", "voucher terbatas", "klaim", "hanya hari ini",
    "beli 1 gratis 1", "flash sale", "hanya untuk anda", "undian", "selamat",

    # Keuangan, Pinjol, & Investasi
    "investasi", "profit", "passive income", "cepat kaya", "binary option",
    "robot trading", "binomo", "deposit", "withdraw", "trading", "pinjol",
    "dana darurat", "bunga 0", "cuma ktp", "cepat cair", "legalitas", "terdaftar",
    "pinjam", "kredit", "cicilan", "kta", "tanpa jaminan", "bunga rendah",
    "cair cepat", "pengajuan mudah", "limit besar", "transfer", "rekening",

    # Judi & Game Berisiko
    "slot", "judi", "taruhan", "betting", "jackpot", "rtp", "spin gratis",
    "maxwin", "kasino", "poker online", "poker", "togel", "pasang nomor", 
    "situs gacor", "daftar sekarang", "jaminan menang", "depo", "wd",

    # Kontak, Urgensi, & Penipuan
    "hubungi", "kontak", "wa ", "wa.me", "dm", "telegram", "t.me",
    "chat saya", "klik link", "kunjungi website", "balas yes", "segera hubungi",
    "penting", "resmi", "akun anda", "verifikasi", "aktivasi", "kode otp",
    "segera", "darurat", "cek di sini", "website resmi", "pemberitahuan", 
    "for your attention", "dear customer",

    # Dewasa, Kesehatan, & Jasa Ilegal
    "obat kuat", "pemutih", "pembesar", "pelangsing", "obat herbal",
    "tanpa efek samping", "pria dewasa", "rahasia", "privasi", "obat jerawat",
    "pembesar alami", "sex", "massage", "open bo", "bo murah", "pijat plus", 
    "chat dewasa", "cewek panggilan", "followers", "jasa followers", "auto like", 
    "endorse murah", "naik followers", "viewer", "followers pasif", "jasa pembuatan"
]

test_keywords = [
    # Testing & Acak
    "test", "coba", "asdf", "qwerty", "123", "cek", "testing", "uji coba", 
    "test 1", "testing 123", "coba input", "zxcv", "qwert", "abc", "def", 
    "aaa", "bbb"
]

INDO_STOPWORDS = [
    "yang","untuk","dengan","dan","di","ke","dari","ini","itu","atau","pada","karena",
    "ada","adalah","saya","kami","kita","mereka","dia","akan","jika","tidak","jadi",
    "sebagai","oleh","dalam","bagi","pada","agar","pun","juga","bahwa","atau","kamu"
]


def detect_spam_rule(text: str) -> int:
    if not isinstance(text, str):
        return 0
    t = text.lower().strip()
    # very short/meaningless (<=3 words and very short length)
    words = t.split()
    if len(words) <= 3:
        if re.fullmatch(r'[a-zA-Z]{1,5}', t) or len(t) <= 6:
            return 1
    # testing keywords
    for k in test_keywords:
        if k in t:
            return 1
    # promotional keywords
    for k in promo_keywords:
        if k in t:
            return 1
    # phone pattern (indication of contact/advertisement)
    if re.search(r'0?8\d{6,12}', t):
        return 1
    # repeated characters or repeated words
    if re.fullmatch(r'(.)\1{4,}', t):
        return 1
    words = t.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return 1
    return 0

# ===============================
# MAIN PREDICTION FUNCTION
# ===============================
def predict(text):
    original_text = text
    text = clean_text(text)

    # ===============================
    # 1️⃣ HARD RULE: VALID COMPLAINT
    # ===============================
    if rule_based_valid_complaint(text):
        X_prio = tfidf_priority.transform([text])
        priority_pred = priority_model.predict(X_prio)[0]

        return {
            "text": original_text,
            "is_spam": 0,
            "spam_confidence": 0.0,
            "kategori": "Infrastruktur",
            "prioritas": priority_pred
        }

    # ===============================
    # 2️⃣ RULE-BASED SPAM
    # ===============================
    if detect_spam_rule(text) == 1:
        return {
            "text": original_text,
            "is_spam": 1,
            "spam_confidence": 1.0,
            "kategori": None,
            "prioritas": "Spam"
        }

    # ===============================
    # 3️⃣ MACHINE LEARNING (SVM)
    # ===============================
    X_spam = tfidf_spam.transform([text])
    spam_proba = float(spam_model.predict_proba(X_spam)[0][1])

    if spam_proba > 0.5:
        return {
            "text": original_text,
            "is_spam": 1,
            "spam_confidence": spam_proba,
            "kategori": None,
            "prioritas": "Spam"
        }

    # ===============================
    # 4️⃣ PRIORITY CLASSIFICATION
    # ===============================
    X_prio = tfidf_priority.transform([text])
    priority_pred = priority_model.predict(X_prio)[0]

    return {
        "text": original_text,
        "is_spam": 0,
        "spam_confidence": spam_proba,
        "kategori": None,
        "prioritas": priority_pred
    }


# ===============================
# FLASK API
# ===============================
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "text is required"}), 400

    return jsonify(predict(text))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
