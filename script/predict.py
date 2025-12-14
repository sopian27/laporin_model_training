from flask import Flask, request, jsonify
import json
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

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
# RULE-BASED SPAM KEYWORDS
# ===============================
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
    "aaa", "bbb",

    # Sapaan & Respon Sederhana
    "halo", "hai", "oke", "ok", "??", "???", "!!!", "....", "..", 
    "assalamualaikum", "wassalamualaikum", "met siang", "met malam", 
    "terima kasih", "makasih", "sama-sama", "thanks", "siap", "ya", 
    "gak", "sip", "mantap", "good", "iya", "tdk", "enggak"
]

nonsense_patterns = [
    r"^[a-z]{1,3}\s*[0-9]{0,3}$",
    r"^[0-9]{1,4}$",
    r"^[a-zA-Z]{1,4}$"
]

# ===============================
# RULE-BASED SPAM DETECTOR
# ===============================
def rule_based_spam(text):
    t = text.lower()
    if any(kw in t for kw in promo_keywords):
        return True
    if len(t.split()) <= 2 and any(kw in t for kw in test_keywords):
        return True
    if len(t) <= 3:
        return True
    for pattern in nonsense_patterns:
        if re.fullmatch(pattern, t):
            return True
    return False

# ===============================
# MAIN PREDICTION FUNCTION
# ===============================
def predict(text):
    original = text
    text = clean_text(text)

    if rule_based_spam(text):
        return {
            "text": original,
            "is_spam": 1,
            "spam_confidence": 1.0,
            "kategori": None,
            "prioritas": "Spam"
        }

    X_spam = tfidf_spam.transform([text])
    spam_proba = float(spam_model.predict_proba(X_spam)[0][1])
    is_spam = 1 if spam_proba > 0.50 else 0

    if is_spam:
        return {
            "text": original,
            "is_spam": 1,
            "spam_confidence": spam_proba,
            "kategori": None,
            "prioritas": "Spam"
        }

    X_prio = tfidf_priority.transform([text])
    priority_pred = priority_model.predict(X_prio)[0]

    return {
        "text": original,
        "is_spam": 0,
        "spam_confidence": spam_proba,
        "kategori": None,
        "prioritas": priority_pred
    }

# ===============================
# FLASK API
# ===============================
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "text is required"}), 400

    result = predict(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
