#!/usr/bin/env python3
"""
train.py
Train pipeline for Laporin project:
- Rule-based expanded spam detection to create initial labels
- Train SVM for spam detection (text)
- Train Decision Tree for priority classification (High/Medium/Low)
- Train Decision Tree for category classification (if 'Kategori' column exists)
- Export models and TF-IDF vectorizers as .pkl files

Usage:
    python3 train.py --input dataset_keluhan.csv --outdir models/
"""

import argparse
import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

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

# ---------------------------
# Priority helper (rule-based initial labels)
# ---------------------------
priority_high = ["kebakaran", "bencana", "infrastruktur", "longsor", "bahaya", "kriminal", "pencurian", "odgj", "darurat", "kecelakaan"]
priority_low = ["saran", "informasi", "apresiasi", "dukung", "reklame", "multitopik"]

def detect_priority_rule(text: str, topik: str=None) -> str:
    if not isinstance(text, str):
        return "Sedang"
    t = text.lower()
    # Check topik first (if provided)
    if isinstance(topik, str):
        topik_l = topik.lower()
        for k in priority_high:
            if k in topik_l:
                return "Tinggi"
        for k in priority_low:
            if k in topik_l:
                return "Rendah"
    # Check text keywords
    for k in priority_high:
        if k in t:
            return "Tinggi"
    for k in priority_low:
        if k in t:
            return "Rendah"
    return "Sedang"

# ---------------------------
# Main training pipeline
# ---------------------------
def main(args):
    # load
    #df = pd.read_excel(args.input)
    df = pd.read_csv(args.input, encoding='utf-8-sig')
    if 'Keluhan' not in df.columns:
        raise ValueError("Excel must contain 'Keluhan' column")
    df['Keluhan'] = df['Keluhan'].astype(str)
    df['Topik Keluhan'] = df['Topik Keluhan'].astype(str) if 'Topik Keluhan' in df.columns else ""

    # 1. create rule-based spam and priority labels
    print("Applying rule-based spam detection and priority rules...")
    df['is_spam_rule'] = df['Keluhan'].apply(detect_spam_rule)
    df['prioritas_rule'] = df.apply(lambda r: detect_priority_rule(r['Keluhan'], r.get('Topik Keluhan','')), axis=1)

    # 2. Prepare training data for spam (use both rule labels and any existing label if present)
    # If dataset already contains 'is_spam' use that as ground truth where present.
    if 'is_spam' in df.columns:
        df['is_spam_train'] = df['is_spam'].fillna(df['is_spam_rule'])
    else:
        df['is_spam_train'] = df['is_spam_rule']

    # 3. Vectorizer and SVM for spam detection
    print("Training TF-IDF vectorizer for spam model...")
    
    tfidf_spam = TfidfVectorizer(stop_words=INDO_STOPWORDS)
   # tfidf_spam = TfidfVectorizer(stop_words='indonesian', max_features=8000)
    X_spam = tfidf_spam.fit_transform(df['Keluhan'])
    y_spam = df['is_spam_train']

    # Split for evaluation
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_spam, y_spam, test_size=0.2, random_state=42, stratify=y_spam if y_spam.nunique()>1 else None)

    print("Training SVM for spam detection...")
    svm = SVC(kernel='linear', probability=True)
    svm.fit(Xs_train, ys_train)
    ys_pred = svm.predict(Xs_test)
    print("Spam detection results:")
    try:
        print(classification_report(ys_test, ys_pred))
    except Exception as e:
        print("Could not create classification report:", e)
    print("Accuracy:", accuracy_score(ys_test, ys_pred))

    # 4. Prepare training data for priority (exclude spam rows)
    df_nonspam = df[df['is_spam_train'] == 0].copy()
    # If dataset has existing 'prioritas' column use it; else use prioritas_rule for training labels.
    if 'prioritas' in df_nonspam.columns:
        y_priority = df_nonspam['prioritas'].fillna(df_nonspam['prioritas_rule'])
    else:
        y_priority = df_nonspam['prioritas_rule']

    # vectorize for priority model (can reuse same tfidf or create new)
    print("Training TF-IDF vectorizer for priority model...")
    tfidf_prio = TfidfVectorizer(stop_words=INDO_STOPWORDS)
    X_prio = tfidf_prio.fit_transform(df_nonspam['Keluhan'])

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_prio, y_priority, test_size=0.2, random_state=42, stratify=y_priority if y_priority.nunique()>1 else None)

    print("Training Decision Tree for priority classification...")
    dt_prio = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt_prio.fit(Xp_train, yp_train)
    yp_pred = dt_prio.predict(Xp_test)
    print("Priority classification results:")
    try:
        print(classification_report(yp_test, yp_pred))
    except Exception as e:
        print("Could not create classification report:", e)
    print("Accuracy:", accuracy_score(yp_test, yp_pred))

    # 5. Optional: Train category classifier if dataset has 'Kategori' or 'Topik Keluhan' labeled more granularly
    category_model = None
    tfidf_cat = None
    if 'Kategori' in df_nonspam.columns:
        print("Training category classifier from 'Kategori' column...")
        df_cat = df_nonspam.copy()
        df_cat['Kategori'] = df_cat['Kategori'].astype(str)
        tfidf_cat = TfidfVectorizer(stop_words=INDO_STOPWORDS)
        X_cat = tfidf_cat.fit_transform(df_cat['Keluhan'])
        y_cat = df_cat['Kategori']
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42, stratify=y_cat if y_cat.nunique()>1 else None)
        dt_cat = DecisionTreeClassifier(max_depth=16, random_state=42)
        dt_cat.fit(Xc_train, yc_train)
        yc_pred = dt_cat.predict(Xc_test)
        print("Category classification results:")
        try:
            print(classification_report(yc_test, yc_pred))
        except Exception as e:
            print("Could not create classification report for category:", e)
        print("Accuracy:", accuracy_score(yc_test, yc_pred))
        category_model = dt_cat

    # 6. Save models and vectorizers
    os.makedirs(args.outdir, exist_ok=True)
    print("Saving models to", args.outdir)
    joblib.dump(svm, os.path.join(args.outdir, "model_svm_spam.pkl"))
    joblib.dump(tfidf_spam, os.path.join(args.outdir, "tfidf_spam.pkl"))
    joblib.dump(dt_prio, os.path.join(args.outdir, "model_dt_priority.pkl"))
    joblib.dump(tfidf_prio, os.path.join(args.outdir, "tfidf_prio.pkl"))
    if category_model is not None:
        joblib.dump(category_model, os.path.join(args.outdir, "model_dt_category.pkl"))
        joblib.dump(tfidf_cat, os.path.join(args.outdir, "tfidf_cat.pkl"))

    # 7. Create labeled CSV with final predictions using trained models
    print("Creating labeled CSV using trained models...")
    df['is_spam_pred'] = svm.predict(tfidf_spam.transform(df['Keluhan']))
    df['prioritas_pred'] = "Spam"
    mask_nonspam = df['is_spam_pred'] == 0
    df.loc[mask_nonspam, 'prioritas_pred'] = dt_prio.predict(tfidf_prio.transform(df.loc[mask_nonspam, 'Keluhan']))
    if category_model is not None:
        df['kategori_pred'] = ""
        df.loc[mask_nonspam, 'kategori_pred'] = category_model.predict(tfidf_cat.transform(df.loc[mask_nonspam, 'Keluhan']))
    df.to_csv(os.path.join(args.outdir, "dataset_labeled.csv"), index=False)
    print("Saved labeled dataset to", os.path.join(args.outdir, "dataset_labeled.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV dataset path")
    parser.add_argument("--outdir", required=False, default="models", help="Output directory for models")
    args = parser.parse_args()
    main(args)
