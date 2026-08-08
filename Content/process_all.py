"""
Full pipeline: rename folders + create train/test splits for all domains.
Run from: C:/Users/Abdullah/Desktop/Code/Research Datasets
"""

import os
import sys
import glob
import shutil
import traceback
import pandas as pd
from sklearn.model_selection import train_test_split

# Force UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = r"C:\Users\Abdullah\Desktop\Code\Research Datasets"
SUMMARY_LINES = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    print(msg)
    SUMMARY_LINES.append(msg)

def read_csv_safe(path, **kwargs):
    """Try utf-8 first, then utf-8-sig, then latin-1."""
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")

def read_excel_safe(path, **kwargs):
    return pd.read_excel(path, **kwargs)

def save_split(df, domain_folder, text_col, label_col, rename_map=None,
               already_split=False, train_df=None, test_df=None):
    """
    Standardise columns to 'text' / 'label', drop nulls, do 80/20 split if needed,
    and save train.csv + test.csv inside domain_folder.
    rename_map: {old_label_value: new_label_value} — applied ONLY when explicitly instructed.
    """
    if already_split:
        tr, te = train_df.copy(), test_df.copy()
    else:
        tr = te = None
        full = df.copy()

    def standardise(frame):
        frame = frame[[text_col, label_col]].copy()
        frame.columns = ["text", "label"]
        frame = frame.dropna(subset=["text", "label"])
        frame["text"] = frame["text"].astype(str).str.strip()
        frame["label"] = frame["label"].astype(str).str.strip()
        if rename_map:
            frame["label"] = frame["label"].map(rename_map).fillna(frame["label"])
        return frame

    if already_split:
        tr = standardise(tr)
        te = standardise(te)
    else:
        full = standardise(full)
        if len(full) < 2:
            log(f"  SKIP {domain_folder}: too few rows ({len(full)})")
            return
        # check class sizes for stratify
        min_class = full["label"].value_counts().min()
        stratify = full["label"] if min_class >= 2 else None
        if stratify is None:
            log(f"  WARNING: cannot stratify (some class has <2 samples), splitting without stratify")
        tr, te = train_test_split(full, test_size=0.2, random_state=42, stratify=stratify)

    out = domain_folder
    os.makedirs(out, exist_ok=True)
    tr.to_csv(os.path.join(out, "train.csv"), index=False, encoding="utf-8-sig")
    te.to_csv(os.path.join(out, "test.csv"),  index=False, encoding="utf-8-sig")

    log(f"  Saved → {out}")
    log(f"    Total: {len(tr)+len(te)}, Train: {len(tr)}, Test: {len(te)}")
    log(f"    Train label dist: {dict(tr['label'].value_counts())}")
    log(f"    Test  label dist: {dict(te['label'].value_counts())}")

# ---------------------------------------------------------------------------
# PART 1 — FOLDER RENAMING
# ---------------------------------------------------------------------------
log("=" * 70)
log("PART 1: FOLDER RENAMING")
log("=" * 70)

rename_ops = [
    # Nastaliq / Sentiment Analysis
    (r"Nastaliq\Sentiment Analysis\data",   r"Nastaliq\Sentiment Analysis\Domain_B_Movie_Reviews"),
    (r"Nastaliq\Sentiment Analysis\data2",  r"Nastaliq\Sentiment Analysis\Domain_D_Governance_Political"),
    (r"Nastaliq\Sentiment Analysis\data3",  r"Nastaliq\Sentiment Analysis\Domain_C_Civic_Social_Topics"),
    (r"Nastaliq\Sentiment Analysis\data4",  r"Nastaliq\Sentiment Analysis\Domain_E_Socioeconomic_Agricultural"),
    (r"Nastaliq\Sentiment Analysis\data5",  r"Nastaliq\Sentiment Analysis\Domain_A_Twitter_Social_Media"),
    # Nastaliq / Hate Speech
    (r"Nastaliq\Hate Speech\data1",  r"Nastaliq\Hate Speech\Domain_A_Social_Media_Offensive"),
    (r"Nastaliq\Hate Speech\data2",  r"Nastaliq\Hate Speech\Domain_B_Social_Media_Hate_Speech"),
    (r"Nastaliq\Hate Speech\data3",  r"Nastaliq\Hate Speech\Domain_C_Politics_Sports_Health_Family"),
    (r"Nastaliq\Hate Speech\data4",  r"Nastaliq\Hate Speech\Domain_D_Sports_Labor_Arts_Education"),
    (r"Nastaliq\Hate Speech\data5",  r"Nastaliq\Hate Speech\Domain_E_Inter_Faith_Sectarian_Ethnic"),
    # Nastaliq / Fake News Detection
    (r"Nastaliq\Fake News Detection\data1", r"Nastaliq\Fake News Detection\Domain_A_Multi_Topic_News"),
    (r"Nastaliq\Fake News Detection\data2", r"Nastaliq\Fake News Detection\Domain_B_Ax_to_Grind"),
    (r"Nastaliq\Fake News Detection\data3", r"Nastaliq\Fake News Detection\Domain_C_General_Pakistani_News"),
    (r"Nastaliq\Fake News Detection\data4", r"Nastaliq\Fake News Detection\Domain_D_Fact_Checking_Platform"),
    # Nastaliq / QA
    (r"Nastaliq\QA\data1", r"Nastaliq\QA\Domain_A_Religious_Hadith"),
    (r"Nastaliq\QA\data2", r"Nastaliq\QA\Domain_B_General_Knowledge"),
    (r"Nastaliq\QA\data3", r"Nastaliq\QA\Domain_C_Long_Form_QnA"),
    (r"Nastaliq\QA\data4", r"Nastaliq\QA\Domain_D_Short_Form_QnA"),
    # Roman / Product or eCommerce Reviews
    (r"Roman\Product or eCommerce Reviews\data1", r"Roman\Product or eCommerce Reviews\Domain_A_Daraz_Ecommerce"),
    (r"Roman\Product or eCommerce Reviews\data2", r"Roman\Product or eCommerce Reviews\Domain_B_Restaurant_Food"),
    (r"Roman\Product or eCommerce Reviews\data3", r"Roman\Product or eCommerce Reviews\Domain_C_General_Product_Service"),
    (r"Roman\Product or eCommerce Reviews\data4", r"Roman\Product or eCommerce Reviews\Domain_D_Electronics_Gaming_Delivery"),
    (r"Roman\Product or eCommerce Reviews\data5", r"Roman\Product or eCommerce Reviews\Domain_E_Mixed_Spelling_Variation"),
    # Roman / Sentiment Analysis
    (r"Roman\Sentiment Analysis\data1", r"Roman\Sentiment Analysis\Domain_A_Mixed_Public_Discourse"),
    (r"Roman\Sentiment Analysis\data2", r"Roman\Sentiment Analysis\Domain_B_Social_Media_Politics_Drama"),
    (r"Roman\Sentiment Analysis\data3", r"Roman\Sentiment Analysis\Domain_C_YouTube_Entertainment"),
    (r"Roman\Sentiment Analysis\data4", r"Roman\Sentiment Analysis\Domain_D_Utilities_Apps_Courses"),
    # Roman / Cyber Abuse
    (r"Roman\Cyber Abuse or Abusive Language\data1", r"Roman\Cyber Abuse or Abusive Language\Domain_B_Content_Creator_Comments"),
    (r"Roman\Cyber Abuse or Abusive Language\data2", r"Roman\Cyber Abuse or Abusive Language\Domain_A_YouTube_Comments"),
    (r"Roman\Cyber Abuse or Abusive Language\data3", r"Roman\Cyber Abuse or Abusive Language\Domain_C_Social_Media_Diverse_Vocab"),
    (r"Roman\Cyber Abuse or Abusive Language\data4", r"Roman\Cyber Abuse or Abusive Language\Domain_D_General_Online_Video"),
    # Roman / Twitter
    (r"Roman\Twitter or Social Media Opinions\data1", r"Roman\Twitter or Social Media Opinions\Domain_A_Twitter_Political_Social"),
    (r"Roman\Twitter or Social Media Opinions\data2", r"Roman\Twitter or Social Media Opinions\Domain_B_Consumer_Electronics_Gaming"),
    (r"Roman\Twitter or Social Media Opinions\data3", r"Roman\Twitter or Social Media Opinions\Domain_C_Public_Civic_Services"),
    (r"Roman\Twitter or Social Media Opinions\data4", r"Roman\Twitter or Social Media Opinions\Domain_D_Mixed_Social_Commentary"),
]

for src_rel, dst_rel in rename_ops:
    src = os.path.join(BASE, src_rel)
    dst = os.path.join(BASE, dst_rel)
    if os.path.exists(src):
        if os.path.exists(dst):
            log(f"  SKIP rename (dest exists): {dst_rel}")
        else:
            os.rename(src, dst)
            log(f"  RENAMED: {src_rel} -> {dst_rel}")
    else:
        log(f"  NOT FOUND (already renamed?): {src_rel}")

# ---------------------------------------------------------------------------
# PART 2 — TRAIN / TEST SPLITS
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("PART 2: TRAIN/TEST SPLITS")
log("=" * 70)

# ── Helper: path builder ────────────────────────────────────────────────────
def D(*parts):
    return os.path.join(BASE, *parts)

# ===========================================================================
# NASTALIQ / SENTIMENT ANALYSIS
# ===========================================================================

# --- Domain_B_Movie_Reviews (already split, columns already text/label) ---
log("\n[Nastaliq/SA] Domain_B_Movie_Reviews")
try:
    folder = D("Nastaliq", "Sentiment Analysis", "Domain_B_Movie_Reviews")
    tr = read_csv_safe(os.path.join(folder, "train.csv"))
    te = read_csv_safe(os.path.join(folder, "test.csv"))
    save_split(None, folder, "text", "label", already_split=True,
               train_df=tr, test_df=te)
except Exception as e:
    log(f"  ERROR: {e}")

# --- Domain_D_Governance_Political (already split) ---
log("\n[Nastaliq/SA] Domain_D_Governance_Political")
try:
    folder = D("Nastaliq", "Sentiment Analysis", "Domain_D_Governance_Political")
    tr = read_csv_safe(os.path.join(folder, "train.csv"))
    te = read_csv_safe(os.path.join(folder, "test.csv"))
    save_split(None, folder, "text", "label", already_split=True,
               train_df=tr, test_df=te)
except Exception as e:
    log(f"  ERROR: {e}")

# --- Domain_C_Civic_Social_Topics (urdu_sentiment_multidomain.csv) ---
log("\n[Nastaliq/SA] Domain_C_Civic_Social_Topics")
try:
    folder = D("Nastaliq", "Sentiment Analysis", "Domain_C_Civic_Social_Topics")
    src    = os.path.join(folder, "urdu_sentiment_multidomain.csv")
    df = read_csv_safe(src)
    # cols: Tweet, Class
    df = df.rename(columns={"Tweet": "text", "Class": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_E_Socioeconomic_Agricultural (urdu_sentiments.csv) ---
log("\n[Nastaliq/SA] Domain_E_Socioeconomic_Agricultural")
try:
    folder = D("Nastaliq", "Sentiment Analysis", "Domain_E_Socioeconomic_Agricultural")
    src    = os.path.join(folder, "urdu_sentiments.csv")
    df = read_csv_safe(src)
    # col may have BOM: strip it
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]
    # cols: Tweet, Class
    df = df.rename(columns={"Tweet": "text", "Class": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_A_Twitter_Social_Media (urdu-sentiment-corpus-v1.tsv) ---
log("\n[Nastaliq/SA] Domain_A_Twitter_Social_Media")
try:
    folder = D("Nastaliq", "Sentiment Analysis", "Domain_A_Twitter_Social_Media")
    src    = os.path.join(folder, "urdu-sentiment-corpus-v1.tsv")
    df = read_csv_safe(src, sep="\t")
    # cols: Tweet, Class
    df = df.rename(columns={"Tweet": "text", "Class": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# ===========================================================================
# NASTALIQ / HATE SPEECH
# ===========================================================================

# --- Domain_A_Social_Media_Offensive ---
log("\n[Nastaliq/HS] Domain_A_Social_Media_Offensive")
try:
    folder = D("Nastaliq", "Hate Speech", "Domain_A_Social_Media_Offensive")
    src    = os.path.join(folder, "22K_Offensive_dataset_Final.xlsx")
    df = read_excel_safe(src)
    # cols: New_Tweet_ID, Tweet_ID, User_ID, Tweet_Text, Human_Benchmark_Offensive, Human_Benchmark_Offensive_Type
    df = df.rename(columns={"Tweet_Text": "text", "Human_Benchmark_Offensive": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_B_Social_Media_Hate_Speech ---
log("\n[Nastaliq/HS] Domain_B_Social_Media_Hate_Speech")
try:
    folder = D("Nastaliq", "Hate Speech", "Domain_B_Social_Media_Hate_Speech")
    src    = os.path.join(folder, "Urdu_Hate_Speech.xlsx")
    df = read_excel_safe(src)
    # cols: Tag, CombinedTweet
    df = df.rename(columns={"CombinedTweet": "text", "Tag": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_C_Politics_Sports_Health_Family ---
log("\n[Nastaliq/HS] Domain_C_Politics_Sports_Health_Family")
try:
    folder = D("Nastaliq", "Hate Speech", "Domain_C_Politics_Sports_Health_Family")
    src    = os.path.join(folder, "urdu_hate_speech_multi_domain.csv")
    df = read_csv_safe(src)
    # cols: Tweet, Class
    df = df.rename(columns={"Tweet": "text", "Class": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_D_Sports_Labor_Arts_Education ---
log("\n[Nastaliq/HS] Domain_D_Sports_Labor_Arts_Education")
try:
    folder = D("Nastaliq", "Hate Speech", "Domain_D_Sports_Labor_Arts_Education")
    src    = os.path.join(folder, "urdu_hate_speech_multidomain.csv")
    df = read_csv_safe(src)
    # cols: Tweet, Class
    df = df.rename(columns={"Tweet": "text", "Class": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_E_Inter_Faith_Sectarian_Ethnic ---
log("\n[Nastaliq/HS] Domain_E_Inter_Faith_Sectarian_Ethnic")
try:
    folder = D("Nastaliq", "Hate Speech", "Domain_E_Inter_Faith_Sectarian_Ethnic")
    src    = os.path.join(folder, "ISE_Level_1_Dataset.xlsx")
    df = read_excel_safe(src)
    # cols: Tweet_ID, Tweet_Text, Hammad, Khurram, Final_decision
    df = df.rename(columns={"Tweet_Text": "text", "Final_decision": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# ===========================================================================
# NASTALIQ / FAKE NEWS DETECTION
# ===========================================================================

# --- Domain_A_Multi_Topic_News (txt files in Train/Fake, Train/Real, Test/Fake, Test/Real) ---
log("\n[Nastaliq/FN] Domain_A_Multi_Topic_News")
try:
    folder = D("Nastaliq", "Fake News Detection", "Domain_A_Multi_Topic_News")
    records_train, records_test = [], []
    for split_name, records in [("Train", records_train), ("Test", records_test)]:
        for label_name, label_val in [("Fake", 0), ("Real", 1)]:
            txt_dir = os.path.join(folder, split_name, label_name)
            if os.path.isdir(txt_dir):
                for fname in os.listdir(txt_dir):
                    if fname.endswith(".txt"):
                        fpath = os.path.join(txt_dir, fname)
                        try:
                            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                                content = f.read().strip()
                        except Exception:
                            content = ""
                        records.append({"text": content, "label": label_val})
    tr_df = pd.DataFrame(records_train)
    te_df = pd.DataFrame(records_test)
    tr_df = tr_df.dropna(subset=["text", "label"])
    te_df = te_df.dropna(subset=["text", "label"])
    tr_df.to_csv(os.path.join(folder, "train.csv"), index=False, encoding="utf-8-sig")
    te_df.to_csv(os.path.join(folder, "test.csv"),  index=False, encoding="utf-8-sig")
    log(f"  Saved → {folder}")
    log(f"    Train: {len(tr_df)}, Test: {len(te_df)}")
    log(f"    Train label dist: {dict(tr_df['label'].value_counts())}")
    log(f"    Test  label dist: {dict(te_df['label'].value_counts())}")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_B_Ax_to_Grind (Combined .csv) ---
log("\n[Nastaliq/FN] Domain_B_Ax_to_Grind")
try:
    folder = D("Nastaliq", "Fake News Detection", "Domain_B_Ax_to_Grind")
    src    = os.path.join(folder, "Combined .csv")
    df = read_csv_safe(src)
    # cols: Sr. No., News Items, Label (FAKE/TRUE)
    df = df.rename(columns={"News Items": "text", "Label": "label"})
    # Map: FAKE -> 0, TRUE -> 1
    label_map = {"FAKE": 0, "TRUE": 1, "Fake": 0, "True": 1, "fake": 0, "true": 1}
    df["label"] = df["label"].map(label_map).fillna(df["label"])
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_C_General_Pakistani_News (Fake News 12166.xlsx + Final True News-11012.xlsx) ---
log("\n[Nastaliq/FN] Domain_C_General_Pakistani_News")
try:
    folder = D("Nastaliq", "Fake News Detection", "Domain_C_General_Pakistani_News")
    df_fake = read_excel_safe(os.path.join(folder, "Fake News 12166.xlsx"))
    df_fake.columns = [c.strip() for c in df_fake.columns]
    df_fake = df_fake.rename(columns={"News Items": "text"})
    df_fake["label"] = 0

    df_true = read_excel_safe(os.path.join(folder, "Final True News-11012.xlsx"))
    df_true.columns = [c.strip() for c in df_true.columns]
    df_true = df_true.rename(columns={"News Items": "text"})
    df_true["label"] = 1

    df = pd.concat([df_fake[["text", "label"]], df_true[["text", "label"]]], ignore_index=True)
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_D_Fact_Checking_Platform ---
log("\n[Nastaliq/FN] Domain_D_Fact_Checking_Platform")
try:
    folder = D("Nastaliq", "Fake News Detection", "Domain_D_Fact_Checking_Platform")
    src    = os.path.join(folder, "Notri-Fact_Real_Unreal_Urdu_NEWS.xlsx")
    df = read_excel_safe(src)
    log(f"  cols: {list(df.columns)}")
    # cols: Index, Headline, News_Text, Category, Date, News_length, Label
    df = df.rename(columns={"News_Text": "text", "Label": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# ===========================================================================
# NASTALIQ / QA
# ===========================================================================

for domain, filename, q_col, a_col in [
    ("Domain_A_Religious_Hadith",  "qa_ahadis.csv",                "Question", "Answer"),
    ("Domain_B_General_Knowledge", "qa_gk.csv",                    "Question", "Answer"),
    ("Domain_C_Long_Form_QnA",     "urdu_qna_long_multidomain.csv","Question", "Answer"),
    ("Domain_D_Short_Form_QnA",    "urdu_qna_multidomain.csv",     "Question", "Answer"),
]:
    log(f"\n[Nastaliq/QA] {domain}")
    try:
        folder = D("Nastaliq", "QA", domain)
        src = os.path.join(folder, filename)
        df = read_csv_safe(src)
        # Normalise column names
        col_map = {}
        for c in df.columns:
            cl = c.strip().lstrip("﻿").lower()
            if cl == "question":
                col_map[c] = "text"
            elif cl == "answer":
                col_map[c] = "label"
        df = df.rename(columns=col_map)
        save_split(df, folder, "text", "label")
    except Exception as e:
        log(f"  ERROR: {e}\n{traceback.format_exc()}")

# ===========================================================================
# ROMAN / PRODUCT OR eCOMMERCE REVIEWS
# ===========================================================================

# --- Domain_A_Daraz_Ecommerce ---
log("\n[Roman/PR] Domain_A_Daraz_Ecommerce")
try:
    folder = D("Roman", "Product or eCommerce Reviews", "Domain_A_Daraz_Ecommerce")
    src    = os.path.join(folder, "daraz-code-mixed-product-reviews.csv")
    df = read_csv_safe(src)
    # cols: Sentiments, Reviews
    df = df.rename(columns={"Reviews": "text", "Sentiments": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_B_Restaurant_Food ---
log("\n[Roman/PR] Domain_B_Restaurant_Food")
try:
    folder = D("Roman", "Product or eCommerce Reviews", "Domain_B_Restaurant_Food")
    src    = os.path.join(folder, "kababjees_Review_2025.csv")
    df = read_csv_safe(src)
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]
    # cols: review, rating, sentiment
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_C_General_Product_Service ---
log("\n[Roman/PR] Domain_C_General_Product_Service")
try:
    folder = D("Roman", "Product or eCommerce Reviews", "Domain_C_General_Product_Service")
    src    = os.path.join(folder, "Roman urdu Reviews and summary - summary 350.csv")
    df = read_csv_safe(src)
    # cols: Reviews, Summary  (summarization dataset — treat Summary as label)
    df = df.rename(columns={"Reviews": "text", "Summary": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_D_Electronics_Gaming_Delivery ---
log("\n[Roman/PR] Domain_D_Electronics_Gaming_Delivery")
try:
    folder = D("Roman", "Product or eCommerce Reviews", "Domain_D_Electronics_Gaming_Delivery")
    src    = os.path.join(folder, "Roman Urdu reviews Dataset with English translation.csv")
    df = read_csv_safe(src)
    # cols: ROMAN URDU REVIEWS, TRANSLATED IN ENGLISH , SENTIMENT
    df = df.rename(columns={"ROMAN URDU REVIEWS": "text", "SENTIMENT": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_E_Mixed_Spelling_Variation ---
log("\n[Roman/PR] Domain_E_Mixed_Spelling_Variation")
try:
    folder = D("Roman", "Product or eCommerce Reviews", "Domain_E_Mixed_Spelling_Variation")
    src    = os.path.join(folder, "Roman Urdu words and English Translation with spelling variation.csv")
    df = read_csv_safe(src)
    # cols: var-1..5, Common, English Translated
    # Use Common as text, English Translated as label (word → translation)
    df = df.rename(columns={"Common": "text", "English Translated": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# ===========================================================================
# ROMAN / SENTIMENT ANALYSIS
# ===========================================================================

# --- Domain_A_Mixed_Public_Discourse (parquet) ---
log("\n[Roman/SA] Domain_A_Mixed_Public_Discourse")
try:
    folder = D("Roman", "Sentiment Analysis", "Domain_A_Mixed_Public_Discourse")
    tr_pq   = os.path.join(folder, "train-00000-of-00001.parquet")
    val_pq  = os.path.join(folder, "validation-00000-of-00001.parquet")
    te_pq   = os.path.join(folder, "test-00000-of-00001.parquet")
    tr_df   = pd.read_parquet(tr_pq)
    val_df  = pd.read_parquet(val_pq)
    te_df   = pd.read_parquet(te_pq)
    # Merge validation into train
    tr_df   = pd.concat([tr_df, val_df], ignore_index=True)
    # cols: text, label  (already correct)
    save_split(None, folder, "text", "label", already_split=True,
               train_df=tr_df, test_df=te_df)
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_B_Social_Media_Politics_Drama (TSV, no header: col0=label, col1=text) ---
log("\n[Roman/SA] Domain_B_Social_Media_Politics_Drama")
try:
    folder = D("Roman", "Sentiment Analysis", "Domain_B_Social_Media_Politics_Drama")
    src    = os.path.join(folder, "Dataset 11000 Reviews.tsv")
    df = read_csv_safe(src, sep="\t", header=None)
    df.columns = ["label", "text"]
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_C_YouTube_Entertainment ---
log("\n[Roman/SA] Domain_C_YouTube_Entertainment")
try:
    folder = D("Roman", "Sentiment Analysis", "Domain_C_YouTube_Entertainment")
    src    = os.path.join(folder, "RomanUrdu_English_YouTube_Sentiment_27K.csv")
    df = read_csv_safe(src)
    # cols: Comment, Label
    df = df.rename(columns={"Comment": "text", "Label": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_D_Utilities_Apps_Courses (CSV, no header: col0=label, col1=text) ---
log("\n[Roman/SA] Domain_D_Utilities_Apps_Courses")
try:
    folder = D("Roman", "Sentiment Analysis", "Domain_D_Utilities_Apps_Courses")
    src    = os.path.join(folder, "RomanUrduSentiment.csv")
    df = read_csv_safe(src, header=None)
    df.columns = ["label", "text"]
    # strip leading space from label
    df["label"] = df["label"].astype(str).str.strip()
    df["text"]  = df["text"].astype(str).str.strip()
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# ===========================================================================
# ROMAN / CYBER ABUSE
# ===========================================================================

# data1 → Domain_B_Content_Creator_Comments
# data2 → Domain_A_YouTube_Comments
# data3 → Domain_C_Social_Media_Diverse_Vocab
# data4 → Domain_D_General_Online_Video

for domain, filename, t_col, l_col in [
    ("Domain_B_Content_Creator_Comments", "roman_urdu_cyber_abuse.csv",         "text",    "label"),
    ("Domain_A_YouTube_Comments",         "roman_urdu_cyber_abuse_dataset.csv",  "comment", "label"),
    ("Domain_C_Social_Media_Diverse_Vocab","roman_urdu_cyber_abuse_diverse.csv", "text",    "label"),
    ("Domain_D_General_Online_Video",     "roman_urdu_cyber_abusing.csv",        "text",    "label"),
]:
    log(f"\n[Roman/CA] {domain}")
    try:
        folder = D("Roman", "Cyber Abuse or Abusive Language", domain)
        src = os.path.join(folder, filename)
        df = read_csv_safe(src)
        if t_col != "text":
            df = df.rename(columns={t_col: "text"})
        save_split(df, folder, "text", "label")
    except Exception as e:
        log(f"  ERROR: {e}\n{traceback.format_exc()}")

# ===========================================================================
# ROMAN / TWITTER OR SOCIAL MEDIA OPINIONS
# ===========================================================================

# --- Domain_A_Twitter_Political_Social (New.csv: Sentences, Labels) ---
log("\n[Roman/TW] Domain_A_Twitter_Political_Social")
try:
    folder = D("Roman", "Twitter or Social Media Opinions", "Domain_A_Twitter_Political_Social")
    src    = os.path.join(folder, "New.csv")
    df = read_csv_safe(src)
    df = df.rename(columns={"Sentences": "text", "Labels": "label"})
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_B_Consumer_Electronics_Gaming ---
log("\n[Roman/TW] Domain_B_Consumer_Electronics_Gaming")
try:
    folder = D("Roman", "Twitter or Social Media Opinions", "Domain_B_Consumer_Electronics_Gaming")
    src    = os.path.join(folder, "roman_urdu_opinions_lifestyle.csv")
    df = read_csv_safe(src)
    # cols: text, label
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_C_Public_Civic_Services ---
log("\n[Roman/TW] Domain_C_Public_Civic_Services")
try:
    folder = D("Roman", "Twitter or Social Media Opinions", "Domain_C_Public_Civic_Services")
    src    = os.path.join(folder, "roman_urdu_opinions_public.csv")
    df = read_csv_safe(src)
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# --- Domain_D_Mixed_Social_Commentary ---
log("\n[Roman/TW] Domain_D_Mixed_Social_Commentary")
try:
    folder = D("Roman", "Twitter or Social Media Opinions", "Domain_D_Mixed_Social_Commentary")
    src    = os.path.join(folder, "roman_urdu_social_opinions.csv")
    df = read_csv_safe(src)
    save_split(df, folder, "text", "label")
except Exception as e:
    log(f"  ERROR: {e}\n{traceback.format_exc()}")

# ===========================================================================
# WRITE SUMMARY FILE
# ===========================================================================
summary_path = os.path.join(BASE, "SPLITS_SUMMARY.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(SUMMARY_LINES))
print(f"\nSummary written to: {summary_path}")
