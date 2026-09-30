"""Single source of truth for the benchmark: domains, tasks, label spaces, provenance flags.

Every other script imports from here, so a domain's name, label set or synthetic
share is defined exactly once.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    """Read KEY = value pairs from ROOT/.env (git-ignored) into os.environ without
    overriding existing variables; HUGGINGFACE_ACCESS_TOKEN is exposed as HF_TOKEN,
    the name huggingface_hub reads. Values are never printed."""
    import os
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'")
        if k and v:
            os.environ.setdefault(k, v)
    if "HUGGINGFACE_ACCESS_TOKEN" in os.environ:
        os.environ.setdefault("HF_TOKEN", os.environ["HUGGINGFACE_ACCESS_TOKEN"])


load_env()
RAW_SPLITS = ROOT                      # legacy per-domain train/test CSVs (pre-dedup)
DATA_V10 = ROOT / "data" / "v1.0"       # = public HF/Zenodo release (exact dedup only)
DATA_V11 = ROOT / "data" / "v1.1"       # corrected release used for all experiments
RUNS = ROOT / "runs"                    # raw per-cell outputs (predictions, JSON)
ANALYSIS = ROOT / "analysis"            # tables, figures, stats derived from RUNS


@dataclass(frozen=True)
class Domain:
    script: str          # "Nastaliq" | "Roman"
    collection: str      # folder name under the script, e.g. "Hate Speech"
    folder: str          # e.g. "Domain_A_Social_Media_Offensive"
    name: str            # short display name
    origin: str          # "organic" | "generated" (LLM-generated from seeds) | "uncertain"; see audit_synthetic.py
    source_file: str     # raw file the split was built from

    @property
    def key(self) -> str:
        return f"{self.script}/{self.collection}/{self.folder}"

    @property
    def letter(self) -> str:
        return self.folder.split("_")[1]

    @property
    def sid(self) -> str:
        """Short id unique within a transfer matrix, e.g. 'HS-A', 'PR-D', 'TW-B'."""
        return f"{COLLECTION_CODE[self.collection]}-{self.letter}"


COLLECTION_CODE = {
    "Sentiment Analysis": "SA", "Hate Speech": "HS", "Fake News Detection": "FN", "QA": "QA",
    "Cyber Abuse or Abusive Language": "CA", "Product or eCommerce Reviews": "PR",
    "Twitter or Social Media Opinions": "TW",
}


def _d(script, coll, folder, name, card_pct, src):
    CARD_SYNTH_PCT[f"{script}/{coll}/{folder}"] = card_pct
    return Domain(script, coll, folder, name, _origin(script, coll, folder, card_pct), src)


# Share of Grok text claimed by the v1.0 dataset card. These figures were estimates,
# not measurements; kept only to document the discrepancy with the audit.
CARD_SYNTH_PCT = {}


def _origin(script, coll, folder, card_pct):
    """Origin from the empirical audit (analysis/synthetic_audit.csv): every domain the
    card marked as partly synthetic shows the generated-text signature (template
    clusters, low lexical diversity, uniform lengths) throughout, so it is treated as
    generated in full. Roman CA-A is recorded as organic but shows augmentation noise."""
    if card_pct > 0:
        return "generated"
    if script == "Nastaliq" and folder.startswith("Domain_B_Movie_Reviews"):
        return "translated"   # machine-translated English IMDB reviews (Kaggle: akkefa)
    if script == "Roman" and folder.startswith("Domain_A_YouTube_Comments"):
        return "uncertain"
    return "organic"


# ---------------------------------------------------------------------------
# All 33 domains of the v1.0 release. `DROPPED_V11` lists the ones removed in v1.1.
# ---------------------------------------------------------------------------
DOMAINS = [
    # Nastaliq sentiment
    _d("Nastaliq", "Sentiment Analysis", "Domain_A_Twitter_Social_Media", "Twitter Social Media", 0, "urdu-sentiment-corpus-v1.tsv"),
    _d("Nastaliq", "Sentiment Analysis", "Domain_B_Movie_Reviews", "Movie Reviews", 0, "train.csv/test.csv (pre-split)"),
    _d("Nastaliq", "Sentiment Analysis", "Domain_C_Civic_Social_Topics", "Civic Social Topics", 25, "urdu_sentiment_multidomain.csv"),
    _d("Nastaliq", "Sentiment Analysis", "Domain_D_Governance_Political", "Governance Political", 0, "UrduPoliticalTweets (pre-split)"),
    _d("Nastaliq", "Sentiment Analysis", "Domain_E_Socioeconomic_Agricultural", "Socioeconomic Agricultural", 20, "urdu_sentiments.csv"),
    # Nastaliq hate speech
    _d("Nastaliq", "Hate Speech", "Domain_A_Social_Media_Offensive", "Social Media Offensive", 0, "22K_Offensive_dataset_Final.xlsx"),
    _d("Nastaliq", "Hate Speech", "Domain_B_Social_Media_Hate_Speech", "Social Media Hate Speech", 0, "Urdu_Hate_Speech.xlsx"),
    _d("Nastaliq", "Hate Speech", "Domain_C_Politics_Sports_Health_Family", "Politics Sports Health Family", 70, "urdu_hate_speech_multi_domain.csv"),
    _d("Nastaliq", "Hate Speech", "Domain_D_Sports_Labor_Arts_Education", "Sports Labor Arts Education", 70, "urdu_hate_speech_multidomain.csv"),
    _d("Nastaliq", "Hate Speech", "Domain_E_Inter_Faith_Sectarian_Ethnic", "Inter Faith Sectarian Ethnic", 0, "ISE_Level_1_Dataset.xlsx"),
    # Nastaliq fake news
    _d("Nastaliq", "Fake News Detection", "Domain_A_Multi_Topic_News", "Multi Topic News", 0, "1.Corpus/ text files"),
    _d("Nastaliq", "Fake News Detection", "Domain_B_Ax_to_Grind", "Ax to Grind", 0, "Combined .csv"),
    _d("Nastaliq", "Fake News Detection", "Domain_C_General_Pakistani_News", "General Pakistani News", 0, "Fake News 12166.xlsx + Final True News-11012.xlsx"),
    _d("Nastaliq", "Fake News Detection", "Domain_D_Fact_Checking_Platform", "Fact Checking Platform", 0, "Notri-Fact_Real_Unreal_Urdu_NEWS.xlsx"),
    # Nastaliq QA pair validation
    _d("Nastaliq", "QA", "Domain_A_Religious_Hadith", "Religious Hadith", 0, "qa_ahadis.csv"),
    _d("Nastaliq", "QA", "Domain_B_General_Knowledge", "General Knowledge", 0, "qa_gk.csv"),
    _d("Nastaliq", "QA", "Domain_C_Long_Form_QnA", "Long Form QnA", 60, "urdu_qna_long_multidomain.csv"),
    _d("Nastaliq", "QA", "Domain_D_Short_Form_QnA", "Short Form QnA", 50, "urdu_qna_multidomain.csv"),
    # Roman cyber abuse
    _d("Roman", "Cyber Abuse or Abusive Language", "Domain_A_YouTube_Comments", "YouTube Comments", 0, "roman_urdu_cyber_abuse_dataset.csv"),
    _d("Roman", "Cyber Abuse or Abusive Language", "Domain_B_Content_Creator_Comments", "Content Creator Comments", 65, "roman_urdu_cyber_abuse.csv"),
    _d("Roman", "Cyber Abuse or Abusive Language", "Domain_C_Social_Media_Diverse_Vocab", "Social Media Diverse Vocab", 65, "roman_urdu_cyber_abuse_diverse.csv"),
    _d("Roman", "Cyber Abuse or Abusive Language", "Domain_D_General_Online_Video", "General Online Video", 65, "roman_urdu_cyber_abusing.csv"),
    # Roman product reviews (3-class)
    _d("Roman", "Product or eCommerce Reviews", "Domain_A_Daraz_Ecommerce", "Daraz Ecommerce", 0, "daraz-code-mixed-product-reviews.csv"),
    _d("Roman", "Product or eCommerce Reviews", "Domain_B_Restaurant_Food", "Restaurant Food", 0, "kababjees_Review_2025.csv"),
    _d("Roman", "Product or eCommerce Reviews", "Domain_D_Electronics_Gaming_Delivery", "General Social Comments", 0, "Roman Urdu reviews Dataset with English translation.csv"),
    # Roman sentiment (binary)
    _d("Roman", "Sentiment Analysis", "Domain_A_Mixed_Public_Discourse", "Mixed Public Discourse", 0, "HF parquet (train/validation/test)"),
    _d("Roman", "Sentiment Analysis", "Domain_B_Social_Media_Politics_Drama", "Social Media Politics Drama", 0, "Dataset 11000 Reviews.tsv"),
    _d("Roman", "Sentiment Analysis", "Domain_C_YouTube_Entertainment", "YouTube Entertainment", 0, "RomanUrdu_English_YouTube_Sentiment_27K.csv"),
    _d("Roman", "Sentiment Analysis", "Domain_D_Utilities_Apps_Courses", "Utilities Apps Courses", 0, "RomanUrduSentiment.csv"),
    # Roman Twitter / social opinions (3-class)
    _d("Roman", "Twitter or Social Media Opinions", "Domain_A_Twitter_Political_Social", "Twitter Political Social", 0, "New.csv"),
    _d("Roman", "Twitter or Social Media Opinions", "Domain_B_Consumer_Electronics_Gaming", "Consumer Electronics Gaming", 70, "roman_urdu_opinions_lifestyle.csv"),
    _d("Roman", "Twitter or Social Media Opinions", "Domain_C_Public_Civic_Services", "Public Civic Services", 70, "roman_urdu_opinions_public.csv"),
    _d("Roman", "Twitter or Social Media Opinions", "Domain_D_Mixed_Social_Commentary", "Mixed Social Commentary", 65, "roman_urdu_social_opinions.csv"),
]
DOMAIN_BY_KEY = {d.key: d for d in DOMAINS}

# Domains whose text is a copy of another domain in the same collection
# (verified 2026-09-28: 100% text overlap, 98.7-100% label agreement).
DROPPED_V11 = {
    "Nastaliq/Hate Speech/Domain_E_Inter_Faith_Sectarian_Ethnic": "same tweets and labels as Hate Speech A (ISE Level-1 re-uses the 22K Offensive pool)",
    "Nastaliq/Sentiment Analysis/Domain_D_Governance_Political": "same texts and labels as Sentiment A",
    "Roman/Sentiment Analysis/Domain_D_Utilities_Apps_Courses": "same 11,000-review corpus as Roman Sentiment B",
    "Roman/Twitter or Social Media Opinions/Domain_A_Twitter_Political_Social": "positive/negative items are the RUSAD corpus (= Roman Sentiment B); all neutral items come from a different source, so the label is predictable from the source",
    # added 2026-09-30, same criterion as TW-A: fake and real items were distributed as two
    # files; 40% of fake items use fact-check wording against 2% of real items
    "Nastaliq/Fake News Detection/Domain_C_General_Pakistani_News": "fake and real items come from two separate files; fake items are largely fact-check verdicts, so the label is predictable from the source",
    # found by pipeline.audit_shortcuts: the outlet dateline "92 News" occurs in 90% of
    # real items and 3.5% of fake items ("web desk" bylines), so the label is read off the source
    "Nastaliq/Fake News Detection/Domain_D_Fact_Checking_Platform": "real items carry an outlet dateline (\"92 News\", 90% of real vs 3.5% of fake items), so the label is predictable from the source",
    # minimum size criterion: at least 100 test items
    "Nastaliq/Sentiment Analysis/Domain_C_Civic_Social_Topics": "too small to evaluate (162 items, 33 in test; minimum is 100 test items)",
}


@dataclass(frozen=True)
class Task:
    """A transfer matrix: every domain in `domains` shares one label space."""
    tid: str
    title: str
    task_type: str       # one of TASK_TYPES
    script: str
    labels: tuple        # label strings exactly as stored in the CSVs, in index order
    label_names: tuple   # verbalizers used in LLM prompts, same order
    domains: tuple = field(default_factory=tuple)
    max_len: int = 128   # encoder tokens; also per-text token budget in LLM prompts
    pair_input: bool = False  # QA: text is "Question: ...\nAnswer: ..."


TASK_TYPES = ("Sentiment / polarity", "Abusive language", "Fake news", "QA pair validation")


def _keys(script, coll, letters):
    return tuple(d.key for d in DOMAINS
                 if d.script == script and d.collection == coll and d.letter in letters
                 and d.key not in DROPPED_V11)


TASKS = [
    Task("N-SA", "Nastaliq Sentiment", "Sentiment / polarity", "Nastaliq",
         ("0", "1"), ("Negative", "Positive"), _keys("Nastaliq", "Sentiment Analysis", "ABCDE")),
    Task("N-HS", "Nastaliq Hate Speech", "Abusive language", "Nastaliq",
         ("0", "1"), ("Normal", "Hate"), _keys("Nastaliq", "Hate Speech", "ABCDE")),
    Task("N-FN", "Nastaliq Fake News", "Fake news", "Nastaliq",
         ("0", "1"), ("Fake", "Real"), _keys("Nastaliq", "Fake News Detection", "ABCD"), max_len=256),
    Task("N-QA", "Nastaliq QA Pair Validation", "QA pair validation", "Nastaliq",
         ("0", "1"), ("Invalid", "Valid"), _keys("Nastaliq", "QA", "ABCD"), max_len=256, pair_input=True),
    Task("R-SA", "Roman Sentiment (binary)", "Sentiment / polarity", "Roman",
         ("0", "1"), ("Negative", "Positive"), _keys("Roman", "Sentiment Analysis", "ABCD")),
    Task("R-CA", "Roman Cyber Abuse", "Abusive language", "Roman",
         ("0", "1"), ("Normal", "Abusive"), _keys("Roman", "Cyber Abuse or Abusive Language", "ABCD")),
    # Product reviews and Twitter opinions share a 3-class label space, so they form
    # ONE transfer matrix (review round 1, item 5; round 2, item 3.4).
    Task("R-P3", "Roman 3-class Polarity (Reviews + Opinions)", "Sentiment / polarity", "Roman",
         ("Negative", "Neutral", "Positive"), ("Negative", "Neutral", "Positive"),
         _keys("Roman", "Product or eCommerce Reviews", "ABD")
         + _keys("Roman", "Twitter or Social Media Opinions", "ABCD")),  # TW-A dropped in v1.1
]
TASK_BY_ID = {t.tid: t for t in TASKS}

# Fake news: the original split script mapped Fake->0 and Real->1 (FN A-C);
# FN D (Notri-Fact Real/Unreal) is checked in build_splits.py.

FT_MODELS = {
    "XLM-R": "xlm-roberta-base",
    "mBERT": "bert-base-multilingual-cased",
}
LLM_MODELS = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
}
SEEDS = (13, 42, 87)          # fine-tuning seeds
EXEMPLAR_SEEDS = (13, 42, 87) # LLM demonstration draws
