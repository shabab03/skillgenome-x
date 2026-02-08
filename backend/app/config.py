# app/config.py

# ---------- General ----------
APP_NAME = "SkillGenome X"
DEBUG = True

# ---------- Data ----------
DATA_PATH = "app/data/skillgenome_ready_dataset.csv"

# ---------- Bot Detection ----------
BOT_POSTS_PER_DAY_THRESHOLD = 40
BOT_DUPLICATE_TEXT_THRESHOLD = 0.75

# ---------- Skills ----------
MIN_SKILL_SUPPORT = 10

# ---------- Clustering ----------
CLUSTERING_METHOD = "dbscan"   # can switch to 'kmeans'
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 2

# ---------- Forecasting ----------
FORECAST_HORIZON_DAYS = 180
TIME_GRANULARITY = "week"