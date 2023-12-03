# GENERAL
FEED_URL = "https://duarteocarmo.com/feed.xml"
VOICE_FILE = "/root/data/duarte_conference_1min_enhanced.wav"
LOCAL_DATA_DIR = "data_raw/"
RESULTS_DIR = "transcripts/"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_DIR = (
    "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
)
LANGUAGE = "en"
TRANSCRIBE_LAST = 24

# PODCAST
PODCAST_NAME = "Duarte O.Carmo's articles"
PODCAST_DESCRIPTION = "My personal website. Here, you can read my blog posts, and learn about my projects. I write about data science, AI, programming, business, and other topics."
PODCAST_WEBSITE = "https://duarteocarmo.com"
PODCAST_EXPLICIT = False
PODCAST_CATEGORIES = ("technology",)
PODCAST_IMAGE = "https://duarteocarmo.com/images/logo.png"
PODCAST_FEED_NAME = "podcast.xml"


# MODAL
MODAL_REMOTE_DATA_DIR = "/root/data"
MODAL_NAME = "podcaster"
MODAL_GPU = "any"
MODAL_VOLUME_NAME = "model-volume"

# STORAGE
BUCKET_NAME = "podcaster"
BUCKET_URL = (
    "https://cd7ef6e066027aea5df8b21d160a0431.r2.cloudflarestorage.com"
)
PUBLIC_BUCKET_URL = "https://r2.duarteocarmo.com"
