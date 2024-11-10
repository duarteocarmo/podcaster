# GENERAL
LANGUAGE = "en"
LOCAL_DATA_DIR = "data_raw/"
MODEL_DIR = (
    "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
)
MODEL_NAME = "F5-TTS"
RESULTS_DIR = "transcripts/"
TRANSCRIBE_LAST = 25
REFERENCE_VOICE = "/root/data/reference_enhanced.wav"
REFERENCE_TEXT = "/root/data/reference.txt"
WEBSITE = "https://duarteocarmo.com"
FEED_URL = f"{WEBSITE}/feed.xml"
PREPROCESS_WITH_LLM = True
PREPROCESSING_MODEL = "gpt-4o-mini"

# MODAL
MODAL_GPU = "any"
MODAL_NAME = "podcaster"
MODAL_REMOTE_DATA_DIR = "/root/data"

# STORAGE
BUCKET_NAME = "podcaster"
BUCKET_URL = (
    "https://cd7ef6e066027aea5df8b21d160a0431.r2.cloudflarestorage.com"
)
PUBLIC_BUCKET_URL = "https://r2.duarteocarmo.com"

# PODCAST
NAME = "Duarte O.Carmo"
PODCAST_CATEGORIES = [{"cat": "Technology"}]
PODCAST_DESCRIPTION = "My personal website. Here, you can read my blog posts, and learn about my projects. I write about data science, AI, programming, business, and other topics."
PODCAST_EXPLICIT = False
PODCAST_FEED_NAME = "podcast.xml"
PODCAST_IMAGE = f"{PUBLIC_BUCKET_URL}/cover.png"
PODCAST_NAME = f"{NAME}'s articles"
PODCAST_WEBSITE = WEBSITE
PODCAST_AUTHOR = {
    "name": NAME,
    "email": "me@duarteocarmo.com",
    "uri": PODCAST_WEBSITE,
}
