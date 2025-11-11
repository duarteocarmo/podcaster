# GENERAL
RESULTS_DIR = "transcripts/"
TRANSCRIBE_LAST_N_ARTICLES = 25
WEBSITE = "https://duarteocarmo.com"
FEED_URL = f"{WEBSITE}/feed.xml"
LLM_PREPROCESSING_MODEL = "openai/gpt-5"

# MODAL
MODAL_GPU = "a10g"
MODAL_MAX_CONTAINERS = 5
MODAL_APP_NAME = "podcaster"
MODAL_PYTHON_VERSION = "3.10"
MODAL_REMOTE_REFERENCE_AUDIO_PATH = "/root/reference.wav"
MODAL_LOCAL_REFERENCE_AUDIO_PATH = "./data_raw/reference_enhanced.wav"

# CHATTERBOX TTS
CHATTERBOX_MAX_CHARS_PER_CHUNK = 200
CHATTERBOX_EXAGGERATION = 0.2
CHATTERBOX_CFG_WEIGHT = 0.6
CHATTERBOX_TEMPERATURE = 0.5
CHATTERBOX_CROSSFADE_MS = 100

# STORAGE
BUCKET_NAME = "podcaster"
BUCKET_URL = (
    "https://cd7ef6e066027aea5df8b21d160a0431.r2.cloudflarestorage.com"
)
PUBLIC_BUCKET_URL = "https://r2.duarteocarmo.com"

# PODCAST
NAME = "Duarte O.Carmo"
PODCAST_CATEGORIES = [{"cat": "Technology"}]
PODCAST_DESCRIPTION = """
A podcast version of my personal website. All the episodes are generated from a text to speech model.
""".strip()
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
