from podcaster.modal_functions import transcribe_list_of
from podcaster.parser import get_articles

FEED_URL = "https://duarteocarmo.com/feed.xml"


if __name__ == "__main__":
    a = get_articles(FEED_URL)
    transcribe_list_of(articles=[a[0]], remote=True)
