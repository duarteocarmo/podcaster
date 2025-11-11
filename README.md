# podcaster

Turn your static blog into a podcast. 

## Requirements

* A static website with an RSS feed
* An S3 compatible bucket (default uses [R2 from Cloudflare](https://developers.cloudflare.com/r2/))
* A [modal](https://modal.com/) account
* A small recording of your voice (~1min should be enough)


## Usage

1. Clone this repo
2. Tweak the configuration under `src/podcaster/config.py`
3. Set env variables (example under `.env_example`) and source them with `source .env_example`
4. Install dependencies
```bash
$ make install
```
5. Run
```bash
$ podcaster
```

Enjoy :)

## Blog posts

- [Original blog post: You can now listen to this blog](https://duarteocarmo.com/blog/you-can-now-listen-to-this-blog)
- [TTS still sucks](https://duarteocarmo.com/blog/tts-still-sucks)
- [Podcast TTS with F5-TTS in Python](https://duarteocarmo.com/blog/podcast-tts-f5-tts-python)
