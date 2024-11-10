<img src="/docs/podcaster.png" width=80 height=80 align="right">

# podcaster

Turn your static blog into a podcast. 

[Original blog post.](https://duarteocarmo.com/blog/you-can-now-listen-to-this-blog)

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
$ make install-dev
```
5. Run
```bash
$ podcaster
```

Enjoy :)

## TODO

- [x] Pin those deps
- [x] Update model to F5 TTS with a decent recording
- [x] Add a better message that this is automated to audio and text 
- [x] IF possible add audio image to podcast
- [x] Better cover for podcast
- [x] Optional pass through OpenAI to clean up audio (make sure audio independent)
