FROM python:3.11-bookworm

COPY requirements.txt src pyproject.toml data_raw /

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .

# CMD ["podcaster"]
