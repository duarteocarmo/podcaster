FROM python:3.11-bookworm

WORKDIR /app
COPY requirements.txt src pyproject.toml data_raw /app/

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .

CMD ["podcaster"]
