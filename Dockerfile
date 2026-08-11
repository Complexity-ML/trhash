FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

ENV TRHASH_HOST=0.0.0.0
ENV TRHASH_PORT=8000
EXPOSE 8000
CMD ["trhash-server"]
