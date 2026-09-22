# Воспроизводимая сборка. Образ не ходит в интернет во время работы:
# все зависимости ставятся на этапе build, данные монтируются томом.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    DFMAS_ROOT=/app \
    LANG=C.UTF-8

WORKDIR /app

# unrar-free нужен только для распаковки исходного архива data_1.rar
RUN apt-get update \
 && apt-get install -y --no-install-recommends unrar-free \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml Makefile ./
COPY src ./src
COPY config ./config
COPY tests ./tests

RUN mkdir -p data/raw data/processed reports artifacts

ENTRYPOINT ["python", "-m", "dfmas.cli"]
CMD ["--help"]
