# Короткие команды. Каждая цель воспроизводима и не требует ручных шагов.
PY ?= python3
export PYTHONPATH := src

.PHONY: help data fit report scenarios sweep backtest dashboard test all docker clean

help:
	@echo "make data       — распаковать архив и собрать витрину parquet"
	@echo "make fit        — обучить и откалибровать модели"
	@echo "make report     — сгенерировать reports/*.md"
	@echo "make scenarios  — прогнать все тестовые сценарии"
	@echo "make sweep      — развёртка решения по сере сырья и марке топлива"
	@echo "make backtest   — бэктест на отложенной истории (~4 мин)"
	@echo "make dashboard  — собрать artifacts/dashboard.html"
	@echo "make test       — запустить тесты"
	@echo "make all        — полный конвейер с нуля"

data/raw/242000_tags.csv:
	@mkdir -p data/raw
	@test -f data_1.rar || (echo "Положите data_1.rar в корень проекта" && exit 1)
	unrar-free x -o+ data_1.rar data/raw/ >/dev/null || unrar x -o+ data_1.rar data/raw/
	@mv -f data/raw/data/*.csv data/raw/ 2>/dev/null || true

data: data/raw/242000_tags.csv
	$(PY) -m dfmas.cli ingest

fit:
	$(PY) -m dfmas.cli fit

report:
	$(PY) -m dfmas.cli report

scenarios:
	$(PY) -m dfmas.cli scenario --all

sweep:
	$(PY) -m dfmas.cli sweep --at "2024-04-26 20:00:00"

backtest:
	$(PY) -m dfmas.cli backtest

dashboard:
	$(PY) -m dfmas.cli dashboard

test:
	$(PY) -m pytest tests -q

all:
	$(PY) -m dfmas.cli all

docker:
	docker compose run --rm dfmas all

clean:
	rm -rf data/processed/* reports/scenarios/* artifacts/* .pytest_cache
