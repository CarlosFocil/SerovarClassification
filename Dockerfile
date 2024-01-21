FROM python:3.11-slim

RUN pip install poetry

WORKDIR /app
COPY ["pyproject.toml", "poetry.lock", "./"]

RUN poetry config virtualenvs.create false
RUN poetry install --no-root

COPY ["inference/predict.py", "models/RandomForest_SerovarClassifier_v1_test.bin", "./"]

EXPOSE 9696

ENV MODEL_FILE="RandomForest_SerovarClassifier_v1_test.bin"
ENV PORT=9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]