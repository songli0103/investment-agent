# Builder
FROM python:3.11-slim AS builder
WORKDIR /app
RUN pip install uv==0.4.*
COPY pyproject.toml uv.lock ./
RUN uv export --no-dev --no-hashes -o requirements.txt && \
    pip wheel --wheel-dir=/wheels -r requirements.txt

# Runtime
FROM python:3.11-slim
WORKDIR /app
RUN useradd --create-home app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/*.whl && rm -rf /wheels
COPY src/ ./src/
USER app
EXPOSE 8501
ENV PYTHONUNBUFFERED=1
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "src/alphaquant/frontend/app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", \
            "--server.headless=true"]