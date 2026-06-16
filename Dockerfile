# Hugging Face Spaces runs this via the Docker SDK (see README.md frontmatter).
# HF requires a non-root user and the app on port 7860.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app
COPY --chown=user . /app

# pyproject.toml is the single source of truth for deps; no requirements.txt.
RUN pip install --no-cache-dir .

EXPOSE 7860
CMD ["solara", "run", "run_app.py", "--host", "0.0.0.0", "--port", "7860"]
