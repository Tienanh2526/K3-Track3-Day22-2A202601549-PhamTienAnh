FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY configs ./configs
COPY data ./data
RUN pip install --no-cache-dir -e '.[dev]'
CMD ["pytest", "-q"]
