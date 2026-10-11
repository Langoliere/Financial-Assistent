FROM python:3.12-slim

# Задаём рабочий каталог
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем inference-пакет и устанавливаем зависимости
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Создаём каталоги данных и артефактов
RUN mkdir -p data artifacts

# Публикуем порт API
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "investbot.main:app", "--host", "0.0.0.0", "--port", "8000"]
