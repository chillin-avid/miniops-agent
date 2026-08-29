# 使用轻量 Python Linux 镜像，并固定 Python 3.11 大版本。
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先复制依赖文件，源码变化时可以复用依赖安装缓存。
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# 使用普通用户运行服务，避免应用默认获得 root 权限。
RUN addgroup --system miniops && \
    adduser --system --ingroup miniops miniops && \
    mkdir -p /app/data && \
    chown -R miniops:miniops /app

COPY --chown=miniops:miniops . .

USER miniops
EXPOSE 8090

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
