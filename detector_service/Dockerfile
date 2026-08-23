FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY detector_service/requirements.txt /tmp/runtime-requirements.txt

RUN python -m pip install \
        --no-cache-dir \
        --requirement /tmp/runtime-requirements.txt \
    && python -c "import cv2; assert cv2.__version__.startswith('4.13.'), cv2.__version__"

RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app

COPY --chown=app:app detector_service/ /app/detector_service/
COPY --chown=app:app LICENSE /app/LICENSE

RUN install -d -o app -g app /app/detector_service/storage/detections

EXPOSE 23000/udp

USER app

CMD ["python", "-m", "detector_service.app"]
