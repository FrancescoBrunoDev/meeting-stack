# meeting-job + the dashboard, in one container.
#
#   podman build -t localhost/meeting-job .
#   docker compose up -d --build    # or: podman-compose up -d --build
#
# A plain service: no podman socket, no device mounts, no nested containers —
# all inference (audio.cpp, Unsloth Studio, Mistral) is reached over HTTP.
FROM docker.io/library/debian:trixie-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 ca-certificates ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY meeting-job meeting-transcribe dashboard.html /app/
RUN chmod +x /app/meeting-job /app/meeting-transcribe

# The container writes into the mounted cache dir only.
ENV PYTHONUNBUFFERED=1
EXPOSE 8099
ENTRYPOINT ["/app/meeting-job"]
