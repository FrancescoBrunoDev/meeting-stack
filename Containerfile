# meeting-job + the dashboard, in one container.
#
#   podman build -t localhost/meeting-job .
#   podman run --rm --network=host \
#     -v ~/.cache/mcp-meeting-transcribe:~/.cache/mcp-meeting-transcribe \
#     -v ~/.cache/huggingface:~/.cache/huggingface \
#     -v /run/user/$(id -u):/run/user/$(id -u) \
#     -e HOME=$HOME -e CONTAINER_HOST=unix:///run/user/$(id -u)/podman/podman.sock \
#     localhost/meeting-job --dir ~/.cache/mcp-meeting-transcribe
#
# The diarizer containers are not nested: `podman-remote` talks to the podman
# socket on the host, so each job still starts a sibling `podman run --rm` with
# the GPU devices, the images and the HF cache where they already are.
FROM docker.io/library/debian:trixie-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 podman-remote ca-certificates ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && command -v podman >/dev/null || ln -sf /usr/bin/podman-remote /usr/bin/podman

WORKDIR /app
COPY meeting-job meeting-transcribe lockfile.py dashboard.html /app/
RUN chmod +x /app/meeting-job /app/meeting-transcribe

# The container writes into the mounted cache dir and talks to the host podman.
ENV PYTHONUNBUFFERED=1
EXPOSE 8099
ENTRYPOINT ["/app/meeting-job"]
