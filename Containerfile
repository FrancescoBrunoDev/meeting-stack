# meeting-job + the dashboard, in one container.
#
#   podman build -t localhost/meeting-job .
#   podman run --rm --network=host \
#     -v ~/.cache/mcp-meeting-transcribe:~/.cache/mcp-meeting-transcribe \
#     -v ~/audio:/asr-share \
#     -e HOME=$HOME -e MISTRAL_API_KEY=$MISTRAL_API_KEY \
#     localhost/meeting-job --dir ~/.cache/mcp-meeting-transcribe --asr-dir /asr-share
#
# A plain service: no podman socket, no device mounts, no nested containers —
# all inference (audio.cpp, Mistral, Lemonade, the Bifrost gateway) is reached
# over HTTP.
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
