#!/usr/bin/env bash
# Start a disposable MinIO and create the tetration test bucket.
# Linux CI and local Docker. Not used on macOS/Windows runners.
#
# Upstream `minio/minio` images are no longer anonymously pullable. This tag is
# the MinIO 2025.7.23 binary packaged by Bitnami.
set -euo pipefail

name=tetration-minio
image=bitnamilegacy/minio:2025.7.23-debian-12-r0
endpoint=http://127.0.0.1:9000
user=tetration
pass=tetration-secret
bucket=tetration-test

docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" \
  -p 9000:9000 \
  -e MINIO_ROOT_USER="$user" \
  -e MINIO_ROOT_PASSWORD="$pass" \
  -e MINIO_DEFAULT_BUCKETS="$bucket" \
  "$image"

ready=0
for _ in $(seq 1 60); do
  if curl -fsS "$endpoint/minio/health/live" >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "MinIO did not become healthy" >&2
  docker logs "$name" >&2 || true
  exit 1
fi

# Default-bucket creation races the health check. `mc` ships in this image.
docker exec "$name" mc alias set local "$endpoint" "$user" "$pass" >/dev/null
docker exec "$name" mc mb -p "local/$bucket" >/dev/null

echo "MinIO ready at $endpoint bucket $bucket"
