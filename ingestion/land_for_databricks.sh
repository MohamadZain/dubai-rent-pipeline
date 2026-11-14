#!/usr/bin/env bash
# =============================================================================
# Landing copy for Databricks: raw (preserved original) -> processed/dld_rents/landing/v1/
#
# Why: Unity Catalog in our workspace has external locations only for the
# `processed` and `curated` containers, so Databricks cannot read `raw`.
# The original in raw/dld_rents/v1/ is never touched. This script uploads the SAME
# file (checked against the SHA-256 in ingestion/manifest.json) to a landing folder
# that the Databricks job reads. Write-once: a rerun skips; a different file STOPS.
#
# Usage (on the VM, after ingest.sh):  LAKE=amazonlake60306080 bash ingestion/land_for_databricks.sh
# =============================================================================
set -euo pipefail
LAKE="${LAKE:?Set LAKE, e.g. LAKE=amazonlake60306080}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${REPO_DIR}/ingestion/manifest.json"
LOCAL="${HOME}/dld_download/DLD_Rents.csv"
DEST_CONTAINER="processed"
DEST_BLOB="dld_rents/landing/v1/DLD_Rents.csv"
export AZCOPY_AUTO_LOGIN_TYPE=AZCLI
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

EXPECTED=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" "$MANIFEST")
say "Manifest SHA-256: $EXPECTED"
say "Hashing local file (about 30 s)..."
ACTUAL=$(sha256sum "$LOCAL" | cut -d' ' -f1)
[[ "$ACTUAL" == "$EXPECTED" ]] || { echo "STOP: local file does not match the manifest"; exit 1; }
say "MATCH: local file is the preserved original."

EXISTING=$(az storage blob show --account-name "$LAKE" -c "$DEST_CONTAINER" -n "$DEST_BLOB" \
             --auth-mode login --query "metadata.sha256" -o tsv 2>/dev/null || true)
if [[ -z "$EXISTING" ]]; then
  ACTION="UPLOADED"
  azcopy copy "$LOCAL" "https://${LAKE}.blob.core.windows.net/${DEST_CONTAINER}/${DEST_BLOB}" \
     --overwrite=false --put-md5 \
     --metadata="sha256=${EXPECTED};copy_of=raw/dld_rents/v1/DLD_Rents.csv"
elif [[ "$EXISTING" == "$EXPECTED" ]]; then
  ACTION="SKIPPED"; say "SKIPPED: landing copy already exists with the same SHA-256."
else
  echo "STOP: landing blob exists with a different hash ($EXISTING). Not overwriting."; exit 2
fi

VERIFY=$(az storage blob show --account-name "$LAKE" -c "$DEST_CONTAINER" -n "$DEST_BLOB" --auth-mode login \
          --query "[properties.contentLength, metadata.sha256, properties.etag]" -o tsv | tr '\n' ' ')
read -r SIZE SHA ETAG <<< "$VERIFY"
[[ "$SHA" == "$EXPECTED" && "$SIZE" == "$(stat -c %s "$LOCAL")" ]] \
  && say "MATCH: landing copy size and SHA-256 equal the manifest." \
  || { echo "FAIL: landing copy does not match"; exit 3; }
echo -e "$(date -u +%Y%m%dT%H%M%SZ)\tLANDING_${ACTION}\t${DEST_CONTAINER}/${DEST_BLOB}\tsha256=${SHA}\tetag=${ETAG//\"/}" \
  >> "${REPO_DIR}/ingestion/ingest_runs.log"
echo "===== RESULT: LANDING ${ACTION} | ${DEST_CONTAINER}/${DEST_BLOB} | etag=${ETAG//\"/} ====="
