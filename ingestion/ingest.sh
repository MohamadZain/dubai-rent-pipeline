#!/usr/bin/env bash
# =============================================================================
# Stage 2 - Ingestion: Kaggle -> VM -> ADLS Gen2 lake (raw, versioned, write-once)
# Project: Predicting Fair Monthly Rent for Dubai Ejari Contracts (DSAI3202 Phase 1)
#
# Adapted from the DSAI3202 Lab-3 ingestion pattern (source receipt + SHA-256,
# copy to raw, verify, safe retry). Changes for this project: Kaggle API source
# instead of the lab download, single CSV, versioned path raw/dld_rents/v1/,
# hash stored as blob metadata, and an explicit STOP if a different file would
# overwrite the preserved original.
#
# Usage (on the VM, after `az login`):
#   LAKE=amazonlake60306080 bash ingestion/ingest.sh
# Run it twice: the 2nd run must report SKIPPED and leave the blob unchanged.
# =============================================================================
set -euo pipefail

# ---------- settings ----------
LAKE="${LAKE:?Set LAKE to your storage account name, e.g. LAKE=amazonlake60306080}"
KAGGLE_REF="zainabhejairi/dubai-real-estates-1969-2023"
SOURCE_FILE="DLD_Rents.csv"
VERSION="v1"                                   # new source release => v2, never overwrite v1
CONTAINER="raw"
BLOB_PATH="dld_rents/${VERSION}/${SOURCE_FILE}"
MANIFEST_BLOB="dld_rents/${VERSION}/_manifest.json"
WORK="${HOME}/dld_download"                    # local working copy (not in git)
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EVID="${REPO_DIR}/ingestion"                   # manifest + run log saved here for git
KAGGLE="${HOME}/.local/bin/kaggle"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
export AZCOPY_AUTO_LOGIN_TYPE=AZCLI            # azcopy reuses your `az login`

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# ---------- 0. pre-checks ----------
say "Checking Azure sign-in and lake '${LAKE}'..."
az account show --query name -o tsv >/dev/null
az storage container show --account-name "$LAKE" -n "$CONTAINER" --auth-mode login -o none
mkdir -p "$WORK" "$EVID"

# ---------- 1. download (only if not already on the VM) ----------
LOCAL="${WORK}/${SOURCE_FILE}"
if [[ -f "$LOCAL" ]]; then
  say "Local copy exists, not downloading again: $LOCAL"
else
  say "Downloading ${SOURCE_FILE} from Kaggle (${KAGGLE_REF})..."
  "$KAGGLE" datasets download -d "$KAGGLE_REF" -f "$SOURCE_FILE" -p "$WORK"
  if [[ -f "${LOCAL}.zip" ]]; then
    say "Unzipping..."
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "${LOCAL}.zip" "$WORK"
  fi
fi
[[ -f "$LOCAL" ]] || { echo "STOP: $LOCAL not found after download"; exit 1; }

# ---------- 2. source receipt: identity, size, hash, rows, columns ----------
say "Hashing and counting rows (about 1-2 min)..."
BYTES=$(stat -c %s "$LOCAL")
SHA256=$(sha256sum "$LOCAL" | cut -d' ' -f1)
MD5_B64=$(openssl dgst -md5 -binary "$LOCAL" | base64)
read -r ROWS NCOLS COLS < <(python3 - "$LOCAL" <<'PY'
import csv, sys, json
with open(sys.argv[1], newline='', encoding='utf-8', errors='replace') as f:
    r = csv.reader(f)
    header = next(r)
    n = sum(1 for _ in r)
print(n, len(header), json.dumps(header, ensure_ascii=False))
PY
)
say "bytes=${BYTES} rows=${ROWS} columns=${NCOLS}"
say "sha256=${SHA256}"

# ---------- 3. upload write-once (never overwrite the preserved original) ----------
EXISTING_SHA=$(az storage blob show --account-name "$LAKE" -c "$CONTAINER" -n "$BLOB_PATH" \
                 --auth-mode login --query "metadata.sha256" -o tsv 2>/dev/null || true)

if [[ -z "$EXISTING_SHA" ]]; then
  ACTION="UPLOADED"
  say "No blob at ${CONTAINER}/${BLOB_PATH} -> uploading..."
  azcopy copy "$LOCAL" "https://${LAKE}.blob.core.windows.net/${CONTAINER}/${BLOB_PATH}" \
        --overwrite=false --put-md5 \
        --metadata="sha256=${SHA256};source=kaggle:${KAGGLE_REF};version=${VERSION}"
elif [[ "$EXISTING_SHA" == "$SHA256" ]]; then
  ACTION="SKIPPED"
  say "SKIPPED: blob already exists with the same SHA-256 - original preserved, nothing written."
else
  echo "STOP: ${CONTAINER}/${BLOB_PATH} exists with a DIFFERENT hash."
  echo "      stored=${EXISTING_SHA}"
  echo "      local =${SHA256}"
  echo "      Do not overwrite. If the source really changed, ingest it as a new VERSION (v2)."
  exit 2
fi

# ---------- 4. verify what is in the lake ----------
say "Verifying blob in the lake..."
VERIFY=$(az storage blob show --account-name "$LAKE" -c "$CONTAINER" \
   -n "$BLOB_PATH" --auth-mode login \
   --query "[properties.contentLength, metadata.sha256, properties.etag, properties.lastModified]" -o tsv | tr '\n' ' ')
read -r L_SIZE L_SHA L_ETAG L_MOD <<< "$VERIFY"
[[ "$L_SIZE" == "$BYTES" && "$L_SHA" == "$SHA256" ]] \
  && say "MATCH: lake size and SHA-256 equal the local source." \
  || { echo "FAIL: lake blob does not match local source"; exit 3; }
L_ETAG="${L_ETAG//\"/}"

# ---------- 5. manifest (written once per version) + append-only run log ----------
MANIFEST="${EVID}/manifest.json"
if [[ "$ACTION" == "UPLOADED" || ! -f "$MANIFEST" ]]; then
python3 - "$MANIFEST" <<PY
import json, sys
m = {
  "dataset": "Dubai Real Estates 1969-2023 (Kaggle mirror of Dubai Land Department open data)",
  "source_url": "https://www.kaggle.com/datasets/${KAGGLE_REF}",
  "kaggle_ref": "${KAGGLE_REF}",
  "file": "${SOURCE_FILE}",
  "format": "CSV, UTF-8, comma-separated, header row",
  "version": "${VERSION}",
  "bytes": ${BYTES},
  "sha256": "${SHA256}",
  "content_md5_base64": "${MD5_B64}",
  "data_rows": ${ROWS},
  "n_columns": ${NCOLS},
  "columns": ${COLS},
  "lake_path": "abfss://${CONTAINER}@${LAKE}.dfs.core.windows.net/${BLOB_PATH}",
  "ingestion_mode": "batch, one-time snapshot (Kaggle API -> VM -> AzCopy)",
  "refresh_strategy": "static snapshot; a new source release is ingested as a new version folder (v2); v1 is never overwritten",
  "lake_uploaded_utc": "${L_MOD}",
  "manifest_written_utc": "${RUN_TS}",
  "lake_etag_at_ingest": "${L_ETAG}"
}
json.dump(m, open(sys.argv[1], "w"), indent=2, ensure_ascii=False)
PY
  az storage blob upload --account-name "$LAKE" -c "$CONTAINER" -n "$MANIFEST_BLOB" \
     -f "$MANIFEST" --auth-mode login --overwrite false -o none 2>/dev/null \
     && say "Manifest saved to ${CONTAINER}/${MANIFEST_BLOB}" \
     || say "Manifest already in lake (kept unchanged)."
fi

LOG="${EVID}/ingest_runs.log"
echo -e "${RUN_TS}\t${ACTION}\tbytes=${BYTES}\tsha256=${SHA256}\tetag=${L_ETAG}\tlastModified=${L_MOD}" >> "$LOG"
say "Run logged in ${LOG}"
echo
echo "===== RESULT: ${ACTION} | rows=${ROWS} | etag=${L_ETAG} | lastModified=${L_MOD} ====="
