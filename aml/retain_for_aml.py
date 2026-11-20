"""retain_for_aml.py - freeze the Databricks export and write the Azure ML data asset file (Phase 2, Stage 0b).

    python3 aml/retain_for_aml.py                      # lake amazonlake60306080, export v1, asset version 1
    python3 aml/retain_for_aml.py --lake <account> --export v1 --version 1

Adapted from Lab-8 project/retain_data.py. What it does:
  1. Reads curated/dld_rents/aml_export/<export>/_export.json (written by databricks/06_export_for_aml.py)
     and stops unless every check in it passed.
  2. Downloads every Parquet part listed there, checks name, size and row count, and hashes each part.
  3. Builds a provenance record (source tables, Delta versions, parts and their SHA-256) and hashes it.
     That hash names the snapshot folder: curated/dld_rents/aml_snapshots/<hash>/.
  4. Copies the parts there with a _manifest.json. A rerun verifies an existing snapshot byte for byte
     and never overwrites it (same rules as Lab-8).
  5. Writes aml/data-asset.yml (dubai-rents-features, uri_folder, the snapshot path, tags) and
     docs/evidence/phase2/data_snapshot.json. You then run: az ml data create -f aml/data-asset.yml

The snapshot layout keeps both tables under one asset:
    <hash>/_manifest.json
    <hash>/features/part-....parquet         3,569,746 rows
    <hash>/baseline_area/part-....parquet    172 rows
train.py re-hashes the parts against _manifest.json before it trains (Lab-8 snapshot_fingerprint).
"""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TABLES = ("features", "baseline_area")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_remote(fs, path):
    h = hashlib.sha256()
    with fs.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def open_lake(lake, local_root):
    """Return (filesystem, path prefix). --local-root is only for testing without Azure."""
    if local_root:
        from fsspec.implementations.local import LocalFileSystem
        return LocalFileSystem(), local_root.rstrip("/") + "/"
    from adlfs import AzureBlobFileSystem
    from azure.identity import AzureCliCredential
    return AzureBlobFileSystem(account_name=lake, credential=AzureCliCredential()), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", default=os.environ.get("LAKE", "amazonlake60306080"))
    ap.add_argument("--export", default="v1", help="export version written by 06_export_for_aml")
    ap.add_argument("--version", default="1", help="data asset version to write into data-asset.yml")
    ap.add_argument("--local-root", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    fs, pre = open_lake(a.lake, a.local_root)
    src = f"{pre}curated/dld_rents/aml_export/{a.export}"

    # 1. The export report must exist and every check in it must have passed.
    if not fs.exists(f"{src}/_export.json"):
        raise SystemExit(f"STOP: {src}/_export.json not found. Run databricks/06_export_for_aml.py first.")
    export = json.loads(fs.cat_file(f"{src}/_export.json"))
    failed = [k for k, v in export.get("checks", {}).items() if v is not True]
    if failed or not export.get("checks"):
        raise SystemExit(f"STOP: the export report has failed checks: {failed}")
    print(f"export {a.export}: created {export['created_utc']}, all {len(export['checks'])} checks passed")

    with tempfile.TemporaryDirectory(prefix="dubai-retain-") as tmp:
        parts, cached = [], []
        # 2. Every part listed in the report, and nothing else, with its size, rows and hash.
        for table in TABLES:
            info = export[table]
            listed = {p["file"]: p["bytes"] for p in info["parts"]}
            found = {Path(p).name for p in fs.find(f"{src}/{info['folder']}") if p.endswith(".parquet")}
            if found != set(listed):
                raise SystemExit(f"STOP: {table} parts in the lake differ from _export.json: "
                                 f"lake {sorted(found)} vs report {sorted(listed)}")
            rows_total = 0
            for name in sorted(listed):
                remote = f"{src}/{info['folder']}/{name}"
                local = Path(tmp) / table / name
                local.parent.mkdir(parents=True, exist_ok=True)
                fs.get_file(remote, str(local))
                size = local.stat().st_size
                if size != listed[name]:
                    raise SystemExit(f"STOP: {table}/{name} is {size} bytes, the report says {listed[name]}")
                rows = pq.ParquetFile(local).metadata.num_rows
                rows_total += rows
                parts.append({"path": f"{table}/{name}", "sha256": sha_file(local), "rows": rows, "bytes": size})
                cached.append(local)
                print(f"hashed {table}/{name}: {rows:,} rows, {size:,} bytes")
            expected_rows = info["rows"] if table == "features" else info["areas"]
            if rows_total != expected_rows:
                raise SystemExit(f"STOP: {table} has {rows_total:,} rows, the report says {expected_rows:,}")

        # 3. Provenance -> snapshot hash (same construction as Lab-8: sha256 of the sorted-keys JSON).
        f, b = export["features"], export["baseline_area"]
        provenance = {
            "format": 1,
            "project": "dubai-rent-pipeline",
            "export_version": export["export_version"],
            "export_created_utc": export["created_utc"],
            "features_source_table": f["source_table"],
            "features_delta_version": f["delta_version"],
            "feature_version": f["feature_version"],
            "baseline_source_table": b["source_table"],
            "baseline_delta_version": b["delta_version"],
            "rows": f["rows"], "train_rows": f["train_rows"], "test_rows": f["test_rows"],
            "baseline_areas": b["areas"],
            "parts": sorted(parts, key=lambda p: p["path"]),
        }
        digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        manifest = {**provenance, "snapshot_sha256": digest}
        dst = f"{pre}curated/dld_rents/aml_snapshots/{digest}"
        marker = f"{dst}/_manifest.json"

        # 4. Copy once; a rerun only verifies (Lab-8 rules).
        if fs.exists(marker):
            if json.loads(fs.cat_file(marker)) != manifest:
                raise SystemExit("STOP: an existing snapshot manifest differs; refusing to overwrite it")
            actual = {p.removeprefix(dst + "/") for p in fs.find(dst) if p.endswith(".parquet")}
            if actual != {p["path"] for p in parts}:
                raise SystemExit("STOP: the retained snapshot part list differs; do not reuse it")
            for part in parts:
                if sha_remote(fs, f"{dst}/{part['path']}") != part["sha256"]:
                    raise SystemExit(f"STOP: retained {part['path']} was modified; do not reuse this snapshot")
            print("verified existing snapshot: same bytes, nothing copied")
        else:
            existing = {p.removeprefix(dst + "/") for p in fs.find(dst)} if fs.exists(dst) else set()
            if existing - {p["path"] for p in parts}:
                raise SystemExit("STOP: the destination has unexpected files; refusing to modify it")
            for local, part in zip(cached, parts):
                remote = f"{dst}/{part['path']}"
                if fs.exists(remote):          # an interrupted earlier copy: keep it only if identical
                    if sha_remote(fs, remote) != part["sha256"]:
                        raise SystemExit(f"STOP: {remote} exists with different bytes; refusing to overwrite")
                    print(f"already copied {part['path']}")
                else:
                    if a.local_root:           # a local test folder needs its directories; the lake does not
                        fs.makedirs(remote.rsplit("/", 1)[0], exist_ok=True)
                    fs.put_file(str(local), remote)
                    print(f"copied {part['path']}")
            fs.pipe_file(marker, json.dumps(manifest, indent=2).encode())
            print(f"wrote {marker.removeprefix(pre)}")

    # 5. The data asset file and the evidence file.
    uri = f"abfss://curated@{a.lake}.dfs.core.windows.net/dld_rents/aml_snapshots/{digest}/"
    yml = (
        "# Written by aml/retain_for_aml.py. Register with: az ml data create -f aml/data-asset.yml\n"
        "# The path is a content-addressed snapshot: its folder name is the hash of its _manifest.json.\n"
        "$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json\n"
        "name: dubai-rents-features\n"
        f"version: {a.version}\n"
        "type: uri_folder\n"
        f"description: Retained export {export['export_version']} of dld_rents_features_v1 (Delta v{f['delta_version']}) "
        f"and dld_rents_baseline_area_v1 (Delta v{b['delta_version']})\n"
        f"path: {uri}\n"
        "tags:\n"
        f"  snapshot_sha256: {digest}\n"
        f"  feature_version: {f['feature_version']}\n"
        f"  features_table: {f['source_table']}\n"
        f"  features_delta_version: \"{f['delta_version']}\"\n"
        f"  baseline_table: {b['source_table']}\n"
        f"  baseline_delta_version: \"{b['delta_version']}\"\n"
        f"  rows: \"{f['rows']}\"\n"
        f"  export_version: {export['export_version']}\n"
    )
    (REPO / "aml" / "data-asset.yml").write_text(yml)
    ev = REPO / "docs" / "evidence" / "phase2"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "data_snapshot.json").write_text(json.dumps({**manifest, "path": uri}, indent=2) + "\n")
    print(f"\nretained {f['rows']:,} feature rows + {b['areas']} baseline areas")
    print(f"snapshot SHA-256 {digest}")
    print(f"path {uri}")
    print("wrote aml/data-asset.yml and docs/evidence/phase2/data_snapshot.json")
    print("next: az ml data create -f aml/data-asset.yml")


if __name__ == "__main__":
    main()
