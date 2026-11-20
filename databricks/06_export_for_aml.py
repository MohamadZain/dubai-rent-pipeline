# Databricks notebook source
# MAGIC %md
# MAGIC # 06_export_for_aml: hand the Phase-1 feature table to Azure ML (Phase 2, Stage 0a)
# MAGIC Azure ML jobs read plain Parquet from the lake; they do not read Unity Catalog Delta tables.
# MAGIC This notebook exports, **once**, the two Phase-1 tables the model needs:
# MAGIC
# MAGIC | Table | Rows | Used for |
# MAGIC |---|---|---|
# MAGIC | `dld_rents_features_v1` | 3,569,746 | features, target, split, `plausible_row` |
# MAGIC | `dld_rents_baseline_area_v1` | 172 areas | the Phase-0 area-median baseline |
# MAGIC
# MAGIC Both are read at a **pinned Delta version** (recorded in `_export.json`), checked against the
# MAGIC Phase-1 numbers, and written as plain Parquet to `curated/dld_rents/aml_export/v1/`.
# MAGIC The export is write-once (`mode("error")`): a rerun stops instead of replacing it.
# MAGIC
# MAGIC **Next (Stage 0b, on the VM):** `aml/retain_for_aml.py` hashes these files, copies them to a folder
# MAGIC named after that hash, and registers it as the Azure ML data asset `dubai-rents-features:1`
# MAGIC (the Lab-8 `retain_data.py` pattern). Nothing in this notebook changes the Phase-1 tables.

# COMMAND ----------

import json, datetime
from pyspark.sql import functions as F

spark.conf.set("spark.sql.session.timeZone", "UTC")

LAKE, CATALOG, SCHEMA = "amazonlake60306080", "dsai3202_databricks", "s_60306080"
FEATURE_VERSION = "v1"      # the Phase-1 feature version being exported
EXPORT_VERSION  = "v1"      # bump to v2 only if a different feature version is exported

FEATURES_TABLE = f"{CATALOG}.{SCHEMA}.dld_rents_features_{FEATURE_VERSION}"
BASELINE_TABLE = f"{CATALOG}.{SCHEMA}.dld_rents_baseline_area_{FEATURE_VERSION}"
EXPORT_ROOT    = f"abfss://curated@{LAKE}.dfs.core.windows.net/dld_rents/aml_export/{EXPORT_VERSION}"

# Phase-1 numbers (README, 05_features report). Any difference stops the export.
EXPECTED = {"rows": 3_569_746, "train": 2_646_969, "test": 922_777,
            "implausible": 32_406, "baseline_areas": 172}

FEATURE_COLUMNS = ["contract_id", "split",
                   "area_name_en", "property_type_group", "bedrooms", "sub_type_group",
                   "size_m2", "log_size_m2", "reg_type", "contract_year", "contract_month",
                   "log_monthly_rent", "monthly_rent_aed", "plausible_row",
                   "feature_version", "gold_version"]


def path_exists(p):
    try:
        dbutils.fs.ls(p)
        return True
    except Exception as e:  # dbutils raises a Java FileNotFoundException for a missing path
        if "FileNotFound" in str(e) or "does not exist" in str(e) or "PathNotFound" in str(e):
            return False
        raise


if path_exists(EXPORT_ROOT):
    raise RuntimeError(f"STOP: {EXPORT_ROOT} already exists. The export is write-once; "
                       "use the existing one, or set EXPORT_VERSION = 'v2' for a new export.")
print("export target is free:", EXPORT_ROOT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pin the Delta versions and read the tables

# COMMAND ----------

def latest_version(table):
    return int(spark.sql(f"DESCRIBE HISTORY {table}").agg(F.max("version")).first()[0])

feat_v = latest_version(FEATURES_TABLE)
base_v = latest_version(BASELINE_TABLE)
feat = spark.sql(f"SELECT * FROM {FEATURES_TABLE} VERSION AS OF {feat_v}")
base = spark.sql(f"SELECT * FROM {BASELINE_TABLE} VERSION AS OF {base_v}")
print(f"{FEATURES_TABLE} Delta version {feat_v}")
print(f"{BASELINE_TABLE} Delta version {base_v}")
feat.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Check against the Phase-1 numbers (any FAIL stops before anything is written)

# COMMAND ----------

stats = feat.agg(
    F.count("*").alias("rows"),
    F.countDistinct("contract_id").alias("ids"),
    F.sum((F.col("split") == "train").cast("int")).alias("train"),
    F.sum((F.col("split") == "test").cast("int")).alias("test"),
    F.sum((~F.col("plausible_row")).cast("int")).alias("implausible"),
    F.max(F.when(F.col("split") == "train", F.col("contract_year"))).alias("train_max_year"),
    F.min(F.when(F.col("split") == "test", F.col("contract_year"))).alias("test_min_year"),
).first().asDict()
base_stats = base.agg(F.count("*").alias("areas"), F.max("last_year").alias("last_year"),
                      F.countDistinct("overall_median_rent_per_m2_train").alias("fallbacks"),
                      F.first("overall_median_rent_per_m2_train").alias("fallback")).first().asDict()

checks = {
    "columns_as_phase1":    feat.columns == FEATURE_COLUMNS,
    "rows":                 stats["rows"] == EXPECTED["rows"],
    "one_row_per_contract": stats["ids"] == stats["rows"],
    "train_rows":           stats["train"] == EXPECTED["train"],
    "test_rows":            stats["test"] == EXPECTED["test"],
    "train_ends_2021":      stats["train_max_year"] == 2021,
    "test_starts_2022":     stats["test_min_year"] == 2022,
    "implausible_rows":     stats["implausible"] == EXPECTED["implausible"],
    "baseline_areas":       base_stats["areas"] == EXPECTED["baseline_areas"],
    "baseline_train_only":  base_stats["last_year"] <= 2021,
    "one_fallback_value":   base_stats["fallbacks"] == 1,
}
for k, v in checks.items():
    print(f"{'PASS' if v else 'FAIL'}  {k}")
print("feature stats:", stats)
print("baseline stats:", base_stats)
if not checks["columns_as_phase1"]:
    print("columns found:", feat.columns)
assert all(checks.values()), f"Export checks failed: {[k for k, v in checks.items() if not v]}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write plain Parquet, once
# MAGIC One file per table, rows sorted by key, so the Azure ML job reads it in one go and a reader can
# MAGIC check row order. `mode("error")` refuses to write into an existing folder.

# COMMAND ----------

FEATURES_OUT = f"{EXPORT_ROOT}/features"
BASELINE_OUT = f"{EXPORT_ROOT}/baseline_area"

(feat.select(*FEATURE_COLUMNS).repartition(1).sortWithinPartitions("contract_id")
     .write.mode("error").parquet(FEATURES_OUT))
(base.repartition(1).sortWithinPartitions("area_name_en")
     .write.mode("error").parquet(BASELINE_OUT))
print("written:", FEATURES_OUT, "and", BASELINE_OUT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Read back, compare, and write `_export.json`

# COMMAND ----------

back_feat = spark.read.parquet(FEATURES_OUT)
back_base = spark.read.parquet(BASELINE_OUT)
same_rows = back_feat.count() == stats["rows"] and back_base.count() == base_stats["areas"]
# exceptAll in both directions: the export holds exactly the table's rows (Lab-6 comparison idea)
same_content = (back_feat.exceptAll(feat.select(*FEATURE_COLUMNS)).limit(1).count() == 0 and
                feat.select(*FEATURE_COLUMNS).exceptAll(back_feat).limit(1).count() == 0)
print(f"read-back rows match: {same_rows}; read-back content matches: {same_content}")
assert same_rows and same_content, "Read-back differs from the source table"


def parquet_parts(folder):
    return [{"file": f.name, "bytes": f.size} for f in dbutils.fs.ls(folder) if f.name.endswith(".parquet")]


report = {
    "export_version": EXPORT_VERSION,
    "export_root": EXPORT_ROOT,
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "features": {"source_table": FEATURES_TABLE, "delta_version": feat_v,
                 "feature_version": FEATURE_VERSION, "folder": "features",
                 "rows": stats["rows"], "train_rows": stats["train"], "test_rows": stats["test"],
                 "implausible_rows": stats["implausible"], "columns": FEATURE_COLUMNS,
                 "parts": parquet_parts(FEATURES_OUT)},
    "baseline_area": {"source_table": BASELINE_TABLE, "delta_version": base_v,
                      "folder": "baseline_area", "areas": base_stats["areas"],
                      "overall_median_rent_per_m2_train": base_stats["fallback"],
                      "columns": base.columns, "parts": parquet_parts(BASELINE_OUT)},
    "checks": {**checks, "read_back_rows": same_rows, "read_back_content": same_content},
}
dbutils.fs.put(f"{EXPORT_ROOT}/_export.json", json.dumps(report, indent=2, default=str), False)
print(json.dumps(report, indent=2, default=str))
