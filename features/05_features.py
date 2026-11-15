# Databricks notebook source
# MAGIC %md
# MAGIC # 05_features: initial feature table (Phase 1, guideline II.5)
# MAGIC Builds the Phase-0 features from `dld_rents_gold` and saves them as a **versioned** table,
# MAGIC `dld_rents_features_v1`, registered in Unity Catalog. The definitions are written down in
# MAGIC `features/features_spec.yaml` in the repo.
# MAGIC
# MAGIC | Feature | From | Phase-0 role |
# MAGIC |---|---|---|
# MAGIC | `area_name_en` | gold | location |
# MAGIC | `property_type_group` | `property_type` | Flat / Villa / Other |
# MAGIC | `bedrooms` | parsed from `property_sub_type` | Studio = 0, "N bed rooms" = N, other labels = null |
# MAGIC | `sub_type_group` | `property_sub_type` | bedroom label, or "other" for rare labels |
# MAGIC | `size_m2`, `log_size_m2` | gold | size |
# MAGIC | `reg_type` | gold | New / Renew |
# MAGIC | `contract_year` | gold | time (main model) |
# MAGIC | `contract_month` | gold | time (secondary question only) |
# MAGIC
# MAGIC Also saved: the **baseline lookup** `dld_rents_baseline_area_v1` (median monthly rent per m2 per area,
# MAGIC **training years only**, plus the overall training median as fallback), exactly as Phase 0 defines it.
# MAGIC
# MAGIC **Leakage checks run in this notebook:** no contract in both splits, baseline built from 2010-2021 only,
# MAGIC no feature computed from rent or from the contract end date. `plausible_row` is derived from rent, so it is
# MAGIC an evaluation filter for Phase 2, **never a model input**.
# MAGIC
# MAGIC Adapted from Lab-7 (versioned feature output, spec with source/type/availability, leakage review).
# MAGIC A managed feature store is not used: the guideline does not require one, and a versioned Delta table
# MAGIC plus a spec file in Git meets "versioned feature output or reproducible feature specification".

# COMMAND ----------

import json, datetime
from pyspark.sql import functions as F

spark.conf.set("spark.sql.session.timeZone", "UTC")

LAKE, CATALOG, SCHEMA = "amazonlake60306080", "dsai3202_databricks", "s_60306080"
VERSION = "v1"                       # bump to v2 for any change; v1 is never overwritten
GOLD_TABLE     = f"{CATALOG}.{SCHEMA}.dld_rents_gold"
FEATURES_TABLE = f"{CATALOG}.{SCHEMA}.dld_rents_features_{VERSION}"
BASELINE_TABLE = f"{CATALOG}.{SCHEMA}.dld_rents_baseline_area_{VERSION}"
FEATURES_PATH  = f"abfss://curated@{LAKE}.dfs.core.windows.net/dld_rents/features/{VERSION}"
BASELINE_PATH  = f"abfss://curated@{LAKE}.dfs.core.windows.net/dld_rents/baseline_area/{VERSION}"
REPORT_PATH    = f"abfss://curated@{LAKE}.dfs.core.windows.net/dld_rents/features/{VERSION}_report.json"

FEATURES = ["area_name_en", "property_type_group", "bedrooms", "sub_type_group", "size_m2", "log_size_m2",
            "reg_type", "contract_year", "contract_month"]
MAIN_MODEL_FEATURES = [f for f in FEATURES if f != "contract_month"]   # Phase-0 hypothesis model
TARGET, TARGET_AED = "log_monthly_rent", "monthly_rent_aed"
MIN_SUBTYPE_ROWS = 1000              # sub-type labels rarer than this become "other"

for t in [FEATURES_TABLE, BASELINE_TABLE]:
    if spark.catalog.tableExists(t):
        raise RuntimeError(f"STOP: {t} already exists. Versions are immutable: set VERSION = 'v2' for a new one.")

gold = spark.table(GOLD_TABLE)
gold_version = int(spark.sql(f"DESCRIBE HISTORY {GOLD_TABLE}").first()["version"])
N = gold.count()
print(f"{GOLD_TABLE} version {gold_version}: {N:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build the features

# COMMAND ----------

sub = F.lower(F.trim("property_sub_type"))
bedrooms = (F.when(sub == "studio", F.lit(0))
             .when(sub.rlike(r"^\d+\s*bed\s*rooms?\s*\+\s*hall"),
                   F.regexp_extract(sub, r"^(\d+)", 1).cast("int"))
             .otherwise(F.lit(None).cast("int")))

frequent = [r["property_sub_type"] for r in
            gold.groupBy("property_sub_type").count().filter(F.col("count") >= MIN_SUBTYPE_ROWS).collect()]

ptype = F.lower(F.trim("property_type"))
type_group = (F.when(ptype.isin("flat", "studio", "penthouse", "hotel apartments"), "Flat")
               .when(ptype.contains("villa"), "Villa")
               .otherwise("Other"))

features = (gold
  .withColumn("bedrooms", bedrooms)
  .withColumn("sub_type_group", F.when(F.col("property_sub_type").isin(frequent), F.col("property_sub_type"))
                                 .otherwise("other"))
  .withColumn("property_type_group", type_group)
  .withColumn("log_size_m2", F.log("size_m2"))
  .withColumn("rent_per_m2", F.col("monthly_rent_aed") / F.col("size_m2"))
  # evaluation filter, NOT a feature (computed from the target): same plain-sense limits as the EDA
  .withColumn("plausible_row", ~((F.col("monthly_rent_aed") > 1_000_000) | (F.col("size_m2") > 2000)
                                 | (F.col("rent_per_m2") < 1) | (F.col("rent_per_m2") > 1000)))
  .select("contract_id", "split", *FEATURES, TARGET, TARGET_AED, "plausible_row",
          F.lit(VERSION).alias("feature_version"), F.lit(gold_version).alias("gold_version")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Leakage and quality checks (any failure stops before anything is saved)

# COMMAND ----------

checks = {}
# a) one row per contract, so no contract can sit in both train and test
ids = features.agg(F.count("*").alias("n"), F.countDistinct("contract_id").alias("d")).first()
checks["one_row_per_contract"] = ids["n"] == ids["d"]
# b) split follows the time rule exactly
bad_split = features.filter("(split = 'train' AND contract_year > 2021) OR "
                            "(split = 'test' AND (contract_year < 2022 OR contract_year > 2023))").count()
checks["split_matches_years"] = bad_split == 0
# c) no feature is built from rent, contract amount, end date or duration
forbidden = {"annual_amount_aed", "contract_amount_aed", "monthly_rent_aed", "log_monthly_rent", "rent_per_m2",
             "end_date", "duration_days", "plausible_row", "published_run_id"}
checks["no_target_or_end_date_in_features"] = not (set(FEATURES) & forbidden)
# d) features have no missing values except bedrooms (null = not a bedroom label, by design)
nulls = features.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in FEATURES]).first().asDict()
checks["no_nulls_except_bedrooms"] = all(v == 0 for k, v in nulls.items() if k != "bedrooms")
for k, v in checks.items():
    print(f"{'PASS' if v else 'FAIL'}  {k}")
print("nulls per feature:", nulls)
assert all(checks.values()), f"Feature checks failed: {[k for k, v in checks.items() if not v]}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline lookup from training years only
# MAGIC Phase 0: predicted rent = size (m2) x the area's median monthly rent per m2, **computed from the training
# MAGIC years only**; areas not seen in training fall back to the overall training median. Building it from
# MAGIC 2010-2021 rows only is what keeps test-period rents out of the baseline.

# COMMAND ----------

train_ok = features.filter("split = 'train' AND plausible_row")
# rent_per_m2 is target-derived, so it is not kept in the feature table; compute it here from train rows
baseline = (train_ok.withColumn("rent_per_m2", F.col(TARGET_AED) / F.col("size_m2"))
            .groupBy("area_name_en")
            .agg(F.percentile_approx("rent_per_m2", 0.5, 100000).alias("median_rent_per_m2_train"),
                 F.count("*").alias("train_rows"),
                 F.min("contract_year").alias("first_year"), F.max("contract_year").alias("last_year")))
overall = (train_ok.withColumn("rent_per_m2", F.col(TARGET_AED) / F.col("size_m2"))
           .agg(F.percentile_approx("rent_per_m2", 0.5, 100000)).first()[0])
baseline = baseline.withColumn("overall_median_rent_per_m2_train", F.lit(float(overall))) \
                   .withColumn("baseline_version", F.lit(VERSION))
assert baseline.agg(F.max("last_year")).first()[0] <= 2021, "Baseline used test-period rows"
print(f"baseline areas: {baseline.count()}, overall training median = {overall:.2f} AED/m2/month")

test_areas = features.filter("split = 'test'").select("area_name_en").distinct()
unseen = test_areas.join(baseline.select("area_name_en"), "area_name_en", "left_anti").count()
print(f"test areas without a training median (use overall fallback): {unseen}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Save both versioned tables and register them

# COMMAND ----------

features.write.format("delta").mode("error").option("path", FEATURES_PATH).saveAsTable(FEATURES_TABLE)
baseline.write.format("delta").mode("error").option("path", BASELINE_PATH).saveAsTable(BASELINE_TABLE)

spark.sql(f"COMMENT ON TABLE {FEATURES_TABLE} IS 'Phase-1 features {VERSION} from dld_rents_gold version "
          f"{gold_version}. Definitions: features/features_spec.yaml. plausible_row is an evaluation filter, not a feature.'")
spark.sql(f"COMMENT ON TABLE {BASELINE_TABLE} IS 'Baseline lookup {VERSION}: median monthly rent per m2 by area, "
          f"training years 2010-2021 only; overall training median as fallback.'")
FEATURE_COMMENTS = {
  "bedrooms": "Parsed from property_sub_type: Studio = 0, N bed rooms+hall = N; other labels = null.",
  "sub_type_group": f"property_sub_type, labels with fewer than {MIN_SUBTYPE_ROWS} rows grouped as other.",
  "property_type_group": "Flat (incl. Studio, Penthouse, Hotel apartments), Villa (any villa type), Other.",
  "log_size_m2": "Natural log of size_m2.",
  "contract_month": "Used only for the secondary question (with vs without timing features).",
  "plausible_row": "FALSE if rent > 1M AED/month, size > 2000 m2, or rent per m2 < 1 or > 1000. Target-derived: NOT a feature.",
}
for col, txt in FEATURE_COMMENTS.items():
    spark.sql(f"ALTER TABLE {FEATURES_TABLE} ALTER COLUMN {col} COMMENT '{txt}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Report

# COMMAND ----------

saved = spark.table(FEATURES_TABLE)
split_rows = {r["split"]: r["count"] for r in saved.groupBy("split").count().collect()}
plaus = {str(r["plausible_row"]): r["count"] for r in saved.groupBy("plausible_row").count().collect()}
bed = {str(r["bedrooms"]): r["count"] for r in saved.groupBy("bedrooms").count().orderBy("bedrooms").collect()}
groups = {r["property_type_group"]: r["count"] for r in saved.groupBy("property_type_group").count().collect()}
report = {
  "feature_table": FEATURES_TABLE, "feature_path": FEATURES_PATH, "version": VERSION,
  "built_from": f"{GOLD_TABLE} version {gold_version}", "rows": saved.count(), "split_rows": split_rows,
  "features": FEATURES, "main_model_features": MAIN_MODEL_FEATURES, "target": TARGET,
  "checks": checks, "nulls": nulls, "bedrooms_counts": bed, "property_type_group_counts": groups,
  "sub_type_groups": len(frequent) + 1, "plausible_row_counts": plaus,
  "baseline_table": BASELINE_TABLE, "baseline_areas": baseline.count(),
  "baseline_overall_median_rent_per_m2": round(float(overall), 3), "test_areas_without_baseline": unseen,
  "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}
dbutils.fs.put(REPORT_PATH, json.dumps(report, indent=2, default=str), False)
print("saved", REPORT_PATH)
print(json.dumps(report, indent=2, default=str))
display(saved.limit(10))
