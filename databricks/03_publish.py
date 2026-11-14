# Databricks notebook source
# MAGIC %md
# MAGIC # 03_publish: publish gold and register it in Unity Catalog (task 3 of 3)
# MAGIC Runs only if `02_validate` succeeded (job setting *Run if = All succeeded*), and checks
# MAGIC `gate.json` again before writing. Order follows Lab-6 `03_publish`:
# MAGIC 1. Build gold from this run's silver: target (`monthly_rent_aed`, `log_monthly_rent`) and the
# MAGIC    time split (train 2010-2021, test 2022-2023).
# MAGIC 2. **Rerun check before changing anything:** if `compare_to` is set, compare this run's full
# MAGIC    content with that earlier run (`exceptAll` both ways + fingerprint). A difference stops here.
# MAGIC 3. Create (first run) or replace (later runs) the Unity Catalog table `dld_rents_gold`,
# MAGIC    stored as Delta in `curated/dld_rents/gold`. Delta keeps every earlier version (no VACUUM).
# MAGIC 4. Add table and column comments (the catalog documentation).
# MAGIC 5. Write `success.json` **last**: if it exists, the whole run finished.
# MAGIC
# MAGIC **Adapted from** Lab-6 `03_publish.py` (table registration, compare-before-replace,
# MAGIC success record last). **Changed:** no product join/quarantine; rent target, log target and
# MAGIC time split added; column comments for the catalog; fingerprint added to the comparison.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

T0 = time.time()
gate = read_json(f"{QUALITY_DIR}/gate.json")
if gate["verdict"] != "PASS":
    raise RuntimeError(f"STOP: gate verdict is {gate['verdict']}; not publishing.")

silver = spark.read.format("delta").load(SILVER_PATH)
silver_rows = silver.count()
gold = (silver
  .withColumn("log_monthly_rent", F.log("monthly_rent_aed"))
  .withColumn("split", F.when(F.col("contract_year") <= TRAIN_YEARS[1], "train").otherwise("test"))
  .withColumnRenamed("run_id", "published_run_id")
  .select("contract_id", "start_date", "contract_year", "contract_month", "reg_type",
          "area_id", "area_name_en", "property_type", "property_sub_type", "usage", "is_free_hold",
          "tenant_type", "size_m2", "duration_days", "annual_amount_aed", "monthly_rent_aed",
          "log_monthly_rent", "split", "published_run_id"))
content_cols = [c for c in gold.columns if c != "published_run_id"]
fp = fingerprint(gold, content_cols)

# COMMAND ----------

# ---- rerun check, BEFORE the published table changes ----
comparison = {"run_id": RUN_ID, "compared_to": COMPARE_TO or None, "checked": bool(COMPARE_TO)}
if COMPARE_TO:
    base = read_json(abfss(f"curated/dld_rents/quality/run_id={COMPARE_TO}/success.json"))
    base_silver = spark.read.format("delta").load(base["silver_path"]).drop("run_id")
    comparison["same_content_exceptAll"] = same_content(silver.drop("run_id"), base_silver)
    comparison["same_gold_fingerprint"] = base["content_fingerprint"] == fp
    comparison["this_fingerprint"], comparison["baseline_fingerprint"] = fp, base["content_fingerprint"]
    write_json(f"{QUALITY_DIR}/comparison.json", comparison)
    assert comparison["same_content_exceptAll"] and comparison["same_gold_fingerprint"], \
        f"Rerun changed values compared with run {COMPARE_TO}. Nothing published."
    print("SAME CONTENT as run", COMPARE_TO)

# COMMAND ----------

# ---- publish: same registration method as Lab-6 ----
# First run: write files + register the table in one step (needs CREATE EXTERNAL TABLE on curated).
# Later runs: check the table's location and schema, then replace its contents (new Delta version).
if spark.catalog.tableExists(GOLD_TABLE):
    detail = spark.sql(f"DESCRIBE DETAIL {GOLD_TABLE}").first()
    assert detail["format"] == "delta" and detail["location"].rstrip("/") == GOLD_PATH, \
        f"{GOLD_TABLE} points to {detail['location']}, expected {GOLD_PATH}. Stop and check."
    existing = spark.table(GOLD_TABLE)
    assert existing.schema.simpleString() == gold.schema.simpleString(), "Existing gold schema differs."
    gold.select(*existing.columns).write.mode("overwrite").insertInto(GOLD_TABLE)
else:
    gold.write.format("delta").mode("error").option("path", GOLD_PATH).saveAsTable(GOLD_TABLE)

# ---- catalog documentation (shows in Catalog Explorer) ----
COMMENTS = {
  "contract_id": "Ejari contract id. Primary key: one row per contract (duplicates removed in 01_clean).",
  "start_date": "Contract start date (source format dd-MM-yyyy).",
  "contract_year": "Year of start_date, 2010-2023. Drives the time split.",
  "contract_month": "Month of start_date, 1-12.",
  "reg_type": "New or Renew (contract_reg_type_en).",
  "area_id": "DLD area id.",
  "area_name_en": "DLD area name, e.g. Business Bay. Baseline groups by this.",
  "property_type": "Ejari property type, e.g. Flat, Villa (ejari_property_type_en).",
  "property_sub_type": "Ejari sub-type, holds bedrooms, e.g. 2bed room+Hall (ejari_property_sub_type_en).",
  "usage": "Property usage; always Residential after cleaning.",
  "is_free_hold": "Freehold flag from source (text).",
  "tenant_type": "Tenant type (Person/Authority). EDA only, not a Phase-0 feature.",
  "size_m2": "Property size in m2 (actual_area), above 10.",
  "duration_days": "end_date minus start_date in days, 330-400.",
  "annual_amount_aed": "Annualised rent in AED from source; at least 5000.",
  "monthly_rent_aed": "TARGET: annual_amount_aed / 12.",
  "log_monthly_rent": "Model target: natural log of monthly_rent_aed (rents are right-skewed).",
  "split": "train = 2010-2021, test = 2022-2023 (time-based split from Phase 0).",
  "published_run_id": "Databricks job run that published this version.",
}
spark.sql(f"COMMENT ON TABLE {GOLD_TABLE} IS 'Cleaned residential Ejari rent contracts (Dubai Land "
          f"Department via Kaggle; original raw/dld_rents/v1/DLD_Rents.csv, read via the hash-checked "
          f"landing copy in processed). One row per contract_id. Built by 01_clean, 02_validate, 03_publish.'")
for col, txt in COMMENTS.items():
    spark.sql(f"ALTER TABLE {GOLD_TABLE} ALTER COLUMN {col} COMMENT '{txt.replace(chr(39), '')}'")

# COMMAND ----------

# ---- verify, then write success.json LAST ----
published = spark.table(GOLD_TABLE)
gold_rows = published.count()
assert gold_rows == silver_rows, f"STOP: gold rows {gold_rows} differ from silver rows {silver_rows}"
assert fingerprint(published, content_cols) == fp, "STOP: published table differs from this run's gold"
version = int(spark.sql(f"DESCRIBE HISTORY {GOLD_TABLE}").first()["version"])
split_counts = {r["split"]: r["count"] for r in published.groupBy("split").count().collect()}

success = {
  "run_id": RUN_ID, "published_utc": now_utc(), "gold_table": GOLD_TABLE, "gold_path": GOLD_PATH,
  "gold_version": version, "gold_rows": gold_rows, "silver_rows": silver_rows, "silver_path": SILVER_PATH,
  "split_rows": split_counts, "content_fingerprint": fp, "comparison": comparison,
  "publish_seconds": round(time.time() - T0, 1),
}
write_json(f"{QUALITY_DIR}/success.json", success)
print(json.dumps(success, indent=2))
print(f"PUBLISHED {GOLD_TABLE} version {version}: {gold_rows:,} rows")
