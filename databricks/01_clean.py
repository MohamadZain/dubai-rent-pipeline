# Databricks notebook source
# MAGIC %md
# MAGIC # 01_clean: raw CSV -> bronze Parquet -> cleaned silver (task 1 of 3)
# MAGIC 1. Read the preserved raw CSV (read-only) with every column as text.
# MAGIC 2. **Bronze:** keep 16 of 28 columns, save as Parquet for this run.
# MAGIC 3. Convert types safely (unreadable values become null, they do not crash the job).
# MAGIC 4. Give each row its first failed rule; keep one row per `contract_id`.
# MAGIC 5. Save accepted rows (silver, Delta) and rejected rows with reasons (Parquet).
# MAGIC 6. Check the accounting: input = accepted + rejected. Save `clean.json`.
# MAGIC
# MAGIC Refuses to run twice with the same `run_id`, so a run's evidence is never overwritten.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

T0 = time.time()
# Two protections, as in Lab-6: this guard, and mode("error") on every write below
# (it fails if the folder already exists), so a run can never overwrite another run.
if path_exists(RUN_DIR):
    raise RuntimeError(f"STOP: {RUN_DIR} already exists. Each run needs a new run_id "
                       "(use a fresh 'Run now', not 'Repair run').")

# ---- 1. read raw CSV as text (no type guessing) ----
raw = (spark.read.option("header", True).option("inferSchema", False)
       .option("quote", '"').option("escape", '"').option("mode", "PERMISSIVE")
       .csv(INPUT_PATH))
missing = [c for c in KEEP_COLUMNS if c not in raw.columns]
if missing:
    raise RuntimeError(f"STOP: input is missing required columns {missing}")
print(f"source columns: {len(raw.columns)} (expected {SOURCE_COLUMNS})")

# ---- 2. bronze: reduce early, save Parquet ----
t = time.time()
raw.select(*KEEP_COLUMNS).write.mode("error").parquet(BRONZE_PATH)
bronze = spark.read.parquet(BRONZE_PATH)
input_rows = bronze.count()
bronze_secs = round(time.time() - t, 1)
print(f"bronze saved: {input_rows:,} rows, {len(KEEP_COLUMNS)} columns, {bronze_secs}s")

# COMMAND ----------

# ---- 3. safe type conversion ----
def num(c):  return F.expr(f"try_cast(trim({c}) AS double)")
def date(c): return F.expr(f"try_to_timestamp(trim({c}), '{DATE_FMT}')").cast("date")

typed = (bronze
  .withColumn("contract_id", F.trim("contract_id"))
  .withColumn("start_date", date("contract_start_date"))
  .withColumn("end_date",   date("contract_end_date"))
  .withColumn("annual_amount_aed", num("annual_amount"))
  .withColumn("contract_amount_aed", num("contract_amount"))
  .withColumn("size_m2", num("actual_area"))
  .withColumn("line_no", num("line_number").cast("int"))
  .withColumn("n_props", num("no_of_prop").cast("int"))
  .withColumn("area_id_int", num("area_id").cast("int"))
  .withColumn("duration_days", F.datediff("end_date", "start_date"))
  .withColumn("contract_year", F.year("start_date"))
  .withColumn("usage", F.lower(F.trim("property_usage_en")))
  .withColumn("type_text", F.concat_ws(" ", F.coalesce("ejari_property_type_en", F.lit("")),
                                            F.coalesce("ejari_property_sub_type_en", F.lit(""))))
)
for c in MODEL_FIELDS:
    typed = typed.withColumn(c, F.trim(F.col(c)))

# ---- 4. first failed rule ----
def blank(c): return F.col(c).isNull() | (F.col(c) == "")
reason = (F.when(blank("contract_id"), "missing_contract_id")
  .when(F.col("start_date").isNull(), "bad_start_date")
  .when(F.col("annual_amount_aed").isNull(), "bad_annual_amount")
  .when(F.coalesce(F.col("usage"), F.lit("")) != "residential", "not_residential")
  .when(F.col("type_text").rlike(EXCLUDED_TYPE_REGEX), "excluded_type")
  .when(~F.col("contract_year").between(YEAR_MIN, YEAR_MAX), "year_out_of_range")
  .when(F.col("duration_days").isNull() | ~F.col("duration_days").between(DUR_MIN, DUR_MAX), "bad_duration")
  .when(F.col("size_m2").isNull() | (F.col("size_m2") <= MIN_SIZE_M2), "bad_size")
  .when(F.col("annual_amount_aed") < MIN_ANNUAL_AED, "low_rent"))
for c in MODEL_FIELDS:
    reason = reason.when(blank(c), "missing_model_field")
classified = typed.withColumn("reject_reason", reason)

# one row per contract_id among the valid rows. Deterministic winner:
# lowest line number, then largest size, then highest rent, then a hash of the full row.
w = Window.partitionBy("contract_id").orderBy(
    F.col("reject_reason").isNull().desc(), F.col("line_no").asc_nulls_last(),
    F.col("size_m2").desc_nulls_last(), F.col("annual_amount_aed").desc_nulls_last(),
    F.xxhash64(*KEEP_COLUMNS).asc())
classified = (classified.withColumn("_rn", F.row_number().over(w))
  .withColumn("reject_reason", F.when(F.col("reject_reason").isNull() & (F.col("_rn") > 1),
                                      "duplicate_contract_id").otherwise(F.col("reject_reason")))
  .drop("_rn"))

# COMMAND ----------

# ---- 5. save silver (accepted) and rejects ----
t = time.time()
silver = (classified.filter(F.col("reject_reason").isNull())
  .select("contract_id", "start_date", "end_date", "contract_year",
          F.month("start_date").alias("contract_month"), "duration_days",
          F.col("contract_reg_type_en").alias("reg_type"),
          F.col("area_id_int").alias("area_id"), "area_name_en",
          F.col("ejari_property_type_en").alias("property_type"),
          F.col("ejari_property_sub_type_en").alias("property_sub_type"),
          F.col("property_usage_en").alias("usage"), "is_free_hold",
          F.col("tenant_type_en").alias("tenant_type"),
          "size_m2", "annual_amount_aed", "contract_amount_aed", F.col("n_props").alias("no_of_prop"),
          (F.col("annual_amount_aed") / 12).alias("monthly_rent_aed"),
          F.lit(RUN_ID).alias("run_id")))
silver.write.format("delta").mode("error").save(SILVER_PATH)

(classified.filter(F.col("reject_reason").isNotNull())
  .select(*KEEP_COLUMNS, "reject_reason", F.lit(RUN_ID).alias("run_id"))
  .write.mode("error").partitionBy("reject_reason").parquet(REJECTS_PATH))
write_secs = round(time.time() - t, 1)

# ---- 6. accounting: read back what was saved ----
accepted = spark.read.format("delta").load(SILVER_PATH).count()
by_reason = {r["reject_reason"]: r["count"] for r in
             spark.read.parquet(REJECTS_PATH).groupBy("reject_reason").count().collect()}
rejected = sum(by_reason.values())
balanced = (input_rows == accepted + rejected)

report = {
  "run_id": RUN_ID, "finished_utc": now_utc(), "input_path": INPUT_PATH,
  "input_rows": input_rows, "accepted_rows": accepted, "rejected_rows": rejected,
  "reject_share": round(rejected / input_rows, 4) if input_rows else None,
  "rejected_by_reason": {r: by_reason.get(r, 0) for r in REASONS},
  "balanced": balanced,
  "paths": {"bronze": BRONZE_PATH, "silver": SILVER_PATH, "rejects": REJECTS_PATH},
  "timing_seconds": {"csv_to_bronze": bronze_secs, "clean_and_write": write_secs,
                     "task_total": round(time.time() - T0, 1)},
}
write_json(f"{QUALITY_DIR}/clean.json", report)
print(json.dumps(report, indent=2))

if not balanced:
    raise RuntimeError(f"STOP: accounting does not balance: {input_rows} != {accepted} + {rejected}")
print(f"BALANCED: {input_rows:,} input = {accepted:,} accepted + {rejected:,} rejected")
