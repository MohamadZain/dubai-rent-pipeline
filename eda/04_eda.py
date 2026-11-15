# Databricks notebook source
# MAGIC %md
# MAGIC # 04_eda: exploratory data analysis of the published gold table
# MAGIC **Project:** Predicting fair monthly rent for Dubai Ejari contracts (DSAI3202 Phase 1, guideline II.4).
# MAGIC
# MAGIC Runs on **all rows** of `dld_rents_gold` (no sampling) plus run 1's bronze and rejects for
# MAGIC before/after counts. Every section ends with **what it means for modelling**.
# MAGIC
# MAGIC 1. Population counts
# MAGIC 2. Target distribution (monthly rent, log rent)
# MAGIC 3. Rent per m2 by area
# MAGIC 4. Change over time
# MAGIC 5. Relationships with the planned features
# MAGIC 6. Outliers
# MAGIC 7. What cleaning removed (before vs after)
# MAGIC 8. Test-period coverage (risk for the time split)
# MAGIC 9. Summary saved to the lake
# MAGIC
# MAGIC Not a job task: run it by hand (Run all), then **File > Export > HTML** into `eda/` in the repo.
# MAGIC Adapted from the Lab-6 `profile` idea (whole-table profile, approximate quantiles); the
# MAGIC questions, charts and checks are new for this dataset.

# COMMAND ----------

import json, datetime
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

spark.conf.set("spark.sql.session.timeZone", "UTC")

LAKE    = "amazonlake60306080"
CATALOG = "dsai3202_databricks"
SCHEMA  = "s_60306080"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.dld_rents_gold"
RUN1 = "1037737501698627"   # run whose bronze/rejects we compare with (same data as the published gold)
BRONZE_PATH  = f"abfss://processed@{LAKE}.dfs.core.windows.net/dld_rents/runs/run_id={RUN1}/bronze"
REJECTS_PATH = f"abfss://processed@{LAKE}.dfs.core.windows.net/dld_rents/runs/run_id={RUN1}/rejects"
EDA_DIR      = f"abfss://curated@{LAKE}.dfs.core.windows.net/dld_rents/eda"
REL_ERR = 0.001             # approxQuantile rank error: each quantile within 0.1% of its true rank

# chart style: one blue for a single series; blue/orange for two groups; recessive grid
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "font.size": 10, "legend.frameon": False})

def thousands(ax, axis="y"):
    f = plt.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)

gold = spark.table(GOLD_TABLE).withColumn("rent_per_m2", F.col("monthly_rent_aed") / F.col("size_m2")).cache()
N = gold.count()
summary = {"gold_table": GOLD_TABLE, "rows": N, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
print(f"{GOLD_TABLE}: {N:,} rows (all used, no sample)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Population counts

# COMMAND ----------

by_year = (gold.groupBy("contract_year", "split").count().orderBy("contract_year").toPandas())
by_type = gold.groupBy("property_type").count().orderBy(F.desc("count")).toPandas()
n_areas = gold.select("area_name_en").distinct().count()
split_counts = {r["split"]: r["count"] for r in gold.groupBy("split").count().collect()}
summary["split_rows"] = split_counts
summary["areas"] = n_areas
summary["rows_per_year"] = {int(r.contract_year): int(r["count"]) for _, r in by_year.iterrows()}
summary["rows_per_property_type"] = dict(zip(by_type.property_type, by_type["count"].astype(int)))

fig, ax = plt.subplots(figsize=(9, 3.6))
for split, color in [("train", BLUE), ("test", ORANGE)]:
    d = by_year[by_year.split == split]
    ax.bar(d.contract_year, d["count"], width=0.7, color=color, label=split)
test_share = split_counts.get("test", 0) / N
ax.set_title(f"Contracts per year: the 2022-2023 test period holds {test_share:.1%} of rows")
ax.set_xticks(by_year.contract_year.unique()); thousands(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
ax.set_ylabel("contracts")
plt.tight_layout(); plt.show()

print(f"areas: {n_areas}   split: {split_counts}")
display(by_type)

# COMMAND ----------

# MAGIC %md
# MAGIC **For modelling:** check the 2010 count (flagged as < 10,000 rows by the quality gate) and the
# MAGIC property-type mix. A very small year or type gets little weight and gives unstable area medians.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Target distribution

# COMMAND ----------

qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
rent_q = gold.approxQuantile("monthly_rent_aed", qs, REL_ERR)
stats = gold.agg(F.mean("monthly_rent_aed").alias("mean"), F.skewness("monthly_rent_aed").alias("skew"),
                 F.skewness("log_monthly_rent").alias("skew_log"), F.min("monthly_rent_aed").alias("min"),
                 F.max("monthly_rent_aed").alias("max")).first().asDict()
summary["monthly_rent_quantiles"] = {str(q): round(v, 1) for q, v in zip(qs, rent_q)}
summary["monthly_rent_stats"] = {k: round(float(v), 3) for k, v in stats.items()}
for q, v in zip(qs, rent_q):
    print(f"p{q*100:g}: {v:,.0f} AED/month")
print(f"mean {stats['mean']:,.0f}  min {stats['min']:,.0f}  max {stats['max']:,.0f}")
print(f"skewness: monthly rent {stats['skew']:.2f}  ->  log rent {stats['skew_log']:.2f}")

p99 = rent_q[qs.index(0.99)]
raw_bins = (gold.filter(F.col("monthly_rent_aed") <= p99)
            .groupBy((F.floor(F.col("monthly_rent_aed") / 1000) * 1000).alias("b")).count().orderBy("b").toPandas())
log_lo, log_hi = gold.approxQuantile("log_monthly_rent", [0.005, 0.995], REL_ERR)
log_bins = (gold.filter(F.col("log_monthly_rent").between(log_lo, log_hi))
            .groupBy((F.floor(F.col("log_monthly_rent") / 0.1) * 0.1).alias("b")).count().orderBy("b").toPandas())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
a1.bar(raw_bins.b + 500, raw_bins["count"], width=900, color=BLUE)
a1.set_title(f"Monthly rent, skewness {stats['skew']:.2f}"); a1.set_xlabel("monthly rent, AED (up to p99)")
a1.set_ylabel("contracts"); thousands(a1); thousands(a1, "x")
a2.bar(log_bins.b + 0.05, log_bins["count"], width=0.09, color=BLUE)
a2.set_title(f"Log rent, skewness {stats['skew_log']:.2f}"); a2.set_xlabel("log(monthly rent), middle 99% of rows")
thousands(a2)
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **For modelling:** the log transform removes most of the skew, which supports the Phase-0 choice
# MAGIC of `log_monthly_rent` as the training target (MAPE is still reported on AED after back-transform).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Rent per m2 by area

# COMMAND ----------

areas = (gold.groupBy("area_name_en").agg(
            F.count("*").alias("contracts"),
            F.percentile_approx("monthly_rent_aed", 0.5, 10000).alias("median_rent"),
            F.percentile_approx("rent_per_m2", 0.5, 10000).alias("median_rent_m2"))
         .toPandas())
small = areas[areas.contracts < 100]
summary["areas_under_100_contracts"] = int(len(small))
summary["rows_in_areas_under_100"] = int(small.contracts.sum())
big = areas[areas.contracts >= 500].sort_values("median_rent_m2")
summary["median_rent_m2_range_areas_500plus"] = [round(float(big.median_rent_m2.min()), 1),
                                                  round(float(big.median_rent_m2.max()), 1)]
top, bottom = big.tail(12), big.head(12)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
a1.barh(top.area_name_en, top.median_rent_m2, color=BLUE, height=0.7)
a1.set_title("12 most expensive areas (>= 500 contracts)"); a1.set_xlabel("median AED per m2 per month")
a2.barh(bottom.area_name_en[::-1], bottom.median_rent_m2[::-1], color=BLUE, height=0.7)
a2.set_title("12 cheapest areas"); a2.set_xlabel("median AED per m2 per month")
for a in (a1, a2):
    a.grid(axis="y", visible=False); a.tick_params(axis="y", labelsize=8.5)
plt.tight_layout(); plt.show()

print(f"{n_areas} areas; {len(small)} have < 100 contracts ({int(small.contracts.sum()):,} rows, "
      f"{small.contracts.sum() / N:.2%} of data)")
print(f"median rent per m2 across areas with >= 500 contracts: "
      f"{big.median_rent_m2.min():.1f} to {big.median_rent_m2.max():.1f} AED/m2/month")

# COMMAND ----------

# MAGIC %md
# MAGIC **For modelling:** the spread between the cheapest and most expensive areas shows why area is the
# MAGIC strongest single feature. It is also exactly the number the Phase-0 baseline uses:
# MAGIC predicted rent = size (m2) x the area's median monthly rent per m2, computed from training years only.
# MAGIC Small areas give unstable medians; unseen areas fall back to the overall median (Phase 0), and the
# MAGIC model needs grouping or smoothing for them.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Change over time

# COMMAND ----------

top_types = list(by_type.property_type.head(2))
trend = (gold.filter(F.col("property_type").isin(top_types))
         .groupBy("contract_year", "property_type")
         .agg(F.percentile_approx("monthly_rent_aed", 0.5, 10000).alias("median_rent"),
              F.count("*").alias("n"))
         .orderBy("contract_year").toPandas())
overall = (gold.groupBy("contract_year").agg(F.percentile_approx("rent_per_m2", 0.5, 10000).alias("median_rent_m2"))
           .orderBy("contract_year").toPandas())
summary["median_rent_m2_by_year"] = {int(r.contract_year): round(float(r.median_rent_m2), 2) for _, r in overall.iterrows()}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for t, color in zip(top_types, [BLUE, ORANGE]):
    d = trend[trend.property_type == t]
    a1.plot(d.contract_year, d.median_rent, color=color, linewidth=2, marker="o", markersize=4, label=t)
a1.set_title("Median monthly rent by year"); a1.set_ylabel("AED per month"); thousands(a1); a1.legend()
a2.plot(overall.contract_year, overall.median_rent_m2, color=BLUE, linewidth=2, marker="o", markersize=4)
a2.axvspan(2021.5, 2023.5, color=ORANGE, alpha=0.10, linewidth=0)
a2.text(2022.5, overall.median_rent_m2.min(), "test", color=MUTED, fontsize=9, ha="center", va="bottom")
a2.set_title("Median rent per m2, all residential"); a2.set_ylabel("AED per m2 per month")
for a in (a1, a2):
    a.set_xticks(overall.contract_year[::2])
plt.tight_layout(); plt.show()
display(trend.pivot(index="contract_year", columns="property_type", values="median_rent"))

# COMMAND ----------

# MAGIC %md
# MAGIC **For modelling:** if the test years (2022-2023) sit above the training years, a model trained on
# MAGIC 2010-2021 will under-predict unless it can use time. `contract_year` stays a feature, and the README
# MAGIC should state that the test period is a level shift, not just new contracts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Relationships with the planned features

# COMMAND ----------

corr = gold.select(F.corr(F.log("size_m2"), F.col("log_monthly_rent")).alias("log_size_vs_log_rent"),
                   F.corr("size_m2", "monthly_rent_aed").alias("size_vs_rent")).first().asDict()
summary["correlations"] = {k: round(float(v), 3) for k, v in corr.items()}
print(f"Pearson correlation, all rows: log(size) vs log(rent) = {corr['log_size_vs_log_rent']:.3f}; "
      f"size vs rent = {corr['size_vs_rent']:.3f}")

subtypes = (gold.groupBy("property_sub_type")
            .agg(F.count("*").alias("contracts"), F.percentile_approx("monthly_rent_aed", 0.5, 10000).alias("median_rent"))
            .orderBy(F.desc("contracts")).toPandas())
summary["property_sub_types"] = int(len(subtypes))
reg = (gold.groupBy("reg_type").agg(F.count("*").alias("contracts"),
       F.percentile_approx("monthly_rent_aed", 0.5, 10000).alias("median_rent")).toPandas())
summary["reg_type"] = {r.reg_type: {"contracts": int(r.contracts), "median_rent": float(r.median_rent)} for _, r in reg.iterrows()}

top_sub = subtypes.head(12).sort_values("median_rent")
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.barh(top_sub.property_sub_type, top_sub.median_rent, color=BLUE, height=0.7)
ax.set_title("Median monthly rent by sub-type (12 most common)")
ax.set_xlabel("AED per month"); thousands(ax, "x"); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.show()
print(f"{len(subtypes)} distinct sub-types; the {min(12, len(subtypes))} most common cover "
      f"{subtypes.head(12).contracts.sum() / N:.1%} of rows")
display(subtypes.head(25)); display(reg)

# COMMAND ----------

# MAGIC %md
# MAGIC **For modelling:** size is strongly related to rent on the log scale, so use `log(size_m2)` or a
# MAGIC tree model. `property_sub_type` mixes bedroom counts with other labels; Stage 7 parses a numeric
# MAGIC `bedrooms` from it and keeps the rest as "other". New vs Renew differ, so `reg_type` stays.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Outliers

# COMMAND ----------

m2_q = gold.approxQuantile("rent_per_m2", [0.001, 0.999], REL_ERR)
size_q = gold.approxQuantile("size_m2", [0.001, 0.999], REL_ERR)
over_1m = gold.filter("monthly_rent_aed > 1000000")
outl = {
  "monthly_rent_over_1M": over_1m.count(),
  "rent_per_m2_below_p0.1": gold.filter(F.col("rent_per_m2") < m2_q[0]).count(),
  "rent_per_m2_above_p99.9": gold.filter(F.col("rent_per_m2") > m2_q[1]).count(),
  "size_above_p99.9": gold.filter(F.col("size_m2") > size_q[1]).count(),
}
summary["outliers"] = {**outl, "rent_per_m2_p0.1": round(m2_q[0], 2), "rent_per_m2_p99.9": round(m2_q[1], 2),
                       "size_p99.9_m2": round(size_q[1], 1)}
for k, v in outl.items():
    print(f"{k:28} {v:>8,}")
print(f"rent per m2 p0.1 = {m2_q[0]:.2f}, p99.9 = {m2_q[1]:.2f} AED/m2/month; size p99.9 = {size_q[1]:,.0f} m2")

print("\nRents over 1M AED/month by property type:")
display(over_1m.groupBy("property_type", "property_sub_type").count().orderBy(F.desc("count")))
print("10 highest rents per m2:")
display(gold.orderBy(F.desc("rent_per_m2")).select("contract_id", "contract_year", "area_name_en", "property_type",
        "property_sub_type", "size_m2", "annual_amount_aed", "monthly_rent_aed", "rent_per_m2").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC **For modelling:** these rows passed the Phase-0 rules but are implausible for one home (very high rent,
# MAGIC very high rent per m2, or a tiny size with a large rent). They are kept in gold (Phase 1 does not drop
# MAGIC data without a rule), and they are flagged for Phase 2: MAPE is sensitive to them, so the README
# MAGIC should state whether they are excluded from evaluation or capped.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. What cleaning removed (before vs after)

# COMMAND ----------

bronze = (spark.read.parquet(BRONZE_PATH)
          .withColumn("year", F.year(F.expr("try_to_timestamp(trim(contract_start_date), 'dd-MM-yyyy')"))))
before = bronze.filter(F.col("year").between(2010, 2023)).groupBy("year").count().withColumnRenamed("count", "before")
after = gold.groupBy(F.col("contract_year").alias("year")).count().withColumnRenamed("count", "after")
ret = before.join(after, "year", "left").fillna(0).orderBy("year").toPandas()
ret["kept_share"] = ret["after"] / ret["before"]
summary["kept_share_by_year"] = {int(r.year): round(float(r.kept_share), 3) for _, r in ret.iterrows()}

fig, ax = plt.subplots(figsize=(9, 3.6))
w = 0.38
ax.bar(ret.year - w / 2, ret.before, width=w, color=BLUE, label="all contracts that year (bronze)")
ax.bar(ret.year + w / 2, ret.after, width=w, color=ORANGE, label="kept (gold)")
ax.set_title("Rows kept per year after cleaning"); ax.set_xticks(ret.year); thousands(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2); plt.tight_layout(); plt.show()
display(ret)

rej = spark.read.parquet(REJECTS_PATH)
by_reason_year = (rej.withColumn("year", F.year(F.expr("try_to_timestamp(trim(contract_start_date), 'dd-MM-yyyy')")))
                  .withColumn("period", F.when(F.col("year") <= 2021, "train years").when(F.col("year") <= 2023, "test years")
                              .otherwise("outside 2010-2023"))
                  .groupBy("reject_reason").pivot("period").count().fillna(0).orderBy("reject_reason"))
display(by_reason_year)

# COMMAND ----------

# MAGIC %md
# MAGIC **For modelling:** if one period loses a much larger share than another, cleaning has changed the
# MAGIC population between train and test. Compare the kept share of 2022-2023 with 2010-2021 before trusting
# MAGIC the time split.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test-period coverage

# COMMAND ----------

train_counts = (gold.filter("split = 'train'").groupBy("area_name_en").count().withColumnRenamed("count", "train_n"))
test_counts = (gold.filter("split = 'test'").groupBy("area_name_en").count().withColumnRenamed("count", "test_n"))
cov = test_counts.join(train_counts, "area_name_en", "left").fillna(0, ["train_n"]).toPandas()
unseen = cov[cov.train_n == 0]
thin = cov[(cov.train_n > 0) & (cov.train_n < 30)]
test_n = int(cov.test_n.sum())
coverage = {
  "test_areas": int(len(cov)),
  "test_areas_unseen_in_train": int(len(unseen)), "test_rows_in_unseen_areas": int(unseen.test_n.sum()),
  "test_areas_with_under_30_train_rows": int(len(thin)), "test_rows_in_thin_areas": int(thin.test_n.sum()),
  "test_rows": test_n,
}
summary["test_coverage"] = coverage
for k, v in coverage.items():
    print(f"{k:38} {v:>8,}")
print(f"=> {(coverage['test_rows_in_unseen_areas'] + coverage['test_rows_in_thin_areas']) / test_n:.2%} "
      "of test rows have no or thin area history")
display(unseen.sort_values("test_n", ascending=False).head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC **For modelling:** test rows in areas never seen in training cannot get an area median rent per m2;
# MAGIC the Phase-0 baseline falls back to the overall training median, and the model must handle an unseen
# MAGIC area category.
# MAGIC Report this share in the README as a known risk to the hypothesis test.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary saved to the lake

# COMMAND ----------

stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
path = f"{EDA_DIR}/eda_summary_{stamp}.json"
dbutils.fs.mkdirs(EDA_DIR)
dbutils.fs.put(path, json.dumps(summary, indent=2, default=str), False)
print("saved", path)
print(json.dumps(summary, indent=2, default=str))
gold.unpersist()
