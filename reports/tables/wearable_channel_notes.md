# Heart rate, GPS and pod age — what the wearable channels are actually worth

Measured 2026-08-23 against the live database (one T1 user with CGM, 99,873
five-minute buckets, 2025-09-10 → 2026-08-23). Every number below is a **paired**
walk-forward lift: the arms share identical folds and the identical persistence
denominator, so the per-fold difference cancels fold-to-fold noise.

## What is in the database

| channel | table | rows | first row | bucket coverage |
|---|---|---|---|---|
| heart rate | `health_pulse_samples` | 65,533 | 2026-07-06 | **93 %** in its era, median 100 %/day, no empty day |
| GPS | `location_entries` | 49,710 | 2026-07-09 | **50 %** in its era, median 42 %/day, 6 empty days |
| pod | `pumps` | 1 | 2026-08-23 | one pod, activated the day of this measurement |
| basal | `basal_entries` | 1,299 | 2025-10-01 | 1.1 % of buckets |

Two corrections to what the code previously asserted:

- The July artifacts (`grid.parquet`, `feature_importance.csv`) show HR at 0 %.
  They were written **77 and 88 minutes before** `57d36c6` taught the loader to
  read `health_pulse_samples` at all, so they could not have shown anything else.
  They are not evidence about the channel.
- `bolus_entries` is **not** empty. It holds 2025-09-22 → 2026-07-04 and
  `nutrition_meals` takes over from 2026-07-19; the two share **zero** five-minute
  buckets, so reading both concatenates one continuous dose history rather than
  double-counting it.

## Heart rate: real, and the personal baseline is what carries it

Restricted to the era where the band actually reports (13,987 buckets, 10 folds):

| arm | skill @30 | skill @60 | paired lift over `no_hr` @30 |
|---|---|---|---|
| `no_hr` | 0.1996 | 0.2454 | — |
| `hr_now` (committed default) | 0.2055 | 0.2471 | +0.0059 (7/10 folds) |
| `hr_dynamics` | **0.2105** | **0.2552** | **+0.0109 (8/10 folds)** |

`hr_dynamics` over `hr_now` alone: **+0.0050 @30 in 9/10 folds** (std 0.0049) and
+0.0081 @60 in 8/10. The personal resting baseline roughly doubles what the raw
bpm delivers — the exact opposite of the synthetic result, and for a knowable
reason: every synthetic user rests at the same 65 bpm and has no circadian heart
rate, so there is no personal baseline there to recover.

## …but production training dilutes it by ~75 %

`train_user_model` fits on the user's **whole** history, where heart rate is
present in only 13 % of rows. Re-running the same comparison that way (train on
everything, test on the tail that has HR):

| arm | paired lift over `no_hr` @30 | @60 |
|---|---|---|
| `hr_now` | +0.0018 (6/10) | +0.0032 (7/10) |
| `hr_dynamics` | +0.0023 (7/10) | +0.0030 (6/10) |

Still positive, but a quarter of the size, and the two HR arms become
indistinguishable. **This is why `use_hr_dynamics` stays off**: today it buys
nothing measurable over the plain bpm under the conditions production actually
runs in.

### Bounding the training window does not fix it — it costs more than it gains

The obvious cure is to train on recent data only, so HR covers most of the rows.
Measured on identical folds, it backfires:

| arm | skill @30 | worst fold @30 | vs `full_hr` |
|---|---|---|---|
| `full_hr` | 0.2015 | 0.0540 | — |
| `win90_hr` | 0.1998 | −0.0040 | −0.0017 (std 0.0212) |
| `win60_hr` | 0.1958 | −0.0292 | −0.0057 (std 0.0273) |

A 90-day window without HR loses just as much (−0.0024), so this is not about
heart rate at all: the model simply wants the data, and there is no concept drift
worth trading it for. Both windowed arms drop a worst fold **below persistence**,
which the promotion gate would reject anyway.

### Recency weighting fails too, and worse

The other way to emphasize the HR era without discarding rows is to keep every row
and weight recent ones up. Exponential recency weights, identical folds:

| arm | skill @30 | worst fold @30 | vs `hr_now` uniform |
|---|---|---|---|
| uniform (production today) | 0.1960 | 0.0827 | — |
| half-life 120 d | 0.1615 | **−0.1451** | −0.0345 |
| half-life 60 d | 0.0884 | **−0.5787** | −0.1076 |

Far worse than truncation, and catastrophically so at 60 days. Down-weighting the
past is the same mistake as cutting it off, just applied smoothly.

**So both levers are exhausted: the dilution cures itself or not at all.** Every
further day of band wear raises HR's share of the training history and nothing
needs building. At the current rate that share reaches

| share of history with HR | date |
|---|---|
| 13.8 % (today) | 2026-08-23 |
| 25 % | 2026-10-13 |
| 30 % | 2026-11-11 |
| 40 % | 2027-01-21 |
| 50 % | 2027-05-01 |

Re-run `evaluation/channel_ablation.py` around the 25-30 % mark and flip
`use_hr_dynamics` when it wins the PRODUCTION table, not just the era table.

## GPS: built, measured, not worth switching on

| arm | skill @30 | skill @60 | vs `hr_now` @60 |
|---|---|---|---|
| `hr_now` | 0.2042 | 0.2422 | — |
| `hr_now+gps` | 0.2024 | 0.2351 | **−0.0071 in 0/5 folds** |

Consistently negative at 60 min — the clearest signal in the whole exercise, just
pointing the wrong way.

The obvious suspect was the fill: `gps_dist_*` fills a missing fix with zero, so at
50 % coverage every GPS blackout reads as *standing still* rather than *unknown*.
**Tested, and that is not it.** Carrying the position across short (≤30 min) gaps,
and additionally refusing to sum a window that is more than half blind, both make
it *worse*:

| arm | vs `no_gps` @60 |
|---|---|
| `gps_today` (fill with zero) | −0.0062 (2/8 folds) |
| `gps_ffill` (carry across short gaps) | −0.0070 (2/8) |
| `gps_blind` (+ refuse mostly-blind windows) | −0.0088 (1/8) |

So the feature engineering is not the problem. The likelier explanation is
**redundancy**: every arm above also has heart rate, which reports movement
continuously at 93 % coverage, whereas GPS reports it at 50 %. A worse-covered
second opinion on a question already answered costs variance and buys nothing.
That also predicts GPS would only ever be worth something for a user with no band
— which is not a case worth carrying six features for. Stays behind
`use_gps: false`; revisit only if a bandless user appears or the sampler's coverage
approaches the band's.

## Pod age

Unmeasurable today: `pumps` holds exactly one row, registered the same day as this
measurement. The feature is wired and tested; it needs a handful of pod changes
before it can be judged.
