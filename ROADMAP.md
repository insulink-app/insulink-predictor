# ROADMAP — Glucose Forecasting Engine

> **Für Claude Code:** Dieses Dokument ist der verbindliche Arbeitsplan. Arbeite **eine Phase pro Session** ab, von oben nach unten. Springe **nicht** vor. Jede Phase hat eine *Definition of Done (DoD)* — sie ist erst abgeschlossen, wenn alle DoD-Punkte grün sind und die Tests laufen. Committe pro Phase. Die Guardrails unten sind nicht verhandelbar.

---

## 0. Kernthese (warum wir bauen, was wir bauen)

1. **Wir sagen nicht „den Blutzucker" vorher — wir schlagen Persistence.** Im Ruhezustand gewinnt „flach bleibt flach". Der gesamte Wert liegt in den **Exkursionen** (Anstieg nach dem Essen, Abfall bei Bewegung). Jedes Modell wird gegen die Persistence-Baseline gemessen; schlägt es sie nicht, ist es wertlos.
2. **Event-getriggert statt Dauer-Flatline.** Vorhersagen sind rund um Events (Mahlzeit, Aktivität, Glukose-Anstieg) am wertvollsten. Dort wird scharf modelliert.
3. **Data-Flywheel ist das Asset, nicht das Modell.** Labels kommen gratis (der zukünftige Wert wird ohnehin beobachtet). Jeder Nutzer trainiert das System passiv.

### Angenommener Wedge (VOR Phase 2 bestätigen — siehe §7)
**Non-Insulin / Lifestyle** (T2 / Prädiabetes / Wellness). Konsequenzen:
- Trägere Dynamik, besser aus Kontext (Essen, Bewegung, Tageszeit) vorhersagbar.
- Regulatorisch **Wellness / „Insights & Muster"**, nicht MDR-Medizinprodukt. Sprache im Produkt konsequent „Hinweise/Muster", nie „Vorhersage für Dosierung".
- Falls doch Insulin/Carbs vorhanden → Feature-Set und Regulatorik ändern sich (§7). Bis zur Bestätigung bauen wir Insulin/Carbs als **optionale** Features ein, die sauber degradieren, wenn sie fehlen.

---

## 1. Tech-Stack (verbindlich)

| Zweck | Wahl | Grund |
|---|---|---|
| Env / Packaging | **uv** | schnell, reproduzierbar |
| Sprache | Python 3.11+ | |
| DataFrames | **pandas** (v1), Polars optional später | Ökosystem |
| Schema-Validierung | **pandera** | Data Contract erzwingen |
| Config | **pydantic-settings** + YAML | typisiert, keine Magie |
| Baseline-Modell | **LightGBM** | schlägt Persistence, frisst Mixed-Features, winzig, on-device-fähig |
| Eval / Utils | scikit-learn | |
| Error Grid | **methcomp** (Parkes/Clarke) | klinische Metrik |
| Experiment-Tracking | **MLflow** (self-hosted, passt in Homelab) | |
| CLI | **typer** | ein Entrypoint pro Pipeline-Schritt |
| Tests | **pytest** | |

**Kein Deep Learning in Phase 0–4.** Erst wenn LightGBM auf echten Per-User-Daten plateaut (§ Phase 5).

---

## 2. Repo-Struktur (Phase 0 legt das an)

```
glucose-forecast/
├── README.md
├── ROADMAP.md                 # dieses Dokument
├── pyproject.toml             # uv
├── config/
│   └── config.yaml            # Horizonte, Grid-Auflösung, Feature-Flags
├── data/                      # gitignored (raw/interim/processed)
├── src/glucose_forecast/
│   ├── config.py              # pydantic Settings
│   ├── data/
│   │   ├── schema.py          # pandera Data Contract
│   │   ├── synth.py           # synthetischer Generator (baut Pipeline VOR echten Daten)
│   │   ├── load.py            # echter Loader (später)
│   │   └── align.py           # Resampling auf reguläres 5-min-Grid pro User
│   ├── features/build.py      # strikt kausaler Feature-Builder
│   ├── models/
│   │   ├── baseline.py        # Persistence
│   │   ├── lgbm.py            # direct multi-horizon
│   │   └── personalize.py     # global+user-features, dann Residual-Stacking
│   ├── eval/
│   │   ├── split.py           # chronologisch (nie random!)
│   │   ├── metrics.py         # RMSE/MAE/Skill-Score
│   │   └── error_grid.py      # Parkes/Clarke
│   └── cli.py                 # typer: gf synth|features|train-*|eval
├── notebooks/                 # Exploration, nicht produktiv
├── reports/                   # Eval-Outputs
└── tests/
```

---

## 3. Data Contract (Phase 0 definiert das zuerst)

Reguläres **5-min-Grid pro User** nach Alignment. Rohdaten sind irregulär → `align.py` resampled/alignt alle Quellen auf dieses Grid.

| Feld | Typ | Pflicht | Notiz |
|---|---|---|---|
| `user_id` | str | ✔ | |
| `ts_utc` | datetime (UTC) | ✔ | Grid-Anker |
| `ts_local` | datetime | ✔ | für circadiane Features nötig |
| `glucose_mgdl` | float | ✔ | **Einheit fixieren** (mg/dL). mmol/L → ×18 |
| `meal_flag` | bool | ✔ | Mahlzeit in diesem Bucket geloggt |
| `carbs_g` | float | optional | falls vorhanden → COB-Proxy |
| `insulin_u` | float | optional | falls vorhanden → IOB-Proxy + Regulatorik-Flag |
| `steps` | int | ✔ | pro Bucket |
| `activity_flag` | bool | optional | Workout aktiv |
| `hr` | float | optional | |
| `weather_temp` | float | optional | vermutlich niedriger Wert — messen, nicht glauben |
| `sensor_gap` | bool | abgeleitet | markiert CGM-Ausfall, **nicht über große Lücken interpolieren** |

**Einheiten- und Intervall-Frage in `config.yaml` explizit machen** (`glucose_unit`, `grid_minutes`). Kein Hardcoding im Code.

---

## 4. Ziel-Definition

Vorhersage `glucose_mgdl` bei `t + h`, für `h ∈ {30, 60}` min (= 6 und 12 Steps bei 5-min-Grid). Beides parallel, **getrennt evaluiert**.

- **Direct multi-horizon** (ein Modell pro Horizont). Verbindlich für v1.
- **Recursive verboten** in v1 (Fehlerakkumulation).
- Kurven-Forecast (Phase 3): alle Steps `t+5 … t+60` via multi-output.

---

## 5. Phasen

### Phase 0 — Scaffold & Data Contract
**Ziel:** Lauffähiges Skelett + synthetische Daten, damit die gesamte Pipeline steht, bevor echte Daten da sind.

Tasks:
- uv-Projekt, Repo-Struktur (§2), `config.yaml`, `config.py`.
- `schema.py`: pandera-Schema aus §3.
- `synth.py`: plausibler Generator — Basalniveau pro User + zufällige Mahlzeiten (Anstieg + Decay), Aktivitäts-Dips, circadianer Drift, Sensor-Gaps, Rauschen. Mehrere User mit **unterschiedlicher Dynamik**.
- `align.py`: irreguläre Rohdaten → reguläres 5-min-Grid; Gaps als `sensor_gap` markieren.
- `cli.py`: `gf synth`.

**DoD:**
- `gf synth` erzeugt DataFrame, der das pandera-Schema besteht.
- Alignment liefert nachweislich reguläres Grid (Test: keine Lücken außer markierte).
- `pytest` grün.

---

### Phase 1 — Eval-Harness & Persistence  ⚠️ ZUERST, VOR JEDEM MODELL
**Ziel:** Der Nordstern. Alles wird hieran gemessen.

Tasks:
- `split.py`: **chronologischer** Split pro User (früh = train, spät = test). Zusätzlich Option „held-out Users" für Cold-Start-Test (Phase 4).
- `metrics.py`: RMSE, MAE pro Horizont **plus Skill-Score** `= 1 − RMSE_model / RMSE_persistence`.
- `error_grid.py`: Parkes (primär) + Clarke via `methcomp`; % in Zonen A/B vs. C/D/E.
- `baseline.py`: Persistence (`ŷ_{t+h} = g_t`).
- Report nach `reports/` (Tabelle pro Horizont + Error-Grid-Plot), MLflow-Logging.

**DoD:**
- Eval-Report für Persistence liegt vor: RMSE/MAE @30 & @60 + Parkes-Zonen.
- Skill-Score-Funktion getestet (Persistence gegen sich selbst = 0).
- **Kein random split irgendwo im Code.**

---

### Phase 2 — Features & LightGBM v1
**Ziel:** Erstes echtes Produkt. Muss Persistence schlagen.

Tasks — `features/build.py`, **strikt kausal** (nie Zukunftsinfo):
- Glukose-Lags: `t, t-5, t-10, t-15, t-30, t-45, t-60`.
- Änderungsraten (kurz/lang) + Beschleunigung (2. Ableitung).
- Rolling mean/std/min/max über 30/60 min.
- `time_since_meal`, `time_since_activity`.
- Circadian: `hour` als sin/cos, `is_weekend`.
- Aktivität: Steps in 15/30/60 min, Workout-Flag.
- *Optional (falls vorhanden):* COB-Proxy aus `carbs_g` (exp. Decay), IOB-Proxy aus `insulin_u`.

Modell — `models/lgbm.py`: direct multi-horizon (ein LGBM je Horizont), Feature-Importances loggen.

**DoD:**
- **Skill-Score > 0 @30 und @60** vs. Persistence.
- **Leakage-Test** grün: kein Feature nutzt Werte aus `> t`. (Unit-Test, der bei künstlich eingebautem Future-Leak fehlschlägt.)
- Feature-Importances in `reports/` — Erkenntnis dokumentiert, welche Kontext-Features tatsächlich tragen (Erwartung: Tageszeit & time-since-meal hoch, Wetter niedrig).

---

### Phase 3 — Event-getriggerte Trajektorien-Prognose
**Ziel:** Scharf sein, wo Persistence versagt.

Tasks:
- Event-Detection: Mahlzeit geloggt / Glukose-Anstiegsrate über Schwelle / Aktivitäts-Peak.
- Multi-output-Forecast: komplette Kurve `t+5 … t+60` bei Event-Trigger.
- Eval **speziell auf Post-Event-Fenstern** (nicht nur global gemittelt).

**DoD:**
- Auf Post-Meal-Fenstern schlägt das Modell Persistence mit **größerem** Skill-Score als im globalen Schnitt.
- Kurven-Output (nächste 60 min) wird erzeugt und geplottet.

---

### Phase 4 — Personalisierung (der Moat + die Retention-Story)
**Ziel:** Wird spürbar besser, je länger man die App nutzt.

- **4a — Global konditioniert:** ein globales Modell mit User-Level-Features (mittlerer Glukosewert, Variabilität, typische Post-Meal-Reaktion). Ein Modell, kein Cold-Start.
- **4b — Residual-Stacking:** globales Basismodell + winziges Per-User-Residualmodell (lernt nur das Delta). Residuum ist beim neuen User ≈ 0 → gradueller, automatischer Übergang generisch → persönlich. Trigger: ~2–4 Wochen User-Daten.

**DoD:**
- Personalisiert schlägt rein-global im **Per-User**-Eval.
- Cold-Start-Test auf held-out Users (Phase 1): degradiert graceful, kein Absturz.

---

### Phase 5 — Später (JETZT NICHT bauen)
Nur wenn LightGBM auf echten Per-User-Daten plateaut:
- Sequenzmodelle (TCN / kleiner GRU) für längere Horizonte.
- On-device: LGBM → ONNX, Latenz/Privacy/Offline.
- Retraining-Pipeline (nächtlicher Batch global, Residuen häufiger) + Drift-Monitoring.

---

## 6. Guardrails (nicht verhandelbar)

- **Nie random split.** Autokorrelation → Leakage. Immer chronologisch.
- **Strikte Kausalität** in allen Features. Leakage-Test muss existieren und grün sein.
- **Persistence-Gate:** Keine Phase gilt als „fertig", wenn ihr Modell Persistence nicht pro Horizont schlägt.
- **Kein Deep Learning vor Phase 5.**
- **Große CGM-Lücken nicht überinterpolieren** — markieren, aus Prädiktion ausschließen.
- **Rote Error-Grid-Zone = Incident**, kein Datenpunkt.
- **Produktsprache:** „Muster/Hinweise", nie „Dosierungsvorhersage" (solange Wellness-Wedge).

---

## 7. Offene Entscheidungen — VOR Phase 2 klären

1. **Insulin/Carbs in den Daten?** Ja → COB/IOB-Features (meist stärkste Prädiktoren) + Regulatorik-Review. Nein → Kontext-only, Wellness-Positionierung bestätigt.
2. **Population:** T1 / T2 / Wellness? Bestimmt Dynamik + Eval-Latte + Recht.
3. **Einheit:** mg/dL vs. mmol/L → in `config.yaml` fixieren.
4. **CGM-Intervall:** 5 min (Standard) vs. anderes → `grid_minutes`.

Diese vier ändern Feature-Set und regulatorische Linie mehr als jede Modellwahl.