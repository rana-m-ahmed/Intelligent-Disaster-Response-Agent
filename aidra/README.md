# AIDRA Core Implementation

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src\main.py --scenario A
python src\main.py --scenario B
python src\main.py --scenario C
```

## Tests

```bash
python run_tests.py
```

## Notes

- Alpha selection uses severity and a risk threshold: critical victims use alpha=0 (time only); non-critical use alpha=1 unless risk exceeds 0.7, then alpha=inf (risk-only).
- Dummy visualization prints a terminal grid after each rescue decision and saves a static route overlay to results/.
- KPIs, comparison tables, kpi_table.csv, and charts are saved to results/.
- Decision logs are written to logs/decision_log.json per scenario run.
