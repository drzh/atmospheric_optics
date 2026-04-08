#  Atmospheric Optics Predictor

Predict atmospheric optical phenomena (halo, sundogs, circumzenithal arc, rainbow) using weather and satellite data.

---

## Features

- Automatic weather data ingestion (NOAA GFS)
- Cirrus cloud detection
- Solar position calculation
- Probability prediction:
  - Sundogs (Parhelia)
  - 22° Halo
  - Circumzenithal Arc (CZA)
  - Rainbow

---

## 🏗️ Architecture

data_ingestion → feature_engineering → models → predictor → API/CLI

---

## Quick Start

```bash
pip install -r requirements.txt
python cli/main.py --lat 32.8 --lon -96.8