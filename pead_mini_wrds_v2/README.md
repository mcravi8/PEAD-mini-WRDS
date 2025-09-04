# PEAD Mini Backtester

A minimal Python backtester for **Post-Earnings Announcement Drift (PEAD)**.  
It sorts stocks into deciles by **Standardized Unexpected Earnings (SUE)** and builds a **long–short strategy** (long top decile, short bottom decile) around earnings announcements.

---

## 🚀 Features
- **Decile Sorting:** Splits stocks into deciles by earnings surprise (SUE).
- **Portfolio Simulation:** Long top decile (D10), short bottom decile (D1).
- **Holding Periods:** Flexible (e.g., 5, 10, 20 days).
- **Sector Neutral Option:** Equal-weight by sector, or plain equal-weight.
- **Performance Metrics:** Information Coefficient (IC), Alpha, Sharpe, Beta.
- **Charts:**  
  - Decile cumulative returns  
  - Long–short cumulative return  

---

## 📂 Project Structure
pead-mini-backtester/
│
├── pead/ # Core backtester code
│ ├── backtest/core.py
│ ├── cli/run.py
│ ├── data/demo.py
│ └── signals/...
│
├── sample_data/ # Demo datasets
│ ├── demo_events.csv
│ └── demo_returns.csv
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore

## ⚡ Quickstart

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
Run demo backtest

bash
Copy code
python -m pead.cli.run --mode demo --hold_days 20 --sector_neutral
Check results

Plots are saved in outputs/

Metrics are saved in outputs/pead_summary.json
