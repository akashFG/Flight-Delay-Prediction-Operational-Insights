1. Business Problem

Flight delays cost airlines millions in crew rescheduling, passenger compensation, gate congestion, and lost customer trust. Operations and network planning teams need to know which routes, times, and conditions carry the highest delay risk — ideally before the aircraft ever leaves the gate — so they can adjust scheduling buffers, crew rotations, and passenger communication proactively.

This project asks two questions:

Where does the delay risk concentrate (route, carrier, day, time, season)?
Can we predict, using only information known before departure, whether a flight will arrive 15+ minutes late?
2. Key Findings
Metric	Value
Overall on-time performance (arrival <15 min late)	75.9%
Overall cancellation rate	3.8%
Worst-performing carrier (delay rate)	B6 — 29.0%
Worst day of week	Tuesday
Worst month	January (winter weather effect)

Delay cause breakdown (share of total delay minutes):

Late-arriving aircraft (knock-on effect) and carrier-caused delays dominate — meaning delays compound through the day: a late aircraft in the morning cascades into afternoon/evening delays on the same tail number.
Weather is a meaningful but not the largest single cause — operational/scheduling inefficiency is the bigger lever.

Business implication: Because late-aircraft delay is the top driver, the highest-leverage intervention isn't weather forecasting — it's schedule buffer design on high-frequency routes and minimum turnaround time policy, especially at hub airports during morning peak hours.

📊 See outputs/01 through outputs/06 for full charts (by airline, day of week, time block, route, cause, and monthly trend).

3. Predictive Model

Target: ArrDel15 — will the flight arrive 15+ minutes late? (binary classification) Features used (deliberately limited to pre-departure information only, to avoid data leakage): carrier, origin, destination, scheduled departure hour, distance, month, day of week, quarter.

Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	55.2%	0.281	0.550	0.372	0.568
Random Forest	60.7%	0.325	0.586	0.418	0.640
XGBoost (best)	61.5%	0.337	0.618	0.436	0.660

Honest interpretation: A naive "always predict on-time" baseline already scores 75.9% accuracy — which is why accuracy alone is misleading for this imbalanced problem. The model is evaluated on ROC-AUC and recall instead, since the operational cost of missing a real delay (false negative) is higher than a false alarm. XGBoost recovers 62% of true delays while using only information available at booking/scheduling time — no weather feed, no live ATC data, no upstream aircraft status. This ceiling is expected and consistent with published airline OTP research: delay is driven heavily by same-day cascading and weather, which are not available until much closer to departure. Feeding in live weather and upstream tail-number status (as real airline ops systems do) is the clear next step to push recall higher.

4. What I'd Do With Real SIA / Aviation Operations Data

This project deliberately used public data to demonstrate the methodology end-to-end. Applied to an airline's internal systems, the same pipeline would extend with:

Upstream aircraft rotation data (tail number's prior leg status) — the single strongest predictor in real airline delay models.
Live weather feeds at origin/destination (METAR/TAF).
Crew scheduling and minimum connect time violations.
Hub congestion signals (e.g., Changi slot utilization at peak banks).
A cost-weighted evaluation metric instead of plain F1 — a missed delay on a long-haul widebody costs more than one on a short regional hop.
5. Project Structure
flight-delay-project/
├── data/                      # cleaned dataset (parquet)
├── src/
│   ├── 01_eda.py               # exploratory analysis + charts
│   └── 02_modeling.py          # feature engineering + model training/evaluation
├── outputs/                    # all charts, model results, dashboard-ready CSV
└── README.md
6. Tech Stack

Python · pandas · scikit-learn · XGBoost · matplotlib/seaborn · (Power BI-ready CSV export included for dashboarding)

7. How to Run
bash
pip install -r requirements.txt
cd src
python 01_eda.py        # generates EDA charts in ../outputs/
python 02_modeling.py   # trains models, generates evaluation charts

Author: Akash Mojumder — Data Scientist / Data Analyst
