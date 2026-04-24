# 🌦️ Weather Data Analysis

A beginner-friendly Python data science project that analyzes historical weather data to uncover trends in **temperature** and **rainfall** using data cleaning and visualization techniques.

---

## 📁 Project Structure

```
weather_analysis/
├── data/
│   ├── raw_weather.csv        # Raw (with missing values)
│   └── cleaned_weather.csv    # After cleaning
├── outputs/
│   ├── weather_dashboard.png  # 4-panel visualization
│   └── yearly_comparison.png  # Year-over-year chart
├── weather_analysis.py        # Main analysis script
├── requirements.txt
└── README.md
```

---

## 🔍 Concepts Covered

| Concept | What's Done |
|---|---|
| **Data Cleaning** | Handle missing values, remove outliers (IQR), parse dates |
| **Feature Engineering** | Extract month, year, season from date |
| **Data Visualization** | Line charts, bar charts, box plots, heatmaps |
| **Trend Analysis** | Rolling averages, year-over-year comparison |

---

## 📊 Visualizations

- **Temperature trend** with 30-day rolling average
- **Monthly rainfall** bar chart
- **Seasonal temperature** box plot
- **Correlation heatmap** between all weather variables
- **Year-over-year** monthly temperature comparison

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/weather-analysis.git
cd weather-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python weather_analysis.py
```

---

## 📌 Using Your Own Data

Replace the `generate_sample_data()` call in `main` with:

```python
df_raw = pd.read_csv('data/your_weather_file.csv')
```

Make sure your CSV has at minimum:
- `date` column (YYYY-MM-DD format)
- `temperature_C` column
- `rainfall_mm` column

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib** — plotting
- **Seaborn** — statistical visualization

---

## 👤 Author

**[Your Name]**  
[GitHub](https://github.com/your-username) • [LinkedIn](https://linkedin.com/in/your-profile)
