
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# STEP 1: Generate / Load Data
# ============================================================

def generate_sample_data(n_days: int = 730) -> pd.DataFrame:
    """
    Generates realistic synthetic weather data for demonstration.
    Replace this with:
        df = pd.read_csv('data/weather.csv')
    when using real data.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')

    # Simulate seasonal temperature (°C) with noise
    day_of_year = np.arange(n_days)
    temp = (
        20                                          # baseline
        + 12 * np.sin(2 * np.pi * day_of_year / 365 - np.pi / 2)  # seasonal cycle
        + np.random.normal(0, 2.5, n_days)          # daily noise
    )

    # Simulate rainfall (mm) — skewed distribution, more in monsoon
    rainfall_base = 3 + 8 * np.sin(np.pi * day_of_year / 365) ** 2
    rainfall = np.random.exponential(scale=rainfall_base)
    rainfall = np.clip(rainfall, 0, 120)            # cap extreme values

    # Introduce some missing values to practice data cleaning
    temp[np.random.choice(n_days, 30, replace=False)] = np.nan
    rainfall[np.random.choice(n_days, 20, replace=False)] = np.nan

    df = pd.DataFrame({
        'date': dates,
        'temperature_C': np.round(temp, 2),
        'rainfall_mm': np.round(rainfall, 2),
        'humidity_pct': np.round(np.random.uniform(40, 95, n_days), 1),
        'wind_speed_kmh': np.round(np.abs(np.random.normal(15, 7, n_days)), 1),
    })
    return df


# ============================================================
# STEP 2: Data Cleaning
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning:
    - Parses dates
    - Reports and fills missing values
    - Removes obvious outliers
    - Adds derived time columns
    """
    print("=" * 50)
    print("🧹 DATA CLEANING REPORT")
    print("=" * 50)

    # ── 2a. Parse dates ───────────────────────────────────────
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # ── 2b. Missing value report ──────────────────────────────
    print(f"\nShape before cleaning : {df.shape}")
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    # Fill temperature with 7-day rolling mean (forward + backward)
    df['temperature_C'] = (
        df['temperature_C']
        .fillna(df['temperature_C'].rolling(7, min_periods=1, center=True).mean())
    )

    # Fill rainfall with 0 (missing rain records ≈ no rain)
    df['rainfall_mm'] = df['rainfall_mm'].fillna(0)

    # Fill remaining numerics with column median
    for col in ['humidity_pct', 'wind_speed_kmh']:
        df[col] = df[col].fillna(df[col].median())

    print(f"\nMissing values after cleaning: {df.isnull().sum().sum()}")

    # ── 2c. Outlier removal (IQR method) ─────────────────────
    for col in ['temperature_C', 'rainfall_mm']:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        before = len(df)
        df = df[df[col].between(Q1 - 3 * IQR, Q3 + 3 * IQR)]
        print(f"Outliers removed in '{col}': {before - len(df)}")

    # ── 2d. Derived columns ───────────────────────────────────
    df['year']    = df['date'].dt.year
    df['month']   = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%b')
    df['season']  = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring',  4: 'Spring', 5: 'Spring',
        6: 'Summer',  7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
    })

    print(f"\nShape after  cleaning : {df.shape}")
    print("=" * 50)
    return df.reset_index(drop=True)


# ============================================================
# STEP 3: Exploratory Analysis
# ============================================================

def explore_data(df: pd.DataFrame) -> None:
    """Prints key summary statistics."""
    print("\n📊 SUMMARY STATISTICS")
    print(df[['temperature_C', 'rainfall_mm', 'humidity_pct', 'wind_speed_kmh']].describe().round(2))

    print("\n🌡️  Monthly Average Temperature (°C):")
    print(df.groupby('month_name')['temperature_C'].mean().round(2).to_string())

    print("\n🌧️  Monthly Total Rainfall (mm):")
    print(df.groupby('month_name')['rainfall_mm'].sum().round(1).to_string())


# ============================================================
# STEP 4: Visualizations
# ============================================================

def plot_all(df: pd.DataFrame, save_path: str = "outputs/") -> None:
    """Creates and saves all visualisation charts."""

    monthly = (
        df.groupby(['year', 'month', 'month_name'])
        .agg(avg_temp=('temperature_C', 'mean'),
             total_rain=('rainfall_mm', 'sum'),
             avg_humidity=('humidity_pct', 'mean'))
        .reset_index()
        .sort_values(['year', 'month'])
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('🌦️  Weather Data Analysis Dashboard', fontsize=18, fontweight='bold', y=1.02)

    # ── Plot 1: Temperature trend over time ───────────────────
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['temperature_C'], color='#f97316', alpha=0.35, linewidth=0.6, label='Daily')
    ax1.plot(df['date'],
             df['temperature_C'].rolling(30, center=True).mean(),
             color='#c2410c', linewidth=2, label='30-day rolling avg')
    ax1.set_title('Temperature Trend Over Time', fontweight='bold')
    ax1.set_ylabel('Temperature (°C)')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.legend(fontsize=9)
    ax1.tick_params(axis='x', rotation=30)

    # ── Plot 2: Monthly rainfall bar chart ────────────────────
    ax2 = axes[0, 1]
    month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    rain_monthly = df.groupby('month_name')['rainfall_mm'].sum().reindex(month_order)
    bars = ax2.bar(rain_monthly.index, rain_monthly.values,
                   color=sns.color_palette("Blues_d", 12), edgecolor='white')
    ax2.set_title('Total Monthly Rainfall', fontweight='bold')
    ax2.set_ylabel('Rainfall (mm)')
    ax2.tick_params(axis='x', rotation=45)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=7.5)

    # ── Plot 3: Seasonal box plot ─────────────────────────────
    ax3 = axes[1, 0]
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    season_palette = {'Winter': '#93c5fd', 'Spring': '#86efac',
                      'Summer': '#fde68a', 'Autumn': '#fdba74'}
    sns.boxplot(data=df, x='season', y='temperature_C', order=season_order,
                palette=season_palette, ax=ax3, linewidth=0.8)
    ax3.set_title('Temperature Distribution by Season', fontweight='bold')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_xlabel('')

    # ── Plot 4: Correlation heatmap ───────────────────────────
    ax4 = axes[1, 1]
    corr_cols = ['temperature_C', 'rainfall_mm', 'humidity_pct', 'wind_speed_kmh']
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                mask=mask, ax=ax4, linewidths=0.5,
                annot_kws={"size": 10}, vmin=-1, vmax=1)
    ax4.set_title('Feature Correlation Heatmap', fontweight='bold')

    plt.tight_layout()
    path = f"{save_path}weather_dashboard.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"\n✅ Dashboard saved → {path}")
    plt.show()

    # ── Extra: Yearly comparison ──────────────────────────────
    fig2, ax = plt.subplots(figsize=(12, 5))
    for year, grp in df.groupby('year'):
        grp_monthly = grp.groupby('month')['temperature_C'].mean()
        ax.plot(grp_monthly.index, grp_monthly.values,
                marker='o', markersize=4, label=str(year), linewidth=1.8)
    ax.set_title('Year-over-Year Monthly Temperature Comparison', fontweight='bold')
    ax.set_ylabel('Avg Temperature (°C)')
    ax.set_xlabel('Month')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.legend(title='Year')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path2 = f"{save_path}yearly_comparison.png"
    plt.savefig(path2, bbox_inches='tight')
    print(f"✅ Yearly comparison saved → {path2}")
    plt.show()


# ============================================================
# STEP 5: Trend Summary
# ============================================================

def print_trend_summary(df: pd.DataFrame) -> None:
    """Prints a plain-English summary of key findings."""
    print("\n" + "=" * 50)
    print("📝 KEY FINDINGS")
    print("=" * 50)
    hottest  = df.loc[df['temperature_C'].idxmax()]
    coldest  = df.loc[df['temperature_C'].idxmin()]
    rainiest = df.loc[df['rainfall_mm'].idxmax()]

    print(f"🔥 Hottest day  : {hottest['date'].date()} — {hottest['temperature_C']}°C")
    print(f"❄️  Coldest day  : {coldest['date'].date()} — {coldest['temperature_C']}°C")
    print(f"🌧️  Rainiest day : {rainiest['date'].date()} — {rainiest['rainfall_mm']} mm")

    yearly_temp = df.groupby('year')['temperature_C'].mean()
    if len(yearly_temp) > 1:
        delta = yearly_temp.iloc[-1] - yearly_temp.iloc[0]
        direction = "📈 warming" if delta > 0 else "📉 cooling"
        print(f"\n🌍 Overall trend: {direction} by {abs(delta):.2f}°C over the dataset period")

    print("=" * 50)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # 1. Load data (using synthetic data here — swap with your CSV)
    print("📂 Loading data...")
    df_raw = generate_sample_data(n_days=730)
    df_raw.to_csv('data/raw_weather.csv', index=False)
    print(f"   Raw data saved to data/raw_weather.csv  ({len(df_raw)} rows)")

    # 2. Clean
    df_clean = clean_data(df_raw)
    df_clean.to_csv('data/cleaned_weather.csv', index=False)
    print("   Cleaned data saved to data/cleaned_weather.csv")

    # 3. Explore
    explore_data(df_clean)

    # 4. Visualise
    plot_all(df_clean, save_path='outputs/')

    # 5. Summary
    print_trend_summary(df_clean)
