import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from io import BytesIO
import re

def extract_number(x):
    if isinstance(x, str):
        match = re.search(r"\d+\.?\d*", x)
        if match:
            return float(match.group())
        return None
    return x
# ---------------- PAGE CONFIG (DARK MODE READY) ----------------
st.set_page_config(
    page_title="Advanced Sports & Finance Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- DARK MODE STYLE ----------------
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Advanced ML Dashboard")
st.markdown("⚽ Champions League & 💎 Precious Metals Analysis")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_cl():
    return pd.read_csv("champions_league_matches.csv")

@st.cache_data
def load_metals():
    return pd.read_csv("precious_metals_historical_data.csv")

cl_df = load_cl()
metals_df = load_metals()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Sahifalar")
page = st.sidebar.radio(
    "Bo‘lim tanlang:",
    [
        "⚽ Champions League – Dataset",
        "🤖 Champions League – ML Model",
        "💎 Precious Metals – Dataset",
        "🔮 Precious Metals – Bashorat"
    ]
)

# ======================================================
# ⚽ CHAMPIONS LEAGUE DATASET
# ======================================================
if page == "⚽ Champions League – Dataset":
    st.subheader("⚽ Champions League Dataset")

    st.dataframe(cl_df, use_container_width=True)

    # Download
    st.download_button(
        "📥 CSV yuklab olish",
        cl_df.to_csv(index=False),
        "champions_league.csv",
        "text/csv"
    )

    fig, ax = plt.subplots()
    cl_df["result"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Match Results Distribution")
    st.pyplot(fig)

# ======================================================
# 🤖 CHAMPIONS LEAGUE ML MODEL
# ======================================================
elif page == "🤖 Champions League – ML Model":
    st.subheader("🤖 Match Result Prediction (Random Forest)")

    features = [
    "home_shots_on_target",
    "away_shots_on_target",
    "home_possession",
    "away_possession"
    ]

    df = cl_df.dropna(subset=features + ["result"])
    X = df[features]
    y = df["result"]

    # 🔥 STRING → NUMBER
    # def extract_first_number(x):
    #     if isinstance(x, str):
    #         return float(x.split(" ")[0])
    #     return x

    for col in X.columns:
        X[col] = X[col].apply(extract_number)

    le = LabelEncoder()
    y = le.fit_transform(y)

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    col1, col2, col3 = st.columns(3)
    acc = accuracy_score(y_test, model.predict(X_test))
    col1.metric("🎯 Accuracy", f"{acc:.2f}")
    col2.metric("📊 Train size", X_train.shape[0])
    col3.metric("🧪 Test size", X_test.shape[0])
    st.markdown("### 📌 Feature Importance")

    importances = model.feature_importances_
    fig, ax = plt.subplots()

    ax.barh(features, importances)
    ax.set_xlabel("Importance")
    ax.set_title("Model Feature Importance")

    st.pyplot(fig)

# ======================================================
# 💎 PRECIOUS METALS DATASET
# ======================================================
elif page == "💎 Precious Metals – Dataset":
    st.subheader("💎 Precious Metals Dataset")

    st.dataframe(metals_df, use_container_width=True)

    excel_buffer = BytesIO()
    metals_df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)

    st.download_button(
        label="📥 Excel yuklab olish",
        data=excel_buffer,
        file_name="precious_metals.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

    date_col = metals_df.columns[0]
    metal_cols = metals_df.columns[1:]

    fig, ax = plt.subplots()
    ax.plot(metals_df[date_col], metals_df[metal_cols[0]])
    ax.set_title(f"{metal_cols[0]} Price Over Time")
    st.pyplot(fig)

# ======================================================
# 🔮 PRECIOUS METALS PRICE PREDICTION
# ======================================================
elif page == "🔮 Precious Metals – Bashorat":
    st.subheader("🔮 Precious Metal Price Prediction")

    date_col = metals_df.columns[0]
    metal_cols = metals_df.columns[1:]

    metal = st.selectbox("Metall tanlang", metal_cols)

    df = metals_df[[metal]].dropna()
    df["index"] = np.arange(len(df))

    X = df[["index"]]
    y = df[metal]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    future_days = st.slider("Necha kun oldinga bashorat qilinsin?", 1, 30, 7)
    future_index = np.arange(len(df), len(df) + future_days).reshape(-1, 1)

    preds = model.predict(future_index)

    fig, ax = plt.subplots()
    ax.plot(df["index"], y, label="History")
    ax.plot(future_index, preds, label="Prediction")
    ax.legend()
    ax.set_title(f"{metal} Price Forecast")
    st.pyplot(fig)
    st.markdown("### 📈 Historical vs Forecasted Prices")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        df.index,
        y,
        label="Historical",
        color="#1f77b4",
        linewidth=2
    )

    ax.plot(
        future_index.flatten(),
        preds,
        label="Forecast",
        linestyle="--",
        color="#ff7f0e",
        linewidth=2
    )

    ax.axvline(
        x=df.index.max(),
        linestyle=":",
        color="gray",
        label="Forecast Start"
    )

    ax.set_title(f"{metal} Price Prediction")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.success("✅ Bashorat muvaffaqiyatli yaratildi")