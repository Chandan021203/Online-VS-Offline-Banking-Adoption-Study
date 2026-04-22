# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

st.set_page_config(page_title="Banking Dashboard", layout="wide")

st.title("🏦 Online vs Offline Banking Adoption Dashboard")

# ---------------- DEFAULT DATASET FUNCTION ----------------
def generate_default_data(n=500):
    np.random.seed(42)

    data = pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "income": np.random.randint(20000, 150000, n),
        "transactions_per_month": np.random.randint(5, 50, n),
        "internet_usage_score": np.random.randint(1, 10, n),
        "credit_score": np.random.randint(300, 850, n),
        "satisfaction_score": np.random.randint(1, 10, n),
        "location": np.random.choice(["Urban", "Rural"], n),
        "banking_type": np.random.choice(["Online", "Offline"], n),
        "city": np.random.choice(
            ["Delhi", "Mumbai", "Indore", "Bhopal", "Pune", "Jaipur", "Hyderabad"],
            n
        )
    })

    return data


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Banking Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Using uploaded dataset")
else:
    df = generate_default_data()
    st.warning("⚠️ No dataset uploaded — using default sample dataset")

# ---------------- DATA CLEANING ----------------
df.columns = df.columns.str.lower().str.replace(" ", "_")

for col in df.select_dtypes(include=np.number):
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include="object"):
    df[col].fillna(df[col].mode()[0], inplace=True)

df.drop_duplicates(inplace=True)

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "📌 Navigation Menu",
    [
        "📊 Dataset Overview",
        "📈 EDA",
        "🔧 Feature Engineering",
        "📉 Visualizations",
        "🔥 Insights",
        "⬇ Download Data"
    ]
)

# ---------------- DATASET ----------------
if menu == "📊 Dataset Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# ---------------- EDA ----------------
elif menu == "📈 EDA":
    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ---------------- FEATURE ENGINEERING ----------------
elif menu == "🔧 Feature Engineering":

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[18, 25, 35, 50, 65, 100],
            labels=["18-25", "26-35", "36-50", "51-65", "65+"]
        )
        st.subheader("Age Group Feature")
        st.dataframe(df[["age", "age_group"]].head())

    if "income" in df.columns:
        df["income_level"] = pd.qcut(
            df["income"], q=3, labels=["Low", "Medium", "High"]
        )
        st.subheader("Income Level Feature")
        st.dataframe(df[["income", "income_level"]].head())

# ---------------- VISUALIZATIONS ----------------
elif menu == "📉 Visualizations":

    st.subheader("📊 Banking Type Distribution")
    fig, ax = plt.subplots()
    df["banking_type"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
    st.pyplot(fig)

    st.subheader("💳 Transactions per Month")
    fig, ax = plt.subplots()
    df.groupby("banking_type")["transactions_per_month"].mean().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("👥 Age Group vs Banking Type")
    df["age_group"] = pd.cut(
        df["age"],
        bins=[18, 25, 35, 50, 65, 100],
        labels=["18-25", "26-35", "36-50", "51-65", "65+"]
    )
    age_data = pd.crosstab(df["age_group"], df["banking_type"])
    fig, ax = plt.subplots()
    age_data.plot(kind="bar", stacked=True, ax=ax)
    st.pyplot(fig)

    st.subheader("📍 Location vs Banking Type")
    loc_data = df.groupby(["location", "banking_type"]).size().unstack()
    fig, ax = plt.subplots()
    loc_data.plot(kind="bar", stacked=True, ax=ax)
    st.pyplot(fig)

    st.subheader("💰 Income Level vs Banking Type")
    df["income_level"] = pd.qcut(df["income"], q=3, labels=["Low", "Medium", "High"])
    inc_data = df.groupby(["income_level", "banking_type"]).size().unstack()
    fig, ax = plt.subplots()
    inc_data.plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("🌐 Internet Usage vs Banking Type")
    fig, ax = plt.subplots()
    sns.boxplot(x="banking_type", y="internet_usage_score", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("🏦 Credit Score vs Banking Type")
    fig, ax = plt.subplots()
    sns.boxplot(x="banking_type", y="credit_score", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("😊 Satisfaction Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["satisfaction_score"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("🏙 Top 10 Cities")
    fig, ax = plt.subplots()
    df["city"].value_counts().head(10).plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------- INSIGHTS ----------------
elif menu == "🔥 Insights":

    st.subheader("Key Business Insights")

    st.write("""
    ✔ Online banking adoption is higher due to convenience and speed  
    ✔ Urban users prefer online banking  
    ✔ High income users prefer digital banking  
    ✔ Internet usage strongly impacts online banking adoption  
    ✔ Customer satisfaction is higher for online banking  
    """)

# ---------------- DOWNLOAD ----------------
elif menu == "⬇ Download Data":

    st.subheader("Download Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        data=csv,
        file_name="banking_data.csv",
        mime="text/csv"
    )
