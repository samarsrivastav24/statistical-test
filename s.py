#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Titanic Statistical Analysis",
    page_icon="🚢",
    layout="wide"
)

# -----------------------------
# Title Section
# -----------------------------
st.markdown(
    "<h1 style='text-align: center;'>🚢 Titanic Statistical Analysis Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray;'>Statistical tests & visual insights</p>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Titanic-Dataset.csv")
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df.drop(columns=["Cabin"], inplace=True)
    return df

df = load_data()

# -----------------------------
# Top Metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Total Passengers", len(df))
col2.metric("✅ Survivors", df["Survived"].sum())
col3.metric("❌ Non-Survivors", (df["Survived"] == 0).sum())
col4.metric("📊 Survival Rate", f"{df['Survived'].mean()*100:.1f}%")

st.divider()

# -----------------------------
# Charts Section
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("👩‍🦰 Survival by Gender")
    gender_survival = df.groupby("Sex")["Survived"].mean()
    st.bar_chart(gender_survival)

with c2:
    st.subheader("🎟 Survival by Passenger Class")
    class_survival = df.groupby("Pclass")["Survived"].mean()
    st.line_chart(class_survival)

st.divider()

# -----------------------------
# Statistical Tests
# -----------------------------
st.subheader("📈 Statistical Test Results")

results = []

# Chi-square
contingency = pd.crosstab(df["Survived"], df["Sex"])
chi2, p1, _, _ = chi2_contingency(contingency)

results.append([
    "Chi-Square Test",
    "Survival vs Gender",
    round(chi2, 3),
    round(p1, 5),
    "Significant" if p1 < 0.05 else "Not Significant"
])

# t-test
t_stat, p2 = ttest_ind(
    df[df["Survived"] == 1]["Age"],
    df[df["Survived"] == 0]["Age"]
)

results.append([
    "T-Test",
    "Age vs Survival",
    round(t_stat, 3),
    round(p2, 5),
    "Significant" if p2 < 0.05 else "Not Significant"
])

# ANOVA
f_stat, p3 = f_oneway(
    df[df["Pclass"] == 1]["Fare"],
    df[df["Pclass"] == 2]["Fare"],
    df[df["Pclass"] == 3]["Fare"]
)

results.append([
    "ANOVA",
    "Fare vs Passenger Class",
    round(f_stat, 3),
    round(p3, 5),
    "Significant" if p3 < 0.05 else "Not Significant"
])

results_df = pd.DataFrame(
    results,
    columns=["Test", "Comparison", "Statistic", "P-Value", "Conclusion"]
)

st.dataframe(results_df, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# Final Insights
# -----------------------------
st.subheader("🧠 Key Insights")

colA, colB, colC = st.columns(3)

with colA:
    st.success("👩 Women had significantly higher survival rates.")

with colB:
    st.info("🎂 Survivors were younger on average.")

with colC:
    st.warning("💰 Fare differs significantly across classes.")

st.markdown(
    "<p style='text-align:center; color:gray;'>p-value < 0.05 indicates statistical significance</p>",
    unsafe_allow_html=True
)


# In[ ]:




