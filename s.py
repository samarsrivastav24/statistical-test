#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Statistical Tests Dashboard", layout="wide")

st.title("📊 Statistical Tests on Titanic Dataset")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Titanic-Dataset.csv")

    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = df.drop(columns=['Cabin'])

    return df

df = load_data()

# -----------------------------
# Show dataset
# -----------------------------
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# Research Questions
# -----------------------------
st.subheader("❓ Research Questions")

st.markdown("""
**Q1.** Is there an association between passenger survival status and sex?  
**Test:** Chi-square test of independence  

**Q2.** Is the mean age of survivors different from the mean age of non-survivors?  
**Test:** Independent two-sample t-test  

**Q3.** Does the mean fare paid differ across passenger classes (1st, 2nd, 3rd)?  
**Test:** One-way ANOVA  
""")

# -----------------------------
# Results DataFrame
# -----------------------------
results = pd.DataFrame(columns=[
    "Test Name",
    "Research Question",
    "Variable 1",
    "Variable 2",
    "Statistic",
    "P-Value",
    "Conclusion"
])

# -----------------------------
# Chi-square Test
# -----------------------------
contingency = pd.crosstab(df['Survived'], df['Sex'])
chi2, p, dof, expected = chi2_contingency(contingency)

results.loc[len(results)] = [
    "Chi-Square",
    "Is survival associated with sex?",
    "Survived",
    "Sex",
    chi2,
    p,
    "Significant" if p < 0.05 else "Not Significant"
]

# -----------------------------
# t-Test
# -----------------------------
survived_age = df[df['Survived'] == 1]['Age']
not_survived_age = df[df['Survived'] == 0]['Age']

t_stat, p = ttest_ind(survived_age, not_survived_age)

results.loc[len(results)] = [
    "T-Test",
    "Is mean age different between survivors and non-survivors?",
    "Age",
    "Survived",
    t_stat,
    p,
    "Significant" if p < 0.05 else "Not Significant"
]

# -----------------------------
# ANOVA
# -----------------------------
fare_class1 = df[df['Pclass'] == 1]['Fare']
fare_class2 = df[df['Pclass'] == 2]['Fare']
fare_class3 = df[df['Pclass'] == 3]['Fare']

f_stat, p = f_oneway(fare_class1, fare_class2, fare_class3)

results.loc[len(results)] = [
    "ANOVA",
    "Does mean fare differ across passenger classes?",
    "Fare",
    "Pclass",
    f_stat,
    p,
    "Significant" if p < 0.05 else "Not Significant"
]

# -----------------------------
# Display results
# -----------------------------
st.subheader("📈 Statistical Test Results")

st.dataframe(results, use_container_width=True, height=300)

# -----------------------------
# Interpretation
# -----------------------------
st.subheader("🧠 Key Interpretation")
st.markdown("""
- **Chi-square:** Tests association between categorical variables  
- **t-test:** Compares means between two groups  
- **ANOVA:** Compares means across multiple groups  

*p-value < 0.05 indicates statistical significance*
""")


# In[ ]:




