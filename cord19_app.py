# cord19_app.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from collections import Counter
from wordcloud import WordCloud

# -----------------------------
# Streamlit App Title
# -----------------------------
st.title("CORD-19 Data Explorer")
st.write("Simple exploration of COVID-19 research papers using the metadata.csv file")

# -----------------------------
# Load and Clean Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("metadata.csv")

    # Convert publish_time to datetime
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['year'] = df['publish_time'].dt.year

    # Fill missing text fields
    df['title'] = df['title'].fillna("No Title")
    df['abstract'] = df['abstract'].fillna("")
    df['journal'] = df['journal'].fillna("Unknown Journal")

    # Add word count for abstracts
    df['abstract_word_count'] = df['abstract'].apply(lambda x: len(x.split()))

    return df

df = load_data()

# -----------------------------
# Data Exploration
# -----------------------------
st.subheader("Dataset Overview")
st.write("Shape of dataset:", df.shape)
st.write("Sample of dataset:")
st.write(df.head())

# -----------------------------
# Interactive Year Filter
# -----------------------------
st.subheader("Filter by Year")
min_year = int(df['year'].dropna().min())
max_year = int(df['year'].dropna().max())
year_range = st.slider("Select year range", min_year, max_year, (2020, 2021))

filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

st.write(f"Number of papers in selected range: {filtered.shape[0]}")
st.write(filtered[['title', 'journal', 'year']].head())

# -----------------------------
# Publications by Year
# -----------------------------
st.subheader("Publications by Year")
year_counts = filtered['year'].value_counts().sort_index()
fig, ax = plt.subplots()
ax.bar(year_counts.index, year_counts.values)
ax.set_title("Publications by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Papers")
st.pyplot(fig)

# -----------------------------
# Top Journals
# -----------------------------
st.subheader("Top Journals Publishing COVID-19 Papers")
top_journals = filtered['journal'].value_counts().head(10)
fig, ax = plt.subplots()
top_journals.plot(kind='barh', ax=ax)
ax.set_title("Top Journals")
ax.set_xlabel("Number of Papers")
st.pyplot(fig)

# -----------------------------
# Word Cloud of Titles
# -----------------------------
st.subheader("Word Cloud of Paper Titles")
all_titles = " ".join(filtered['title'])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_titles)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# -----------------------------
# Source Distribution
# -----------------------------
if 'source_x' in df.columns:
    st.subheader("Distribution of Papers by Source")
    top_sources = filtered['source_x'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_sources.values, y=top_sources.index, ax=ax)
    ax.set_title("Top Sources")
    st.pyplot(fig)

st.write("Analysis Complete ✅")
