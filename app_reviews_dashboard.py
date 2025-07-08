import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud

# Configure page
st.set_page_config(page_title="App Reviews Sentiment Analyzer", layout="wide")

# Download NLTK data (only once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Constants
sentiment_order = ['Positive', 'Neutral', 'Negative']
palette = {'Positive': '#66c2a5', 'Neutral': '#fc8d62', 'Negative': '#8da0cb'}

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("apps_reviews.csv", parse_dates=['review_date'])
    df.dropna(subset=['review_text'], inplace=True)
    if 'sentiment' not in df.columns:
        df['polarity'] = df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['sentiment'] = df['polarity'].apply(
            lambda score: 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'
        )
    return df

df = load_data()

# Functions
def generate_wordcloud(text):
    stop_words = set(stopwords.words('english'))
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stop_words,
        max_words=100,
        max_font_size=40,
        scale=3,
        random_state=1
    ).generate(text)
    return wordcloud

import re
from collections import Counter

def get_word_frequencies(text):
    # Simple tokenizer using regex to split on non-word characters
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set(['the', 'and', 'to', 'a', 'is', 'in', 'it', 'you', 'of', 'for', 'this', 'on', 'with', 'was', 'that', 'are', 'as', 'but', 'be', 'have', 'not', 'at', 'my', 'so', 'if'])
    filtered_words = [word for word in words if word not in stopwords]
    return dict(Counter(filtered_words).most_common(20))


# Title and Description
st.title("📊 App Reviews Sentiment Analyzer")
st.markdown("Analyze app reviews by sentiment, app name, date, and rating.")

# Sidebar Filters
with st.sidebar:
    st.header("Filters")
    apps = sorted(df['app_name'].dropna().unique())
    selected_apps = st.multiselect("Select App(s):", apps)
    selected_sentiments = st.multiselect("Select Sentiment(s):", sentiment_order)

    min_date, max_date = df['review_date'].min(), df['review_date'].max()
    date_range = st.date_input("Select Date Range:", [min_date, max_date], min_value=min_date, max_value=max_date)

    rating_min, rating_max = float(df['rating'].min()), float(df['rating'].max())
    rating_range = st.slider("Select Rating Range:", min_value=rating_min, max_value=rating_max, value=(rating_min, rating_max))

    st.markdown("---")
    show_pie = st.checkbox("Show Pie Chart of Sentiment %")
    show_hist = st.checkbox("Show Polarity Score Histogram")
    show_wordcloud = st.checkbox("Show Word Cloud")
    show_wordfreq = st.checkbox("Show Word Frequency Bar Chart")

# Apply Filters
filtered_df = df.copy()
if selected_apps:
    filtered_df = filtered_df[filtered_df['app_name'].isin(selected_apps)]
if selected_sentiments:
    filtered_df = filtered_df[filtered_df['sentiment'].isin(selected_sentiments)]
filtered_df = filtered_df[
    (filtered_df['review_date'] >= pd.to_datetime(date_range[0])) &
    (filtered_df['review_date'] <= pd.to_datetime(date_range[1])) &
    (filtered_df['rating'] >= rating_range[0]) &
    (filtered_df['rating'] <= rating_range[1])
]

# Charts Layout
col1, col2 = st.columns(2)

# Sentiment Distribution
with col1:
    st.subheader("Sentiment Distribution by App")
    if not filtered_df.empty:
        sentiment_counts = filtered_df.groupby(['app_name', 'sentiment']).size().reset_index(name='count')
        plt.figure(figsize=(6,4))
        sns.barplot(data=sentiment_counts, x='app_name', y='count', hue='sentiment',
                    palette=palette, order=apps, hue_order=sentiment_order)
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.warning("No data available for selected filters.")

# Rating Trend Over Time
with col2:
    st.subheader("Time Trend: Average Rating per Month")
    if not filtered_df.empty:
        time_df = filtered_df.copy()
        time_df['month'] = time_df['review_date'].dt.to_period('M').astype(str)
        trend = time_df.groupby('month')['rating'].mean().reset_index()
        plt.figure(figsize=(6,4))
        sns.lineplot(data=trend, x='month', y='rating', marker='o', color='teal')
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.warning("No data available for selected filters.")

# Pie and Histogram
col3, col4 = st.columns(2)

with col3:
    if show_pie and not filtered_df.empty:
        st.subheader("Pie Chart: Sentiment Breakdown")
        sentiment_counts = filtered_df['sentiment'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(3.5, 3.5), dpi=70)
        ax1.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=[palette[s] for s in sentiment_counts.index],
            startangle=90
        )
        ax1.axis('equal')
        st.pyplot(fig1)
        plt.close(fig1)

with col4:
    if show_hist and not filtered_df.empty:
        st.subheader("Polarity Score Distribution")
        fig2, ax2 = plt.subplots(figsize=(4, 2.5), dpi=70)
        sns.histplot(filtered_df['polarity'], bins=30, kde=True, color='skyblue', ax=ax2)
        ax2.set_xlabel("Polarity Score")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
        plt.close(fig2)

# Word Cloud and Word Frequency
col5, col6 = st.columns(2)

with col5:
    if show_wordcloud and not filtered_df.empty:
        st.subheader("Word Cloud")
        all_text = " ".join(filtered_df['review_text'].astype(str))
        wc = generate_wordcloud(all_text)
        plt.figure(figsize=(6,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt.gcf())
        plt.clf()

with col6:
    if show_wordfreq and not filtered_df.empty:
        st.subheader("Word Frequency Bar Chart")
        all_text = " ".join(filtered_df['review_text'].astype(str))
        word_freq = get_word_frequencies(all_text)
        if word_freq:
            words, counts = zip(*word_freq.items())
            plt.figure(figsize=(6,4))
            sns.barplot(x=list(counts), y=list(words), palette="viridis")
            plt.xlabel("Frequency")
            plt.ylabel("Word")
            st.pyplot(plt.gcf())
            plt.clf()
        else:
            st.info("Not enough text data to show word frequency.")

# Table and Download
st.subheader("Filtered Reviews Table")
st.dataframe(filtered_df[['app_name', 'review_text', 'sentiment', 'rating', 'review_date']].reset_index(drop=True), height=200)

csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Filtered Reviews", data=csv, file_name='filtered_reviews.csv', mime='text/csv')

# Footer
st.markdown("---")
st.markdown("✅ *App Reviews Sentiment Analyzer — Streamlit Dashboard*")

