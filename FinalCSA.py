import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
from textblob import TextBlob
from collections import Counter

# 🎨 Set Page Config
st.set_page_config(page_title="Customer Sentiment Analysis", layout="wide")

# 🌈 Header with Emojis
st.title("📊 Customer Sentiment Analysis Dashboard")
st.markdown("### Gain insights from customer reviews instantly!")

# 💂 Upload CSV File
uploaded_file = st.sidebar.file_uploader("📝 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ✅ Check if 'Review' column exists
    if "Review" not in df.columns:
        st.error("❌ The uploaded file must contain a 'Review' column.")
    else:
        # ✅ Sentiment Analysis Function
        def get_sentiment(text):
            polarity = TextBlob(str(text)).sentiment.polarity
            return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

        df["Sentiment"] = df["Review"].astype(str).apply(get_sentiment)
        
        # Sidebar Navigation
        menu = st.sidebar.radio("🔍 Select Analysis", [
            "📌 Data Preview",
            "📝 Sentiment Distribution",
            "🔠 Word Clouds",
            "📊 Sentiment Over Time",
            "💡 Common Keywords",
            "📉 Sentiment vs. Product Price",
            "🤔 Expectation Gap Analysis"
        ])

        if menu == "📌 Data Preview":
            st.subheader("📌 Data Preview")
            st.dataframe(df.head())
            st.write(f"✅ **Total Reviews Analyzed:** {len(df)}")
            
            # 📊 Data Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Sentiment Distribution")
                sentiment_counts = df["Sentiment"].value_counts()
                fig = px.bar(x=sentiment_counts.index, y=sentiment_counts.values, labels={'x':'Sentiment', 'y':'Count'}, title="Sentiment Distribution")
                st.plotly_chart(fig)
            
            with col2:
                st.markdown("### Review Length Histogram")
                df["Review_Length"] = df["Review"].str.len()
                fig = px.histogram(df, x="Review_Length", nbins=30, title="Review Length Distribution")
                st.plotly_chart(fig)
            
            st.markdown("### Sentiment Pie Chart")
            fig = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, title="Sentiment Breakdown")
            st.plotly_chart(fig)

        if menu == "📝 Sentiment Distribution":
            st.subheader("📝 Sentiment Distribution")
            sentiment_counts = df["Sentiment"].value_counts()
            fig = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, title="📊 Sentiment Breakdown", color=sentiment_counts.index)
            st.plotly_chart(fig)

        if menu == "🔠 Word Clouds":
            st.subheader("🔠 Word Clouds")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**💚 Positive Reviews**")
                pos_words = " ".join(df[df["Sentiment"] == "Positive"]["Review"])
                pos_wc = WordCloud(background_color="white", colormap="Greens").generate(pos_words)
                st.image(pos_wc.to_array())
            
            with col2:
                st.markdown("**💔 Negative Reviews**")
                neg_words = " ".join(df[df["Sentiment"] == "Negative"]["Review"])
                neg_wc = WordCloud(background_color="white", colormap="Reds").generate(neg_words)
                st.image(neg_wc.to_array())

        if menu == "📊 Sentiment Over Time":
            st.subheader("📊 Sentiment Over Time")
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                df = df.dropna(subset=["Date"])
                if not df.empty:
                    sentiment_trend = df.groupby(["Date", "Sentiment"]).size().reset_index(name="Count")
                    fig = px.line(sentiment_trend, x="Date", y="Count", color="Sentiment", title="📅 Sentiment Over Time")
                    st.plotly_chart(fig)
                else:
                    st.warning("⏳ No valid timestamps found in the dataset.")
            else:
                st.warning("⏳ No 'Date' column found in the dataset.")

        if menu == "💡 Common Keywords":
            st.subheader("💡 Most Common Keywords")
            words = " ".join(df["Review"].astype(str)).split()
            common_words = Counter(words).most_common(10)
            common_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])
            st.table(common_df)


        if menu == "📉 Sentiment vs. Product Price":
            if "product_price" in df.columns:
                df["product_price"] = pd.to_numeric(df["product_price"], errors='coerce')
                if df["product_price"].notna().any():
                    st.subheader("📉 Sentiment vs. Product Price")
                    fig = px.box(df, x="Sentiment", y="product_price", color="Sentiment", title="Product Price Distribution by Sentiment")
                    st.plotly_chart(fig)
                else:
                    st.warning("Product Price data not available or not numeric.")
            else:
                st.warning("Product Price column not found in dataset.")

        if menu == "🤔 Expectation Gap Analysis":
            df["Review"].fillna("", inplace=True)
            gap_reviews = df[df["Review"].str.contains("expected|better than expected|not as expected", case=False)]
            st.write(gap_reviews[["Review", "Sentiment"]].head(5))

else:
    st.sidebar.warning("⚠️ Please upload a dataset to proceed.")
