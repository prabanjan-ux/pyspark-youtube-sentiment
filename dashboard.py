
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from config import FILE_PATHS

# Page configuration
st.set_page_config(
    page_title="YouTube Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load processed sentiment data"""
    try:
        if os.path.exists(FILE_PATHS['processed_sentiment']):
            df = pd.read_csv(FILE_PATHS['processed_sentiment'])
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            st.warning(f"Data file not found: {FILE_PATHS['processed_sentiment']}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 YouTube Comments Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None or len(df) == 0:
        st.error("No data available. Please run the analysis first using main.py")
        st.info("Run: `python main.py <youtube_url>`")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Sentiment filter
    if 'sentiment' in df.columns:
        sentiments = ['All'] + list(df['sentiment'].unique())
        selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
        
        if selected_sentiment != 'All':
            df = df[df['sentiment'] == selected_sentiment]
    
    # Date range filter
    if 'timestamp' in df.columns:
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            df = df[(df['timestamp'].dt.date >= date_range[0]) & 
                   (df['timestamp'].dt.date <= date_range[1])]
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Comments", f"{len(df):,}")
    
    with col2:
        if 'author' in df.columns:
            st.metric("Unique Authors", f"{df['author'].nunique():,}")
    
    with col3:
        if 'likes' in df.columns:
            st.metric("Total Likes", f"{df['likes'].sum():,}")
    
    with col4:
        if 'compound_score' in df.columns:
            avg_sentiment = df['compound_score'].mean()
            st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    
    st.divider()
    
    # Sentiment Distribution
    if 'sentiment' in df.columns:
        st.subheader("📈 Sentiment Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            sentiment_counts = df['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Breakdown",
                color=sentiment_counts.index,
                color_discrete_map={
                    'positive': '#00CC96',
                    'neutral': '#FFA15A',
                    'negative': '#EF553B'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Sentiment Counts",
                labels={'x': 'Sentiment', 'y': 'Count'},
                color=sentiment_counts.index,
                color_discrete_map={
                    'positive': '#00CC96',
                    'neutral': '#FFA15A',
                    'negative': '#EF553B'
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Sentiment over time
    if 'timestamp' in df.columns and 'sentiment' in df.columns:
        st.subheader("📅 Sentiment Trends Over Time")
        
        # Group by hour and sentiment
        df_time = df.groupby([df['timestamp'].dt.floor('1H'), 'sentiment']).size().reset_index(name='count')
        df_time.columns = ['timestamp', 'sentiment', 'count']
        
        fig_timeline = px.line(
            df_time,
            x='timestamp',
            y='count',
            color='sentiment',
            title="Sentiment Distribution Over Time",
            labels={'count': 'Number of Comments', 'timestamp': 'Time'},
            color_discrete_map={
                'positive': '#00CC96',
                'neutral': '#FFA15A',
                'negative': '#EF553B'
            }
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Engagement analysis
    if 'engagement_level' in df.columns:
        st.subheader("💬 Engagement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            engagement_counts = df['engagement_level'].value_counts()
            fig_engagement = px.bar(
                x=engagement_counts.index,
                y=engagement_counts.values,
                title="Engagement Levels",
                labels={'x': 'Engagement Level', 'y': 'Count'},
                color=engagement_counts.index
            )
            st.plotly_chart(fig_engagement, use_container_width=True)
        
        with col2:
            if 'sentiment' in df.columns:
                engagement_sentiment = pd.crosstab(df['engagement_level'], df['sentiment'])
                fig_heatmap = px.imshow(
                    engagement_sentiment,
                    title="Engagement vs Sentiment Heatmap",
                    labels=dict(x="Sentiment", y="Engagement Level", color="Count"),
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Word statistics
    if 'word_count' in df.columns and 'text_length' in df.columns:
        st.subheader("📝 Text Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_words = px.histogram(
                df,
                x='word_count',
                title="Word Count Distribution",
                labels={'word_count': 'Words per Comment'},
                nbins=50
            )
            st.plotly_chart(fig_words, use_container_width=True)
        
        with col2:
            fig_length = px.histogram(
                df,
                x='text_length',
                title="Character Length Distribution",
                labels={'text_length': 'Characters per Comment'},
                nbins=50
            )
            st.plotly_chart(fig_length, use_container_width=True)
    
    # Top comments
    st.subheader("🔝 Top Comments by Likes")
    
    if 'likes' in df.columns and 'text' in df.columns:
        top_comments = df.nlargest(10, 'likes')[['text', 'author', 'likes', 'sentiment', 'compound_score']]
        st.dataframe(top_comments, use_container_width=True)
    
    # Raw data
    with st.expander("📄 View Raw Data"):
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()