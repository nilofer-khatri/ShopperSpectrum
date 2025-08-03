import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Shopper Spectrum Dashboard", layout="wide")

st.title("🛍️ Shopper Spectrum Dashboard")
st.markdown("Analyze customer behavior, segmentation, and get product recommendations!")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("OnlineRetail.csv", encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

df = load_data()

# --- Load Models ---
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("rfm_scaler.pkl")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 EDA", "👥 Customer Segmentation", "🤝 Recommendations"])

# --- Tab 1: EDA ---
with tab1:
    st.header("📈 Exploratory Data Analysis")
    
    st.subheader("Top 10 Countries by Revenue (excluding UK)")
    top_countries = df[df['Country'] != 'United Kingdom'].groupby('Country')['TotalPrice'].sum().nlargest(10)
    st.bar_chart(top_countries)

    st.subheader("Top 10 Products by Quantity Sold")
    top_products = df.groupby('Description')['Quantity'].sum().nlargest(10)
    st.bar_chart(top_products)

    st.subheader("Monthly Revenue Trend")
    monthly_sales = df.set_index('InvoiceDate').resample('M')['TotalPrice'].sum()
    st.line_chart(monthly_sales)

# --- Tab 2: Customer Segmentation ---
with tab2:
    st.header("👥 RFM Segmentation with KMeans")

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    })

    rfm_scaled = scaler.transform(rfm)
    rfm['Cluster'] = kmeans.predict(rfm_scaled)

    st.subheader("Sample of RFM Table with Cluster")
    st.dataframe(rfm.reset_index().head())

    st.subheader("Cluster Counts")
    st.bar_chart(rfm['Cluster'].value_counts())

    st.subheader("2D Cluster Plot: Recency vs Monetary")
    fig, ax = plt.subplots()
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='Set2', ax=ax)
    st.pyplot(fig)

# --- Tab 3: Product Recommendation ---
with tab3:
    st.header("🤝 Product Recommendation")

    user_item_matrix = df.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum').fillna(0)
    item_user_matrix = user_item_matrix.T
    similarity_matrix = cosine_similarity(item_user_matrix)
    sim_df = pd.DataFrame(similarity_matrix, index=item_user_matrix.index, columns=item_user_matrix.index)

    st.subheader("Get Similar Product Recommendations")
    product_list = sorted(sim_df.columns)
    selected_product = st.selectbox("Select a Product", product_list)
    top_n = st.slider("Number of Recommendations", 1, 10, 5)

    if st.button("Show Recommendations"):
        recommended = sim_df[selected_product].sort_values(ascending=False)[1:top_n+1]
        st.write("**Recommended Products:**")
        st.dataframe(recommended)
