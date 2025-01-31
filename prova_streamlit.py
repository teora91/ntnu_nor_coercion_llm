import streamlit as st
import pandas as pd

# Load or create a sample DataFrame
@st.cache
def load_data():
    return pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie", "David"],
        "Age": [25, 30, 35, 40],
        "City": ["New York", "Los Angeles", "Chicago", "Houston"]
    })

df = load_data()

# Title
st.title("📊 Interactive DataFrame Viewer")

# Sidebar filters
name_filter = st.sidebar.text_input("Search by Name:")
age_filter = st.sidebar.slider("Filter by Age", min_value=int(df["Age"].min()), max_value=int(df["Age"].max()), value=(df["Age"].min(), df["Age"].max()))

# Apply filters
filtered_df = df[df["Name"].str.contains(name_filter, case=False, na=False)]
filtered_df = filtered_df[(filtered_df["Age"] >= age_filter[0]) & (filtered_df["Age"] <= age_filter[1])]

# Display DataFrame
st.dataframe(filtered_df)

# Download button
st.download_button(label="Download Data", data=filtered_df.to_csv(index=False), file_name="filtered_data.csv", mime="text/csv")
