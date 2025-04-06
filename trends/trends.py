import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Jean Trend Comparison", layout="centered")
st.markdown("<h1 style='text-align: center;'>Compare Jean Style Trends 📈👖</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Select one or more styles to compare their monthly popularity since 2004.</p>", unsafe_allow_html=True)

# Folder with CSV files
csv_folder = "trends/trends_dropdown"
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

# Multiselect input
selected_files = st.multiselect("Select jean styles to compare:", sorted(csv_files))

if selected_files:
    combined_df = pd.DataFrame()

    for file in selected_files:
        style_name = file.replace(".csv", "").replace("-", " ").title()
        df = pd.read_csv(os.path.join(csv_folder, file))
        df.columns = [col.strip() for col in df.columns]
        df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
        df["Style"] = style_name
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Plot all styles with no markers and thinner lines
    fig = px.line(
        combined_df,
        x="Month",
        y="Popularity",
        color="Style",
        title="Jean Style Trends Since 2004"
    )

    fig.update_layout(
        title={
            'text': "Jean Style Trends Since 2004",
            'x': 0.5,
            'xanchor': 'center'
        },
        title_font=dict(
            size=24,
            family="Arial, sans-serif",
            color="black"
        ),
        width=1000,
        height=500
    )

    fig.update_traces(line=dict(width=1.5), marker=dict(size=0))  # thinner lines, no points

    st.plotly_chart(fig, use_container_width=False)
