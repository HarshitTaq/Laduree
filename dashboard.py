import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

# --- Branding Header ---
st.image("your_combined_logo_filename.png", use_container_width=True)
st.markdown(
    """<h2 style='text-align: center; color: #6A5ACD;'>French Spirit - Laduree Dashboard</h2>
    <p style='text-align: center; color: gray;'>Powered by Taqtics</p>""",
    unsafe_allow_html=True
)

def read_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        return None

def find_column(columns, target_names):
    col_lower = [c.lower() for c in columns]
    for t in target_names:
        if t.lower() in col_lower:
            return columns[col_lower.index(t.lower())]
    return None

uploaded_file = st.file_uploader("Upload your Performance CSV or Excel file (all months)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    df = read_file(uploaded_file)
    if df is None:
        st.error("Unsupported file type! Please upload CSV or Excel.")
        st.stop()

    # Find required columns flexibly
    col_country = find_column(df.columns, ["Country"])
    col_store = find_column(df.columns, ["Store"])
    col_audit_status = find_column(df.columns, ["Audit Status"])
    col_entity_id = find_column(df.columns, ["Entity Id"])
    col_employee_name = find_column(df.columns, ["Employee Name"])
    col_result = find_column(df.columns, ["Result"])
    col_submitted_for = find_column(df.columns, ["Submitted For", "Submitted_For", "Submission Date", "Submission_Date"])
    col_store_kpi = find_column(df.columns, ["Store KPI", "Store_KPI"])
    col_ind_kpi = find_column(df.columns, ["Individual KPI", "Individual_KPI", "Individual Kpi"])

    # Validate presence
    missing = []
    for col, name in zip(
        [col_country, col_store, col_audit_status, col_entity_id, col_employee_name, col_result, col_submitted_for, col_store_kpi, col_ind_kpi],
        ["Country", "Store", "Audit Status", "Entity Id", "Employee Name", "Result", "Submitted For", "Store KPI", "Individual KPI"]):
        if col is None:
            missing.append(name)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    # Numeric and date conversion
    df[col_result] = pd.to_numeric(df[col_result], errors='coerce')
    df[col_store_kpi] = pd.to_numeric(df[col_store_kpi], errors="coerce")
    df[col_ind_kpi] = pd.to_numeric(df[col_ind_kpi], errors="coerce")
    df[col_submitted_for] = pd.to_datetime(df[col_submitted_for], errors='coerce')
    df = df.dropna(subset=[col_submitted_for])

    # Add Month-Year column for filtering
    df["__month_label__"] = df[col_submitted_for].dt.strftime('%B %Y')
    unique_months = sorted(df["__month_label__"].dropna().unique(), 
                           key=lambda x: pd.to_datetime(x, format='%B %Y'))

    # ---- SIDEBAR FILTERS ----
    st.sidebar.header("Filters")
    # Month filter
    month_options = ["All"] + unique_months
    month_selected = st.sidebar.selectbox("Select Month", month_options)
    if month_selected != "All":
        data_df = df[df["__month_label__"] == month_selected].copy()
    else:
        data_df = df.copy()

    # Country filter with "All"
    unique_countries = sorted(data_df[col_country].dropna().unique())
    country_options = ["All"] + unique_countries

    # Store filter with "All"
    unique_stores = sorted(data_df[col_store].dropna().unique())
    store_options = ["All"] + unique_stores

    # Store Performance by Country (with "All" countries or just one)
    st.subheader("🏆 Store Performance by Country")
    selected_country_perf = st.selectbox("Select Country", country_options, key="perf_country")
    selected_store_perf = st.selectbox("Select Store", store_options, key="perf_store")
    if selected_country_perf == "All":
        perf_df = data_df.copy()
    else:
        perf_df = data_df[data_df[col_country] == selected_country_perf]
    if selected_store_perf != "All":
        perf_df = perf_df[perf_df[col_store] == selected_store_perf]
    country_label = selected_country_perf if selected_country_perf != "All" else "All Countries"
    store_label = selected_store_perf if selected_store_perf != "All" else "All Stores"

    country_store_avg = perf_df.groupby(col_store)[col_result].mean().reset_index()
    country_store_avg = country_store_avg.sort_values(by=col_result, ascending=False)
    fig_country_perf = px.bar(
        country_store_avg,
        x=col_store,
        y=col_result,
        text=col_result,
        title=f"Store Performance in {country_label} (High to Low)",
        labels={col_result: "Average Score"}
    )
    fig_country_perf.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_country_perf.update_layout(xaxis_tickangle=-45, yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_country_perf)

    # Country-wise Bell Curve and Drilldown (with "All" countries or just one)
    st.subheader("Country-wise Bell Curve and Drilldown")
    selected_country_drill = st.selectbox("Select Country", country_options, key="drilldown_country")
    selected_store_drill = st.selectbox("Select Store", store_options, key="drilldown_store")
    if selected_country_drill == "All":
        drill_df = data_df.copy()
    else:
        drill_df = data_df[data_df[col_country] == selected_country_drill]
    if selected_store_drill != "All":
        drill_df = drill_df[drill_df[col_store] == selected_store_drill]
    drill_country_label = selected_country_drill if selected_country_drill != "All" else "All Countries"
    drill_store_label = selected_store_drill if selected_store_drill != "All" else "All Stores"

    fig_country = px.histogram(
        drill_df,
        x=col_result,
        nbins=20,
        color=col_audit_status,
        hover_data=[col_entity_id, col_audit_status, col_employee_name],
        labels={col_result: "Performance Score"},
        title=f"Performance Bell Curve for {drill_country_label} - {drill_store_label}"
    )
    fig_country.update_layout(bargap=0.1)
    st.plotly_chart(fig_country)

    st.markdown(f"### Employees in {drill_country_label} - {drill_store_label}")
    st.dataframe(
        drill_df[[col_employee_name, col_store, col_entity_id, col_audit_status, col_result]]
        .sort_values(by=col_result, ascending=False)
    )

    # --- (rest of your dashboard code goes here, unchanged) ---
    # Store-wise Bell Curve, Probability Density, KPIs, etc.

else:
    st.info("Please upload a CSV or Excel file to begin.")
