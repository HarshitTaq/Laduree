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
    country_selected_perf = st.sidebar.selectbox("Country for 'Store Performance by Country'", country_options, key="perf_country")
    if country_selected_perf == "All":
        perf_df = data_df.copy()
    else:
        perf_df = data_df[data_df[col_country] == country_selected_perf]

    country_selected_drill = st.sidebar.selectbox("Country for 'Country-wise Bell Curve'", country_options, key="drill_country")
    if country_selected_drill == "All":
        drill_df = data_df.copy()
    else:
        drill_df = data_df[data_df[col_country] == country_selected_drill]

    # New KPI Country Filter
    country_selected_kpi = st.sidebar.selectbox("Country for 'Store and Individual KPI Analysis'", country_options, key="kpi_country")
    if country_selected_kpi == "All":
        kpi_df = data_df.copy()
    else:
        kpi_df = data_df[data_df[col_country] == country_selected_kpi]

    # Store filter
    stores_selected = st.sidebar.multiselect("Select Store", options=data_df[col_store].unique(), default=data_df[col_store].unique())
    data_df = data_df[data_df[col_store].isin(stores_selected)]
    perf_df = perf_df[perf_df[col_store].isin(stores_selected)]
    drill_df = drill_df[drill_df[col_store].isin(stores_selected)]
    kpi_df = kpi_df[kpi_df[col_store].isin(stores_selected)]

    # Keep only earliest submission per employee-store-country-month
    for df_sub in [data_df, perf_df, drill_df, kpi_df]:
        df_sub.sort_values([col_country, col_store, col_employee_name, col_submitted_for], inplace=True)
        df_sub.drop_duplicates(subset=[col_country, col_store, col_employee_name, "__month_label__"], keep='first', inplace=True)

    # Month/cumulative header
    display_label = month_selected if month_selected != "All" else "All Months"
    st.markdown(f"<h3 style='text-align: center; color: #20B2AA;'>Data for: {display_label}</h3>", unsafe_allow_html=True)

    # --- Store KPI/Individual KPI Chart ---
    st.subheader("📈 Store and Individual KPI Analysis")
    kpi_options = ["All", "Store KPI", "Individual KPI"]
    kpi_selected = st.selectbox("Select KPI for Store Chart", kpi_options)

    if kpi_selected == "All":
        avg_kpi = kpi_df.groupby(col_store)[[col_store_kpi, col_ind_kpi]].mean().reset_index()
        avg_kpi = avg_kpi.sort_values(by=col_store_kpi, ascending=False)
        fig_kpi = px.bar(
            avg_kpi.melt(id_vars=col_store, value_vars=[col_store_kpi, col_ind_kpi],
                         var_name="KPI Type", value_name="Average"),
            x=col_store, y="Average", color="KPI Type",
            barmode="group",
            title="Average Store KPI and Individual KPI by Store (Descending Store KPI)"
        )
        fig_kpi.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_kpi)
    elif kpi_selected == "Store KPI":
        avg_store_kpi = kpi_df.groupby(col_store)[col_store_kpi].mean().reset_index()
        avg_store_kpi = avg_store_kpi.sort_values(by=col_store_kpi, ascending=False)
        fig_storekpi = px.bar(
            avg_store_kpi,
            x=col_store,
            y=col_store_kpi,
            title="Average Store KPI by Store (Descending)",
            labels={col_store_kpi: "Average Store KPI"}
        )
        fig_storekpi.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_storekpi)
    elif kpi_selected == "Individual KPI":
        avg_ind_kpi = kpi_df.groupby(col_store)[col_ind_kpi].mean().reset_index()
        avg_ind_kpi = avg_ind_kpi.sort_values(by=col_ind_kpi, ascending=False)
        fig_indkpi = px.bar(
            avg_ind_kpi,
            x=col_store,
            y=col_ind_kpi,
            title="Average Individual KPI by Store (Descending)",
            labels={col_ind_kpi: "Average Individual KPI"}
        )
        fig_indkpi.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_indkpi)
else:
    st.info("Please upload a CSV or Excel file to begin.")
