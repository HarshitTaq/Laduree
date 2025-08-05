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

    # Store filter
    stores_selected = st.sidebar.multiselect("Select Store", options=data_df[col_store].unique(), default=data_df[col_store].unique())
    data_df = data_df[data_df[col_store].isin(stores_selected)]
    perf_df = perf_df[perf_df[col_store].isin(stores_selected)]
    drill_df = drill_df[drill_df[col_store].isin(stores_selected)]

    # Keep only earliest submission per employee-store-country-month
    data_df = data_df.sort_values([col_country, col_store, col_employee_name, col_submitted_for])
    data_df = data_df.drop_duplicates(subset=[col_country, col_store, col_employee_name, "__month_label__"], keep='first')
    perf_df = perf_df.sort_values([col_country, col_store, col_employee_name, col_submitted_for])
    perf_df = perf_df.drop_duplicates(subset=[col_country, col_store, col_employee_name, "__month_label__"], keep='first')
    drill_df = drill_df.sort_values([col_country, col_store, col_employee_name, col_submitted_for])
    drill_df = drill_df.drop_duplicates(subset=[col_country, col_store, col_employee_name, "__month_label__"], keep='first')

    # Month/cumulative header
    display_label = month_selected if month_selected != "All" else "All Months"
    st.markdown(f"<h3 style='text-align: center; color: #20B2AA;'>Data for: {display_label}</h3>", unsafe_allow_html=True)

    # --- Store-wise Count by Audit Status ---
    st.subheader("📊 Store-wise Count by Audit Status")
    selected_stores_bar = st.multiselect(
        "Select Store(s) for Audit Status Count Chart",
        options=sorted(data_df[col_store].dropna().unique()),
        default=sorted(data_df[col_store].dropna().unique())
    )
    filtered_status_df = data_df[data_df[col_store].isin(selected_stores_bar)]
    fig_store_audit_status = px.bar(
        filtered_status_df.groupby([col_store, col_audit_status]).size().reset_index(name='Count'),
        x=col_store,
        y='Count',
        color=col_audit_status,
        barmode='stack',
        title='Store-wise Count by Audit Status',
        labels={'Count': 'Number of Employees'}
    )
    fig_store_audit_status.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_store_audit_status)

    # --- Store Performance by Country, with "All" as cumulative
    st.subheader("🏆 Store Performance by Country")
    if country_selected_perf == "All":
        country_store_avg = perf_df.groupby(col_store)[col_result].mean().reset_index()
        country_label = "All Countries"
    else:
        country_store_avg = perf_df.groupby(col_store)[col_result].mean().reset_index()
        country_label = country_selected_perf
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

    # --- Country-wise Bell Curve and Drilldown with "All"
    st.subheader("Country-wise Bell Curve and Drilldown")
    if country_selected_drill == "All":
        drill_country_label = "All Countries"
    else:
        drill_country_label = country_selected_drill
    fig_country = px.histogram(
        drill_df,
        x=col_result,
        nbins=20,
        color=col_audit_status,
        hover_data=[col_entity_id, col_audit_status, col_employee_name],
        labels={col_result: "Performance Score"},
        title=f"Performance Bell Curve for {drill_country_label}"
    )
    fig_country.update_layout(bargap=0.1)
    st.plotly_chart(fig_country)

    st.markdown(f"### Employees in {drill_country_label}")
    st.dataframe(
        drill_df[[col_employee_name, col_store, col_entity_id, col_audit_status, col_result]]
        .sort_values(by=col_result, ascending=False)
    )

    # --- Store-wise Bell Curve ---
    st.subheader("Store-wise Bell Curve")
    store_options = ["All"] + list(data_df[col_store].dropna().unique())
    selected_store = st.selectbox("Select Store", store_options, key="drilldown_store")
    if selected_store == "All":
        store_df = data_df.copy()
        store_label = "All Stores"
    else:
        store_df = data_df[data_df[col_store] == selected_store]
        store_label = selected_store

    fig_store = px.histogram(
        store_df,
        x=col_result,
        nbins=20,
        color=col_audit_status,
        hover_data=[col_country, col_entity_id, col_employee_name],
        labels={col_result: "Performance Score"},
        title=f"Performance Bell Curve for {store_label}"
    )
    fig_store.update_layout(bargap=0.1)
    st.plotly_chart(fig_store)

    # --- Probability Distribution Chart ---
    st.subheader("Probability Density of Performance Scores")
    mean_score = data_df[col_result].mean()
    std_dev = data_df[col_result].std()
    x = np.linspace(data_df[col_result].min(), data_df[col_result].max(), 500)
    pdf_y = norm.pdf(x, mean_score, std_dev)
    fig_pdf = go.Figure()
    fig_pdf.add_trace(go.Scatter(x=x, y=pdf_y, mode='lines', name='PDF'))
    fig_pdf.add_vline(x=mean_score, line_dash='dash', line_color='green', annotation_text='Mean', annotation_position='top left')
    fig_pdf.update_layout(
        title='Probability Density Function (PDF) of Performance Scores',
        xaxis_title='Performance Score',
        yaxis_title='Probability Density'
    )
    st.plotly_chart(fig_pdf)
    st.markdown(f"**Mean Score:** {mean_score:.2f}  \n**Standard Deviation:** {std_dev:.2f}")

    # --- Country vs Score by Audit Status ---
    st.subheader("Score Distribution by Country and Audit Status")
    fig_country_status = px.strip(
        data_df,
        x=col_country,
        y=col_result,
        color=col_audit_status,
        hover_data=[col_employee_name, col_store, col_entity_id],
        stripmode="overlay",
        labels={col_result: "Performance Score"},
        title="Performance Scores by Country Grouped by Audit Status"
    )
    fig_country_status.update_layout(yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_country_status)

    # --- Store KPI/Individual KPI Chart ---
    st.subheader("📈 Store and Individual KPI Analysis")
    kpi_options = ["All", "Store KPI", "Individual KPI"]
    kpi_selected = st.selectbox("Select KPI for Store Chart", kpi_options)

    kpi_df = data_df.copy()
    kpi_plot_data = pd.DataFrame()

    if kpi_selected == "All":
        # Show average of both KPIs (side by side)
        avg_kpi = kpi_df.groupby(col_store)[[col_store_kpi, col_ind_kpi]].mean().reset_index()
        avg_kpi = avg_kpi.sort_values(by=col_store_kpi, ascending=False)
        kpi_plot_data = avg_kpi
        fig_kpi = px.bar(
            kpi_plot_data.melt(id_vars=col_store, value_vars=[col_store_kpi, col_ind_kpi],
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
