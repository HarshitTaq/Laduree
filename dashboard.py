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
    # Return first matching column name (case-insensitive)
    col_lower = [c.lower() for c in columns]
    for t in target_names:
        if t.lower() in col_lower:
            return columns[col_lower.index(t.lower())]
    return None

uploaded_file = st.file_uploader("Upload your Performance CSV or Excel file", type=["csv", "xlsx", "xls"])

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

    # Validate presence
    missing = []
    for col, name in zip([col_country, col_store, col_audit_status, col_entity_id, col_employee_name, col_result, col_submitted_for],
                         ["Country", "Store", "Audit Status", "Entity Id", "Employee Name", "Result", "Submitted For"]):
        if col is None:
            missing.append(name)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    # Numeric conversions and date parsing
    df[col_result] = pd.to_numeric(df[col_result], errors='coerce')
    df[col_submitted_for] = pd.to_datetime(df[col_submitted_for], errors='coerce')

    df = df.dropna(subset=[col_submitted_for])

    # Keep only earliest submission per employee-store-country
    df = df.sort_values([col_country, col_store, col_employee_name, col_submitted_for])
    df = df.drop_duplicates(subset=[col_country, col_store, col_employee_name], keep='first')

    # Month display string
    month_str = df[col_submitted_for].dt.strftime('%B %Y').mode()
    month_str = month_str[0] if not month_str.empty else "Unknown Month"
    st.markdown(f"<h3 style='text-align: center; color: #20B2AA;'>Data for: {month_str}</h3>", unsafe_allow_html=True)

    # Sidebar filters
    st.sidebar.header("Filters")
    countries_selected = st.sidebar.multiselect("Select Country", options=df[col_country].unique(), default=df[col_country].unique())
    stores_selected = st.sidebar.multiselect("Select Store", options=df[col_store].unique(), default=df[col_store].unique())

    filtered_df = df[(df[col_country].isin(countries_selected)) & (df[col_store].isin(stores_selected))]

    st.subheader("📊 Store-wise Count by Audit Status")
    selected_stores_bar = st.multiselect(
        "Select Store(s) for Audit Status Count Chart",
        options=sorted(df[col_store].dropna().unique()),
        default=sorted(df[col_store].dropna().unique())
    )
    filtered_status_df = df[df[col_store].isin(selected_stores_bar)]

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

    st.subheader("🏆 Store Performance by Country")
    selected_country_perf = st.selectbox(
        "Select Country to View Store Performance",
        sorted(df[col_country].dropna().unique())
    )
    country_store_avg = df[df[col_country] == selected_country_perf].groupby(col_store)[col_result].mean().reset_index()
    country_store_avg = country_store_avg.sort_values(by=col_result, ascending=False)

    fig_country_perf = px.bar(
        country_store_avg,
        x=col_store,
        y=col_result,
        text=col_result,
        title=f"Store Performance in {selected_country_perf} (High to Low)",
        labels={col_result: "Average Score"}
    )
    fig_country_perf.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_country_perf.update_layout(xaxis_tickangle=-45, yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_country_perf)

    st.subheader("Country-wise Bell Curve and Drilldown")
    selected_country = st.selectbox("Select Country for Drilldown", sorted(df[col_country].dropna().unique()), key="drilldown_country")
    country_df = df[df[col_country] == selected_country]

    fig_country = px.histogram(
        country_df,
        x=col_result,
        nbins=20,
        color=col_audit_status,
        hover_data=[col_entity_id, col_audit_status, col_employee_name],
        labels={col_result: "Performance Score"},
        title=f"Performance Bell Curve for {selected_country}"
    )
    fig_country.update_layout(bargap=0.1)
    st.plotly_chart(fig_country)

    st.markdown(f"### Employees in {selected_country}")
    st.dataframe(
        country_df[[col_employee_name, col_store, col_entity_id, col_audit_status, col_result]]
        .sort_values(by=col_result, ascending=False)
    )

    st.subheader("Store-wise Bell Curve")
    selected_store = st.selectbox("Select Store", sorted(df[col_store].dropna().unique()), key="drilldown_store")
    store_df = df[df[col_store] == selected_store]

    fig_store = px.histogram(
        store_df,
        x=col_result,
        nbins=20,
        color=col_audit_status,
        hover_data=[col_country, col_entity_id, col_employee_name],
        labels={col_result: "Performance Score"},
        title=f"Performance Bell Curve for {selected_store}"
    )
    fig_store.update_layout(bargap=0.1)
    st.plotly_chart(fig_store)

    st.subheader("Probability Density of Performance Scores")
    mean_score = filtered_df[col_result].mean()
    std_dev = filtered_df[col_result].std()

    x = np.linspace(filtered_df[col_result].min(), filtered_df[col_result].max(), 500)
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

    st.subheader("Score Distribution by Country and Audit Status")
    fig_country_status = px.strip(
        filtered_df,
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

else:
    st.info("Please upload a CSV or Excel file to begin.")
