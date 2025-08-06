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

# Consistent Audit Status color coding
status_colors = {
    "Outstanding": "#006400",            # Dark Green
    "Meets Expectations": "#32CD32",     # Brighter Green
    "Needs Improvement": "#FFC0CB",      # Light Pink
    "Below Expectations": "#FF0000"      # Red
}

uploaded_file = st.file_uploader("Upload your Performance CSV or Excel file (all months)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    df = read_file(uploaded_file)
    if df is None:
        st.error("Unsupported file type! Please upload CSV or Excel.")
        st.stop()

    # Find required columns
    col_country = find_column(df.columns, ["Country"])
    col_store = find_column(df.columns, ["Store"])
    col_audit_status = find_column(df.columns, ["Audit Status"])
    col_entity_id = find_column(df.columns, ["Entity Id"])
    col_employee_name = find_column(df.columns, ["Employee Name"])
    col_result = find_column(df.columns, ["Result"])
    col_submitted_for = find_column(df.columns, ["Submitted For", "Submission_Date", "Submission Date"])
    col_store_kpi = find_column(df.columns, ["Store KPI", "Store_KPI"])
    col_ind_kpi = find_column(df.columns, ["Individual KPI", "Individual_KPI", "Individual Kpi"])

    # Validate
    missing = []
    for col, name in zip(
        [col_country, col_store, col_audit_status, col_entity_id, col_employee_name, col_result, col_submitted_for, col_store_kpi, col_ind_kpi],
        ["Country", "Store", "Audit Status", "Entity Id", "Employee Name", "Result", "Submitted For", "Store KPI", "Individual KPI"]):
        if col is None:
            missing.append(name)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    # Clean up data
    df[col_result] = pd.to_numeric(df[col_result], errors='coerce')
    df[col_store_kpi] = pd.to_numeric(df[col_store_kpi], errors="coerce")
    df[col_ind_kpi] = pd.to_numeric(df[col_ind_kpi], errors="coerce")
    df[col_submitted_for] = pd.to_datetime(df[col_submitted_for], errors='coerce')
    df = df.dropna(subset=[col_submitted_for])

    df["__month_label__"] = df[col_submitted_for].dt.strftime('%B %Y')
    unique_months = sorted(df["__month_label__"].dropna().unique(), 
                           key=lambda x: pd.to_datetime(x, format='%B %Y'))

    # ---- SIDEBAR FILTERS ----
    st.sidebar.header("Filters")
    month_options = ["All"] + unique_months
    month_selected = st.sidebar.selectbox("Select Month", month_options)
    data_df = df if month_selected == "All" else df[df["__month_label__"] == month_selected]

    unique_countries = sorted(data_df[col_country].dropna().unique())
    country_options = ["All"] + unique_countries
    country_selected_perf = st.sidebar.selectbox("Country for 'Performance by Store'", country_options, key="perf_country")
    perf_df = data_df if country_selected_perf == "All" else data_df[data_df[col_country] == country_selected_perf]

    country_selected_drill = st.sidebar.selectbox("Country for 'Country-wise Bell Curve'", country_options, key="drill_country")
    drill_df = data_df if country_selected_drill == "All" else data_df[data_df[col_country] == country_selected_drill]

    country_selected_kpi = st.sidebar.selectbox("Country for 'Store and Individual KPI Analysis'", country_options, key="kpi_country")
    kpi_df = data_df if country_selected_kpi == "All" else data_df[data_df[col_country] == country_selected_kpi]

    stores_selected = st.sidebar.multiselect("Select Store", options=data_df[col_store].unique(), default=data_df[col_store].unique())
    data_df = data_df[data_df[col_store].isin(stores_selected)]
    perf_df = perf_df[perf_df[col_store].isin(stores_selected)]
    drill_df = drill_df[drill_df[col_store].isin(stores_selected)]
    kpi_df = kpi_df[kpi_df[col_store].isin(stores_selected)]

    # Deduplicate
    for df_sub in [data_df, perf_df, drill_df, kpi_df]:
        df_sub.sort_values([col_country, col_store, col_employee_name, col_submitted_for], inplace=True)
        df_sub.drop_duplicates(subset=[col_country, col_store, col_employee_name, "__month_label__"], keep='first', inplace=True)

    # --- Store-wise Count by Audit Status ---
    st.subheader("📊 Store-wise Count by Audit Status")
    fig_store_audit_status = px.bar(
        data_df.groupby([col_store, col_audit_status]).size().reset_index(name='Count'),
        x=col_store, y='Count', color=col_audit_status,
        barmode='stack',
        labels={'Count': 'Number of Employees'},
        title='Store-wise Count by Audit Status',
        color_discrete_map=status_colors
    )
    fig_store_audit_status.update_layout(xaxis_tickangle=-45)
    fig_store_audit_status.update_traces(marker_line_color='black', marker_line_width=0.5)
    st.plotly_chart(fig_store_audit_status)

    # --- Performance by Store ---
    st.subheader("🏆 Performance by Store")
    country_store_avg = perf_df.groupby(col_store)[col_result].mean().reset_index()
    country_store_avg = country_store_avg.sort_values(by=col_result, ascending=False)
    fig_country_perf = px.bar(
        country_store_avg,
        x=col_store, y=col_result, text=col_result,
        labels={col_result: "Average Score"},
        title=f"Store Performance in {country_selected_perf}"
    )
    fig_country_perf.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_country_perf.update_layout(xaxis_tickangle=-45, yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_country_perf)

    # --- Performance by Country ---
    st.subheader("🏆 Performance by Country")
    country_avg = perf_df.groupby(col_country)[col_result].mean().reset_index()
    country_avg = country_avg.sort_values(by=col_result, ascending=False)
    fig_perf_by_country = px.bar(
        country_avg,
        x=col_country,
        y=col_result,
        text=col_result,
        labels={col_country: "Country", col_result: "Average Score"},
        title="Average Performance by Country"
    )
    fig_perf_by_country.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_perf_by_country.update_layout(xaxis_tickangle=-45, yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_perf_by_country)


    
    # --- Country-wise Bell Curve ---
    st.subheader("Country-wise Bell Curve")
    fig_country = px.histogram(
        drill_df, x=col_result, nbins=20, color=col_audit_status,
        hover_data=[col_entity_id, col_employee_name],
        labels={col_result: "Performance Score"},
        title=f"Performance Bell Curve for {country_selected_drill}",
        color_discrete_map=status_colors
    )
    fig_country.update_traces(marker_line_color='black', marker_line_width=0.5)
    st.plotly_chart(fig_country)

    # --- Grid of Employees (Restored) ---
    st.markdown(f"### Employees in {country_selected_drill}")
    st.dataframe(
        drill_df[[col_employee_name, col_store, col_entity_id, col_audit_status, col_result]]
        .sort_values(by=col_result, ascending=False)
    )

    # --- Store-wise Bell Curve ---
    st.subheader("Store-wise Bell Curve")
    store_options = ["All"] + list(data_df[col_store].dropna().unique())
    selected_store = st.selectbox("Select Store", store_options, key="drilldown_store")
    store_df = data_df if selected_store == "All" else data_df[data_df[col_store] == selected_store]
    store_label = "All Stores" if selected_store == "All" else selected_store

    fig_store = px.histogram(
        store_df, x=col_result, nbins=20, color=col_audit_status,
        hover_data=[col_country, col_entity_id, col_employee_name],
        labels={col_result: "Performance Score"},
        title=f"Performance Bell Curve for {store_label}",
        color_discrete_map=status_colors
    )
    fig_store.update_traces(marker_line_color='black', marker_line_width=0.5)
    st.plotly_chart(fig_store)

    # --- Probability Distribution Chart ---
    st.subheader("Probability Density of Performance Scores")
    mean_score = data_df[col_result].mean()
    std_dev = data_df[col_result].std()
    x = np.linspace(data_df[col_result].min(), data_df[col_result].max(), 500)
    pdf_y = norm.pdf(x, mean_score, std_dev)
    fig_pdf = go.Figure()
    fig_pdf.add_trace(go.Scatter(x=x, y=pdf_y, mode='lines', name='PDF'))
    fig_pdf.add_vline(x=mean_score, line_dash='dash', line_color='green',
                      annotation_text='Mean', annotation_position='top left')
    fig_pdf.update_layout(
        title='Probability Density Function (PDF) of Performance Scores',
        xaxis_title='Performance Score', yaxis_title='Probability Density'
    )
    st.plotly_chart(fig_pdf)
    st.markdown(f"**Mean Score:** {mean_score:.2f}  \n**Standard Deviation:** {std_dev:.2f}")

    # --- Score Distribution by Country and Audit Status ---
    st.subheader("Score Distribution by Country and Audit Status")
    fig_country_status = px.strip(
        data_df, x=col_country, y=col_result, color=col_audit_status,
        hover_data=[col_employee_name, col_store, col_entity_id],
        stripmode="overlay",
        labels={col_result: "Performance Score"},
        title="Performance Scores by Country Grouped by Audit Status",
        color_discrete_map=status_colors
    )
    fig_country_status.update_layout(yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_country_status)

    # --- Consolidated Bell Curve with Performance Bands ---
    st.subheader("📈 Consolidated Bell Curve with Performance Bands")
    bell_scope = st.radio("Select Bell Curve Scope", ["Consolidated", "By Country", "By Store"], horizontal=True)

    if bell_scope == "Consolidated":
        bell_df = data_df.copy()
        bell_title = "All Countries & Stores"
    elif bell_scope == "By Country":
        bell_country = st.selectbox("Select Country", ["All"] + list(data_df[col_country].unique()))
        if bell_country == "All":
            bell_df = data_df.copy()
            bell_title = "All Countries"
        else:
            bell_df = data_df[data_df[col_country] == bell_country]
            bell_title = f"{bell_country}"
    else:  # By Store
        bell_store = st.selectbox("Select Store", ["All"] + list(data_df[col_store].unique()))
        if bell_store == "All":
            bell_df = data_df.copy()
            bell_title = "All Stores"
        else:
            bell_df = data_df[data_df[col_store] == bell_store]
            bell_title = f"{bell_store}"

    if not bell_df.empty:
        mean_score = bell_df[col_result].mean()
        std_dev = bell_df[col_result].std()
        x = np.linspace(0, 100, 500)
        pdf_y = norm.pdf(x, mean_score, std_dev)

        fig_bell = go.Figure()
        fig_bell.add_trace(go.Histogram(
            x=bell_df[col_result],
            nbinsx=20,
            name="Scores",
            marker_color="lightgrey",
            marker_line_color="black",
            marker_line_width=0.5,
            opacity=0.6,
            histnorm='probability'
        ))
        fig_bell.add_trace(go.Scatter(
            x=x, y=pdf_y, mode='lines',
            name="Bell Curve", line=dict(color="blue", width=2)
        ))
        fig_bell.add_vrect(x0=0, x1=60, fillcolor="#FF0000", opacity=0.2, line_width=0, annotation_text="Below Expectations", annotation_position="top left")
        fig_bell.add_vrect(x0=60.1, x1=75.5, fillcolor="#FFC0CB", opacity=0.2, line_width=0, annotation_text="Needs Improv", annotation_position="top left")
        fig_bell.add_vrect(x0=75.6, x1=95, fillcolor="#32CD32", opacity=0.2, line_width=0, annotation_text="Meets Expectations", annotation_position="top left")
        fig_bell.add_vrect(x0=95.1, x1=100, fillcolor="#006400", opacity=0.2, line_width=0, annotation_text="Outstanding", annotation_position="top left")

        fig_bell.add_annotation(x=30, y=max(pdf_y)/1.5, text="Non-Performers", showarrow=False, font=dict(size=12, color="black"))
        fig_bell.add_annotation(x=67.5, y=max(pdf_y)/.5, text="Developing Performers", showarrow=False, font=dict(size=12, color="black"))
        fig_bell.add_annotation(x=90, y=max(pdf_y)/1.5, text="Top Performers", showarrow=False, font=dict(size=12, color="black"))

        # Calculate % of employees in each range
        total_count = len(bell_df)
        counts = {
            "Below Expectations": bell_df[bell_df[col_result] <= 60].shape[0],
            "Needs Improvement": bell_df[(bell_df[col_result] > 60) & (bell_df[col_result] <= 75.5)].shape[0],
            "Meets Expectations": bell_df[(bell_df[col_result] > 75.5) & (bell_df[col_result] <= 95)].shape[0],
            "Outstanding": bell_df[bell_df[col_result] > 95].shape[0],
        }
        percents = {k: (v / total_count * 100) if total_count > 0 else 0 for k, v in counts.items()}

        # Add percentage labels on top (no duplicate status text)
        fig_bell.add_annotation(
            x=30, y=max(pdf_y)*3,
            text=f"{percents['Below Expectations']:.1f}%",
            showarrow=False, font=dict(size=13, color="black")
        )
        fig_bell.add_annotation(
            x=67, y=max(pdf_y)*3,
            text=f"{percents['Needs Improvement']:.1f}%",
            showarrow=False, font=dict(size=13, color="black")
        )
        fig_bell.add_annotation(
            x=85, y=max(pdf_y)*3,
            text=f"{percents['Meets Expectations']:.1f}%",
            showarrow=False, font=dict(size=13, color="black")
        )
        fig_bell.add_annotation(
            x=97, y=max(pdf_y)*3,
            text=f"{percents['Outstanding']:.1f}%",
            showarrow=False, font=dict(size=13, color="black")
        )

        # Update layout and render chart
        fig_bell.update_layout(
            title=f"Performance Bell Curve for {bell_title}",
            xaxis_title="Performance Score",
            yaxis_title="Probability",
            bargap=0.05,
            margin=dict(t=100)  # extra top space for percentages
        )
        st.plotly_chart(fig_bell)


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
            title=f"Average Store KPI and Individual KPI by Store ({country_selected_kpi})"
        )
        fig_kpi.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_kpi)
    elif kpi_selected == "Store KPI":
        avg_store_kpi = kpi_df.groupby(col_store)[col_store_kpi].mean().reset_index()
        avg_store_kpi = avg_store_kpi.sort_values(by=col_store_kpi, ascending=False)
        fig_storekpi = px.bar(
            avg_store_kpi,
            x=col_store, y=col_store_kpi,
            title=f"Average Store KPI by Store ({country_selected_kpi})",
            labels={col_store_kpi: "Average Store KPI"}
        )
        fig_storekpi.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_storekpi)
    elif kpi_selected == "Individual KPI":
        avg_ind_kpi = kpi_df.groupby(col_store)[col_ind_kpi].mean().reset_index()
        avg_ind_kpi = avg_ind_kpi.sort_values(by=col_ind_kpi, ascending=False)
        fig_indkpi = px.bar(
            avg_ind_kpi,
            x=col_store, y=col_ind_kpi,
            title=f"Average Individual KPI by Store ({country_selected_kpi})",
            labels={col_ind_kpi: "Average Individual KPI"}
        )
        fig_indkpi.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_indkpi)

else:
    st.info("Please upload a CSV or Excel file to begin.")
