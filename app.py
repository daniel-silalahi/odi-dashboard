import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas
import pycountry
from adjustText import adjust_text
from itertools import combinations
from streamlit_option_menu import option_menu
import base64
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="AI Policy Analysis Dashboard",
    page_icon="🤖"
)

# --- Global Mappings & Definitions ---
bucket_mapping = {1: 'Sovereignty Type', 2: 'Data Flow Openness', 3: 'Security Justification', 4: 'Global Alignment', 5: 'Transparency & Accountability'}
label_mapping = {
    'Sovereignty Type': {-1: 'State-Centric', 0: 'Firm-Centric', 1: 'Individual-Centric'},
    'Data Flow Openness': {-1: 'Restrictive', 0: 'Conditional', 1: 'Supportive'},
    'Security Justification': {-1: 'Dominant', 0: 'Balanced', 1: 'Minor'},
    'Global Alignment': {-1: 'Unilateral', 0: 'Conditional', 1: 'Multilateral'},
    'Transparency & Accountability': {-1: 'Opaque', 0: 'Weak', 1: 'Strong'}
}
eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']

# --- Data for Blocs and Income Groups ---
bloc_mapping = {
    'African Union (AU)': ['DZ', 'AO', 'BJ', 'BW', 'BF', 'BI', 'CV', 'CM', 'CF', 'TD', 'KM', 'CG', 'CD', 'CI', 'DJ', 'EG', 'GQ', 'ER', 'SZ', 'ET', 'GA', 'GM', 'GH', 'GN', 'GW', 'KE', 'LS', 'LR', 'LY', 'MG', 'MW', 'ML', 'MR', 'MU', 'MA', 'MZ', 'NA', 'NE', 'NG', 'RW', 'ST', 'SN', 'SC', 'SL', 'SO', 'ZA', 'SS', 'SD', 'TZ', 'TG', 'TN', 'UG', 'ZM', 'ZW'],
    'ASEAN': ['BN', 'KH', 'ID', 'LA', 'MY', 'MM', 'PH', 'SG', 'TH', 'VN'],
    'CIS': ['AM', 'AZ', 'BY', 'KZ', 'KG', 'MD', 'RU', 'TJ', 'UZ'],
    'Council of Europe (CoE)': ['AL', 'AD', 'AM', 'AT', 'AZ', 'BE', 'BA', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'GE', 'DE', 'GR', 'HU', 'IS', 'IE', 'IT', 'LV', 'LI', 'LT', 'LU', 'MT', 'MD', 'MC', 'ME', 'NL', 'MK', 'NO', 'PL', 'PT', 'RO', 'SM', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'TR', 'UA', 'GB'],
    'G7': ['CA', 'FR', 'DE', 'IT', 'JP', 'GB', 'US'],
    'G20': ['AR', 'AU', 'BR', 'CA', 'CN', 'FR', 'DE', 'IN', 'ID', 'IT', 'JP', 'MX', 'KR', 'RU', 'SA', 'ZA', 'TR', 'GB', 'US'],
    'OECD': ['AU', 'AT', 'BE', 'CA', 'CL', 'CO', 'CR', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IS', 'IE', 'IL', 'IT', 'JP', 'KR', 'LV', 'LT', 'LU', 'MX', 'NL', 'NZ', 'NO', 'PL', 'PT', 'SK', 'SI', 'ES', 'SE', 'CH', 'TR', 'GB', 'US'],
}

# https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups
income_group_mapping = {
    'High income': ['AD', 'AE', 'AG', 'AR', 'AU', 'AT', 'BS', 'BH', 'BB', 'BE', 'BM', 'BN', 'BG', 'CA', 'CL', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FO', 'FI', 'FR', 'DE', 'GI', 'GR', 'GU', 'HK', 'HU', 'IS', 'IE', 'IL', 'IT', 'JP', 'KR', 'KW', 'LV', 'LI', 'LT', 'LU', 'MO', 'MT', 'MC', 'NL', 'NZ', 'NO', 'OM', 'PA', 'PL', 'PT', 'PR', 'QA', 'RO', 'SM', 'SA', 'SC', 'SG', 'SK', 'SI', 'ES', 'SE', 'CH', 'TT', 'GB', 'US', 'UY'],
    'Upper-middle income': ['AL', 'DZ', 'AO', 'AM', 'AZ', 'BY', 'BZ', 'BA', 'BW', 'BR', 'CN', 'CO', 'CR', 'CU', 'DJ', 'DO', 'EC', 'GQ', 'FJ', 'GA', 'GE', 'GT', 'GY', 'ID', 'IR', 'IQ', 'JM', 'KZ', 'KE', 'XK', 'LB', 'LY', 'MY', 'MV', 'MU', 'MX', 'MD', 'ME', 'MA', 'NA', 'NG', 'MK', 'PK', 'PY', 'PE', 'PH', 'RU', 'RS', 'ZA', 'LK', 'TH', 'TN', 'TR', 'TM', 'VE'],
    'Lower-middle income': ['BD', 'BJ', 'BT', 'BO', 'KH', 'CM', 'EG', 'GH', 'HN', 'IN', 'JO', 'KG', 'LA', 'MR', 'MN', 'MM', 'NP', 'NI', 'PS', 'SN', 'SL', 'SD', 'SY', 'TJ', 'UA', 'UZ', 'VN', 'ZM', 'ZW'],
    'Low income': ['AF', 'BF', 'BI', 'CF', 'TD', 'CD', 'ER', 'ET', 'GM', 'GN', 'GW', 'LR', 'MG', 'MW', 'ML', 'MZ', 'NE', 'RW', 'SO', 'SS', 'TZ', 'TG', 'UG'],
}

# Predefined Clusters
country_groups = {
    'Cluster 1 (Pragmatic & Commercial)': ['Argentina', 'Kenya', 'Nigeria', 'Singapore'],
    'Cluster 2 (The "Big Three" & Aligned States)': ['United States', 'Europe', 'China', 'Belize', 'Kyrgyzstan', 'Sri Lanka', 'Türkiye'],
    'Cluster 3 (Asian Security Paradigm)': ['Hong Kong', 'India', 'Japan', 'Pakistan'],
    'Cluster 4 (Outlier: Thailand)': ['Thailand'],
    'Cluster 5 (Outlier: Iceland)': ['Iceland']
}

# --- Caching Data Loading Functions ---
@st.cache_data
def load_main_data():
    """Loads and preprocesses the main dataset."""
    df = pd.read_csv('df_final.csv')
    df['Bucket_label'] = df['Bucket_majority'].map(bucket_mapping)
    df['Theme'] = df['Bucket_label']
    df['Year'] = pd.to_datetime(df['Date'], format='%Y', errors='coerce').dt.year
    def assign_display_name(country_code):
        if country_code in eu_countries: return 'Europe'
        try: return pycountry.countries.get(alpha_2=country_code).name
        except: return country_code
    df['Display_Name'] = df['Country'].apply(assign_display_name)
    return df

@st.cache_data
def load_geospatial_data():
    """Loads the world map shapefile."""
    try:
        world = geopandas.read_file('ne_110m_admin_0_countries.shp')
        return world
    except Exception as e:
        st.error(f"Error loading shapefile: {e}")
        st.error("Please ensure 'ne_110m_admin_0_countries.shp' and associated files are in the same directory.")
        return None

@st.cache_data
def get_image_as_base64(path):
    """Encodes an image file to a Base64 string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Image file not found at {path}. Please make sure it's in the correct directory.")
        return None

# --- Header Function ---
def styled_header(label, description=None):
    st.markdown(f'<div style="background-color: #e0e3e0; padding: 20px 10px; margin-top: -1.7rem;"><h1 style="color: #12308c; margin-bottom: 5px;">{label}</h1><p style="color: #747474; margin-bottom: 0;">{description}</p></div>', unsafe_allow_html=True)

# --- Load Data ---
df = load_main_data()
world = load_geospatial_data()

# --- CSS to remove gap between columns ---
st.markdown('<style>div[data-testid="stHorizontalBlock"] { gap: 0; }</style>', unsafe_allow_html=True)

# --- Top Navigation Bar with Logo ---
logo_path = "wordmark-72-W-170px.png"
logo_base64 = get_image_as_base64(logo_path)

nav_col1, nav_col2 = st.columns([1, 4])
with nav_col1:
    if logo_base64:
        st.markdown(f'<div style="background-color: #12308c; display: flex; align-items: center; justify-content: center; height: 100%; padding: 10px;"><a href="https://theodi.org/" target="_blank"><img src="data:image/png;base64,{logo_base64}" alt="ODI Logo" style="height: 50px;"></a></div>', unsafe_allow_html=True)
with nav_col2:
    page = option_menu(menu_title=None, options=["Global Overview", "Temporal Trends", "Comparative Analysis", "Cluster Analysis"], icons=["globe2", "graph-up", "people", "pie-chart"], orientation="horizontal", styles={"container": {"padding": "0!important", "background-color": "#12308c", "border-radius": "0"}, "icon": {"color": "#afd950", "font-size": "22px"}, "nav-link": {"font-size": "18px", "padding": "20px 5px", "text-align": "center", "margin": "0px", "--hover-color": "#42b9b4", "color": "#FFFFFF"}, "nav-link-selected": {"background-color": "#0a1e5c"}})

# ==============================================================================
# PAGE 1: GLOBAL OVERVIEW
# ==============================================================================
if page == "Global Overview":    
    styled_header(
        label="Policy Overview for Data-centric AI",
        description="An interactive dashboard for exploring the rhetoric of global AI and data governance."
    )

    st.markdown("""
        Welcome to the Policy Overview. This dashboard provides a semantic analysis of over 16,000 sentences from more than 300 national AI policy documents. As AI takes an increasingly important role in regulation, the language used in these policies reveals crucial insights into how nations approach data governance. Using a method supported by Large Language Models (LLMs), this research moves beyond keywords to classify the underlying tone and thematic emphasis, mapping the implicit values embedded in policy language.

        The visualisations are structured around five key data-centric themes. Each sentence from the policy documents has been classified into one of these "buckets" and assigned a score to indicate its rhetorical stance:

        * **Sovereignty Type:** Who controls data? A score of **+1** indicates individual control, **0** reflects firm/corporate control, and **-1** signifies state control.
        * **Data Flow Openness:** Are cross-border data flows encouraged? **+1** is supportive, **0** is conditional, and **-1** is restrictive.
        * **Security Justification:** Is national security used to justify data controls? **+1** indicates it's a minor rationale, **0** is balanced with other concerns, and **-1** means it is a dominant justification.
        * **Global Alignment:** Does the policy align with multilateral efforts (e.g., G7, OECD)? **+1** indicates a multilateral approach, **0** is conditional, and **-1** is unilateral.
        * **Transparency & Accountability:** How strong are oversight and reporting rules? **+1** indicates strong mechanisms, **0** is weak, and **-1** is opaque.

        As you explore the data, look for key trends identified in the research. There is a surprising global consensus around **Transparency & Accountability**, which often serves as a medium for expressing more contested views on sovereignty and security. Observe the downward trend in **Data Flow Openness** and the corresponding rise in **Security Justification**, suggesting an increasingly cautious global stance. Finally, challenge the "Three Digital Kingdoms" framework by comparing the rhetoric of the U.S., EU, and China, there may be more convergence than expected.
        """)

    unique_display_names = sorted(df['Display_Name'].unique())
    all_filter_options = (
        ["Global (All)"] +
        ["-------Blocs-------"] + sorted(bloc_mapping.keys()) +
        ["-------Income Groups-------"] + sorted(income_group_mapping.keys()) +
        ["-------Individual Countries-------"] + unique_display_names
    )
    
    selected_options = st.multiselect(
        'Filter by Country, Bloc, or Income Group:',
        options=all_filter_options,
        default=["Global (All)"]
    )

    if "Global (All)" in selected_options or not selected_options:
        plot_df = df.copy()
    else:
        countries_to_filter = set()
        for selection in selected_options:
            if selection in bloc_mapping:
                countries_to_filter.update(df[df['Country'].isin(bloc_mapping[selection])]['Display_Name'].unique())
            elif selection in income_group_mapping:
                countries_to_filter.update(df[df['Country'].isin(income_group_mapping[selection])]['Display_Name'].unique())
            elif selection in unique_display_names:
                countries_to_filter.add(selection)
        plot_df = df[df['Display_Name'].isin(countries_to_filter)]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution of Data-Centric Themes")
        fig, ax = plt.subplots(figsize=(10, 6))
        bucket_counts = plot_df['Bucket_label'].value_counts()
        sns.barplot(x=bucket_counts.index, y=bucket_counts.values, palette='viridis', order=bucket_counts.index, ax=ax)
        ax.set_title('Frequency of Themes in AI Policies', fontsize=16, weight='bold')
        ax.set_xlabel('Thematic Bucket', fontsize=12)
        ax.set_ylabel('Number of Policy Sentences', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        st.subheader("Distribution of Rhetorical Stance by Theme")
        plot_df.dropna(subset=['Bucket_label', 'Score_majority'], inplace=True)
        score_mapping_generic = {-1: 'Score -1', 0: 'Score 0', 1: 'Score +1'}
        plot_df['Score_label_generic'] = plot_df['Score_majority'].map(score_mapping_generic)
        if not plot_df.empty and 'Bucket_label' in plot_df and 'Score_label_generic' in plot_df:
            percentage_df = pd.crosstab(plot_df['Bucket_label'], plot_df['Score_label_generic'], normalize='index') * 100
            percentage_df = percentage_df.reindex(columns=list(score_mapping_generic.values()), fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 6))
            percentage_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', ax=ax)
            ax.set_title('Percentage of Rhetorical Stance by Theme', fontsize=16, weight='bold')
            ax.set_ylabel('Percentage of Sentences (%)', fontsize=12)
            ax.set_xlabel('')
            ax.set_ylim(0, 100)
            plt.xticks(rotation=45, ha='right')
            ax.legend(title='Stance', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No data available for the selected group.")

# ==============================================================================
# PAGE 2: TEMPORAL TRENDS
# ==============================================================================
elif page == "Temporal Trends":
    styled_header(label="Temporal Trends", description="Explore how the focus on different AI policy themes has evolved over time.")

    unique_display_names = sorted(df['Display_Name'].unique())
    all_filter_options = (
        ["Global (All)"] +
        ["-------Blocs-------"] + sorted(bloc_mapping.keys()) +
        ["-------Income Groups-------"] + sorted(income_group_mapping.keys()) +
        ["-------Individual Countries-------"] + unique_display_names
    )

    st.subheader("Filters")
    col1, col2 = st.columns(2)
    with col1:
        selected_options = st.multiselect(
            'Select Countries, Blocs, or Income Groups:',
            options=all_filter_options,
            default=["Global (All)"]
        )
    with col2:
        min_year, max_year = 2018, 2025
        start_year, end_year = st.slider('Select Year Range:', min_value=min_year, max_value=max_year, value=(2018, max_year))

    if "Global (All)" in selected_options or not selected_options:
        country_filtered_df = df.copy()
    else:
        countries_to_filter = set()
        for selection in selected_options:
            if selection in bloc_mapping:
                countries_to_filter.update(df[df['Country'].isin(bloc_mapping[selection])]['Display_Name'].unique())
            elif selection in income_group_mapping:
                countries_to_filter.update(df[df['Country'].isin(income_group_mapping[selection])]['Display_Name'].unique())
            elif selection in unique_display_names:
                countries_to_filter.add(selection)
        country_filtered_df = df[df['Display_Name'].isin(countries_to_filter)]

    final_filtered_df = country_filtered_df[(country_filtered_df['Year'] >= start_year) & (country_filtered_df['Year'] <= end_year)]
    final_filtered_df = final_filtered_df.dropna(subset=['Year', 'Bucket_label'])
    
    st.markdown("---")
    tab1, tab2 = st.tabs(["Evolution of Thematic Focus", "Evolution of Rhetorical Stances"])
    with tab1:
        timeline_df = pd.crosstab(final_filtered_df['Year'], final_filtered_df['Bucket_label'])
        timeline_df = timeline_df[timeline_df.sum(axis=1) > 20]
        if not timeline_df.empty:
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                timeline_df.plot(kind='area', stacked=True, ax=ax, colormap='viridis', linewidth=1)
                ax.set_title('Volume of Themes Over Time', fontsize=16, weight='bold')
                ax.set_ylabel('Number of Policy Sentences')
                ax.set_xlabel('Year')
                ax.legend().set_visible(False)
                st.pyplot(fig)
            with t_col2:
                percentage_timeline_df = timeline_df.div(timeline_df.sum(axis=1), axis=0) * 100
                fig, ax = plt.subplots(figsize=(10, 6))
                percentage_timeline_df.plot(kind='area', stacked=True, ax=ax, colormap='viridis', linewidth=1)
                ax.set_title('Share of Themes Over Time (%)', fontsize=16, weight='bold')
                ax.set_ylabel('Share of Policy Sentences (%)')
                ax.set_xlabel('Year')
                ax.set_ylim(0, 100)
                ax.legend(title='Theme', bbox_to_anchor=(1.02, 1), loc='upper left')
                st.pyplot(fig)
        else:
            st.warning("Not enough data for the selected filters to generate thematic focus plots.")
    with tab2:
        def get_stance_label(row): return label_mapping.get(row['Theme'], {}).get(row['Score_majority'])
        stance_df = final_filtered_df.copy()
        stance_df['Stance'] = stance_df.apply(get_stance_label, axis=1)
        stance_df.dropna(subset=['Stance'], inplace=True)
        all_themes = list(bucket_mapping.values())
        selected_themes = st.multiselect('Select themes to display:', options=all_themes, default=all_themes, key='stance_theme_selector')
        if not selected_themes:
            st.warning("Please select at least one theme to display.")
        elif not stance_df.empty:
            import math
            num_themes, ncols = len(selected_themes), 2 if len(selected_themes) > 1 else 1
            nrows = math.ceil(num_themes / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 6), sharex=True, squeeze=False)
            axes = axes.flatten()
            all_stances = [s for stances in label_mapping.values() for s in stances.values()]
            color_palette = sns.color_palette("viridis", len(set(all_stances)))
            stance_color_map = {s: c for s, c in zip(sorted(list(set(all_stances))), color_palette)}
            for i, theme in enumerate(selected_themes):
                ax = axes[i]
                theme_data = stance_df[stance_df['Theme'] == theme]
                if not theme_data.empty:
                    crosstab_df = pd.crosstab(theme_data['Year'], theme_data['Stance'])
                    stance_order = list(label_mapping[theme].values())
                    crosstab_df = crosstab_df.reindex(columns=stance_order, fill_value=0)
                    row_sums = crosstab_df.sum(axis=1)
                    if not row_sums.empty and not (row_sums == 0).all():
                        percentage_df = crosstab_df.div(row_sums, axis=0).fillna(0) * 100
                        percentage_df.plot(kind='area', stacked=True, ax=ax, linewidth=0.5, color=[stance_color_map.get(s, 'gray') for s in percentage_df.columns])
                ax.set_title(theme, fontsize=16, weight='bold')
                ax.set_ylabel('Share of Sentences (%)')
                ax.set_ylim(0, 100)
                ax.legend(title='Stance')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            for i in range(num_themes, len(axes)): axes[i].set_visible(False)
            fig.supxlabel('Year', fontsize=14)
            plt.tight_layout(rect=[0, 0.05, 1, 0.98])
            st.pyplot(fig)
        else:
            st.warning("No stance data available for the selected filters.")

# ==============================================================================
# PAGE 3: COMPARATIVE ANALYSIS
# ==============================================================================
elif page == "Comparative Analysis":
    styled_header(label="Comparative Analysis", description="Compare the AI policy stances and thematic focus of different countries and blocs.")
    
    unique_display_names = sorted(df['Display_Name'].unique())
    all_filter_options = (
        ["Global (All)"] +
        ["-------Blocs-------"] + sorted(bloc_mapping.keys()) +
        ["-------Income Groups-------"] + sorted(income_group_mapping.keys()) +
        ["-------Individual Countries-------"] + unique_display_names
    )
    
    tab1, tab2, tab3, tab4 = st.tabs(["Policy Profiles", "Thematic Focus Heatmap", "Global Stance Maps", "Theme vs. Theme Scatter"])
    with tab1:
        st.subheader("Comparative Policy Profiles")
        st.markdown("This chart shows the average rhetorical stance for each country across the five themes. A score of `+1` is more supportive and open, while `-1` is more restrictive. The default selection below shows all countries with a complete policy profile.")
        
        # Create a single list of all countries from the clusters
        all_cluster_countries = []
        for cluster_list in country_groups.values():
            all_cluster_countries.extend(cluster_list)
        # Remove duplicates and sort, if necessary
        default_selections = sorted(list(set(all_cluster_countries)))
        #default_selections = ['United States', 'Europe', 'China', 'Singapore', 'Nigeria', 'Japan', 'India']

        selected_options = st.multiselect(
            'Select Countries, Blocs, or Income Groups to Compare:',
            options=all_filter_options,
            default=default_selections
        )
        
        if "Global (All)" in selected_options or not selected_options:
            countries_to_filter = set(unique_display_names) # Select all countries for a global profile
        else:
            countries_to_filter = set()
            for selection in selected_options:
                if selection in bloc_mapping:
                    countries_to_filter.update(df[df['Country'].isin(bloc_mapping[selection])]['Display_Name'].unique())
                elif selection in income_group_mapping:
                    countries_to_filter.update(df[df['Country'].isin(income_group_mapping[selection])]['Display_Name'].unique())
                elif selection in unique_display_names:
                    countries_to_filter.add(selection)

        if countries_to_filter:
            filtered_df_tab1 = df[df['Display_Name'].isin(countries_to_filter)]
            df_comp = filtered_df_tab1.dropna(subset=['Theme', 'Display_Name', 'Score_majority'])
            avg_scores = df_comp.groupby(['Display_Name', 'Theme'])['Score_majority'].mean().unstack()
            avg_scores = avg_scores.reindex(columns=list(bucket_mapping.values()))
            plot_df = avg_scores
            
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = plt.get_cmap('turbo', len(plot_df))
            for i, (name, row) in enumerate(plot_df.iterrows()):
                ax.plot(row.index, row.values, marker='o', linewidth=2.5, alpha=0.9, color=colors(i), label=name)
            ax.legend(title='Country/Bloc', bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.set_title('Comparative Profile of National AI Policy Stances', fontsize=16, weight='bold')
            ax.set_ylabel('Average Score\n(Restrictive (-1) to Supportive (+1))')
            ax.set_ylim(-1.1, 1.1)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            st.pyplot(fig)
        else:
            st.info("Select one or more countries/blocs to display their profiles.")
    
    if countries_to_filter:
        filtered_df = df[df['Display_Name'].isin(countries_to_filter)]
    else:
        filtered_df = df.copy()

    df_comp = filtered_df.dropna(subset=['Theme', 'Display_Name', 'Score_majority'])
    avg_scores = df_comp.groupby(['Display_Name', 'Theme'])['Score_majority'].mean().unstack()
    avg_scores = avg_scores.reindex(columns=list(bucket_mapping.values()))

    with tab2:
        st.subheader("Thematic Focus Heatmap")
        st.markdown("This heatmap shows the percentage of a country's AI policy sentences that are dedicated to each theme for the selected group.")
        if not df_comp.empty:
            focus_counts = pd.crosstab(df_comp['Display_Name'], df_comp['Theme'])
            focus_percentage = focus_counts.div(focus_counts.sum(axis=1), axis=0) * 100
            fig, ax = plt.subplots(figsize=(12, max(8, len(focus_percentage) * 0.4)))
            sns.heatmap(focus_percentage, annot=True, fmt=".1f", cmap='viridis', linewidths=.5, ax=ax)
            ax.set_title('Thematic Focus of AI Policies', fontsize=16, weight='bold')
            ax.set_xlabel('Data-Centric Theme')
            ax.set_ylabel('Country / Bloc')
            plt.yticks(rotation=0)
            st.pyplot(fig)
        else:
            st.warning("No data available to generate a heatmap for the selected group.")

    with tab3:
        st.subheader("Global Stance Maps")
        st.markdown("This map visualises the average policy stance for a selected theme across the globe.")
        if world is not None:
            selected_theme_map = st.selectbox("Select a Theme to Map:", options=list(bucket_mapping.values()))
            @st.cache_data
            def get_iso3(iso2):
                try: return pycountry.countries.get(alpha_2=iso2).alpha_3
                except: return None
            df_map_theme_scores = filtered_df[filtered_df['Bucket_label'] == selected_theme_map].groupby('Country')['Score_majority'].mean().reset_index()
            df_map_theme_scores['iso_a3'] = df_map_theme_scores['Country'].apply(get_iso3)
            merged_map_data = world.merge(df_map_theme_scores, left_on='ISO_A3', right_on='iso_a3', how='left')
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            merged_map_data.plot(column='Score_majority', ax=ax, legend=True, cmap='RdYlBu_r', linewidth=0.8, edgecolor='0.8', missing_kwds={"color": "lightgrey", "label": "No Data"}, legend_kwds={'label': f"Average Score (-1 to +1)", 'orientation': "horizontal", 'pad': 0.05})
            ax.set_title(f'Global Stances on: {selected_theme_map}', fontdict={'fontsize': '16', 'fontweight': 'bold'})
            ax.set_axis_off()
            st.pyplot(fig)

    with tab4:
        st.subheader("Theme vs. Theme Scatter Plot")
        st.markdown("Position countries based on their average stance on two different themes simultaneously.")
        themes = list(bucket_mapping.values())
        col1, col2 = st.columns(2)
        with col1: theme_x = st.selectbox('Select X-Axis Theme:', options=themes, index=0)
        with col2: theme_y = st.selectbox('Select Y-Axis Theme:', options=themes, index=1)
        if not avg_scores.empty and theme_x != theme_y:
            plot_df_scatter = avg_scores[[theme_x, theme_y]].dropna()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(data=plot_df_scatter, x=theme_x, y=theme_y, s=150, alpha=0.7, edgecolor='w', linewidth=0.5, ax=ax)
            texts = [ax.text(row[theme_x], row[theme_y], country, fontsize=9) for country, row in plot_df_scatter.iterrows()]
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
            ax.set_title(f'{theme_y} vs. {theme_x}', fontsize=16, weight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.5)
            ax.axvline(0, color='black', linestyle='--', lw=0.8, alpha=0.5)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            st.pyplot(fig)
        elif theme_x == theme_y:
            st.warning("Please select two different themes to compare.")
        else:
            st.warning("No data available to generate a scatter plot for the selected group.")

# ==============================================================================
# PAGE 4: CLUSTER ANALYSIS
# ==============================================================================
elif page == "Cluster Analysis":
    styled_header(label="Country Cluster Analysis", description="Based on their policy profiles, countries can be grouped into distinct clusters with similar approaches to AI governance. This was achieved through K-means clustering")
    
    df_cluster = df.dropna(subset=['Theme', 'Display_Name', 'Score_majority'])
    avg_scores = df_cluster.groupby(['Display_Name', 'Theme'])['Score_majority'].mean().unstack()
    avg_scores = avg_scores.reindex(columns=list(bucket_mapping.values()))
    plot_df = avg_scores.dropna()
    fig, axes = plt.subplots(3, 2, figsize=(20, 24), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.suptitle('Country Profiles Based on Predefined Clusters', fontsize=24, weight='bold')
    key_colors = {'United States': 'blue', 'Europe': 'red', 'China': 'green'}
    for i, (title, country_list) in enumerate(country_groups.items()):
        ax = axes[i]
        valid_countries = [c for c in country_list if c in plot_df.index]
        if not valid_countries:
            ax.set_title(title, fontsize=18, weight='bold')
            ax.text(0.5, 0.5, 'No data for countries in this cluster', ha='center', va='center')
            ax.set_axis_off()
            continue
        group_df = plot_df.loc[valid_countries]
        other_colors = plt.get_cmap('tab20', len(group_df))
        texts = []
        for j, (name, row) in enumerate(group_df.iterrows()):
            color = key_colors.get(name, other_colors(j))
            linewidth = 4.0 if name in key_colors else 2.0
            alpha = 1.0 if name in key_colors else 0.8
            fontweight = 'bold' if name in key_colors else 'normal'
            ax.plot(row.index, row.values, marker='o', lw=linewidth, alpha=alpha, color=color, zorder=3 if name in key_colors else 2)
            texts.append(ax.text(len(row)-1, row.values[-1], f' {name}', color=color, fontweight=fontweight, fontsize=12))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        ax.set_title(title, fontsize=18, weight='bold')
        ax.set_ylim(-1.2, 1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.5)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
    for i in range(len(country_groups), len(axes)): axes[i].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)