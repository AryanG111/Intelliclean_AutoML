# streamlit_intelliclean_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import tempfile
import os
from pathlib import Path

# Import your enhanced preprocessing functions
# (You'll need to save the improved_intelliclean_automl.py in the same directory)
try:
    from intelliclean_automl import (
        complete_preprocessing_pipeline,
        preprocess_for_ml,
        compare_preprocessing_approaches,
        create_enhanced_test_data
    )
except ImportError:
    st.error("❌ Please ensure 'improved_intelliclean_automl.py' is in the same directory as this Streamlit app.")
    st.stop()

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="IntelliClean AutoML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================
@st.cache_data
def load_and_preview_data(file_path):
    """Load data and return preview info."""
    try:
        # Determine file type and load
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        return df, None
    except Exception as e:
        return None, str(e)

def create_data_overview_plots(df):
    """Create overview plots for the dataset."""
    
    # Missing data heatmap
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) > 0:
        fig_missing = px.bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            title="Missing Data by Column",
            labels={'x': 'Missing Count', 'y': 'Columns'}
        )
        fig_missing.update_layout(height=max(400, len(missing_data) * 25))
    else:
        fig_missing = None
    
    # Data types distribution
    dtype_counts = df.dtypes.value_counts()
    fig_dtypes = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index.astype(str),
        title="Data Types Distribution"
    )
    
    return fig_missing, fig_dtypes

def create_preprocessing_summary_plots(original_df, processed_df, characterization):
    """Create summary plots showing preprocessing results."""
    
    # Before/After comparison
    comparison_data = {
        'Metric': ['Rows', 'Columns', 'Missing Values', 'Duplicates'],
        'Before': [
            len(original_df),
            len(original_df.columns),
            original_df.isnull().sum().sum(),
            original_df.duplicated().sum()
        ],
        'After': [
            len(processed_df),
            len(processed_df.columns),
            processed_df.isnull().sum().sum(),
            processed_df.duplicated().sum()
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    fig_comparison = px.bar(
        comparison_df,
        x='Metric',
        y=['Before', 'After'],
        barmode='group',
        title="Before vs After Preprocessing"
    )
    
    return fig_comparison

def display_characterization_results(characterization):
    """Display characterization results in organized sections."""
    
    st.subheader("🔍 Data Characterization Results")
    
    # Dataset level info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Dataset Overview**")
        st.info(f"**Size Category:** {characterization['dataset_level']['size_category']}")
        st.info(f"**Sparsity Level:** {characterization['dataset_level']['sparsity_level']}")
        st.info(f"**Missing Pattern:** {characterization['dataset_level']['missing_data_pattern']}")
    
    with col2:
        st.markdown("**📈 Feature Breakdown**")
        col_cats = characterization['column_categories']
        st.metric("Numerical Features", len(col_cats['numerical']))
        st.metric("Categorical Features", len(col_cats['categorical']))
        st.metric("Binary Features", len(col_cats['binary']))
    
    with col3:
        st.markdown("**⚠️ Data Issues**")
        quality_issues = characterization['data_quality_issues']
        st.metric("Duplicates Found", quality_issues['duplicates'])
        st.metric("High Correlation Pairs", len(quality_issues['highly_correlated_features']))
        st.metric("Columns to Drop", len(col_cats['constant_columns']) + len(col_cats['high_missing_columns']))

def display_preprocessing_plan(preprocessing_plan):
    """Display the preprocessing plan in an organized way."""
    
    st.subheader("📋 Preprocessing Plan")
    
    # Data cleaning section
    cleaning = preprocessing_plan['data_cleaning']
    if any(len(v) > 0 if isinstance(v, list) else v for v in cleaning.values()):
        with st.expander("🧹 Data Cleaning Steps", expanded=True):
            if cleaning['drop_constant']:
                st.write("**Constant columns to drop:**", cleaning['drop_constant'])
            if cleaning['drop_high_missing']:
                st.write("**High missing columns to drop:**", cleaning['drop_high_missing'])
            if cleaning['drop_high_cardinality']:
                st.write("**High cardinality columns to drop:**", cleaning['drop_high_cardinality'])
            if cleaning['handle_duplicates']:
                st.write("**Will remove duplicate rows**")
    
    # Numerical processing
    num_strategies = preprocessing_plan['strategy_groups']['numerical']
    if any(len(v) > 0 for v in num_strategies.values()):
        with st.expander("🔢 Numerical Data Processing"):
            for strategy_type, strategies in num_strategies.items():
                if strategies:
                    st.write(f"**{strategy_type.title()} Strategies:**")
                    for strategy, columns in strategies.items():
                        if columns:
                            st.write(f"  • {strategy}: {columns}")
    
    # Categorical processing
    cat_strategies = preprocessing_plan['strategy_groups']['categorical']
    if any(len(v) > 0 for v in cat_strategies.values()):
        with st.expander("📝 Categorical Data Processing"):
            for strategy_type, strategies in cat_strategies.items():
                if strategies:
                    st.write(f"**{strategy_type.title()} Strategies:**")
                    for strategy, columns in strategies.items():
                        if columns:
                            st.write(f"  • {strategy}: {columns}")

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 IntelliClean AutoML</h1>', unsafe_allow_html=True)
    st.markdown("**Intelligent Data Preprocessing for Machine Learning**")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Upload File", "Use Sample Data"]
    )
    
    uploaded_file = None
    target_column = None
    
    if data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Upload your dataset for preprocessing"
        )
        
    elif data_source == "Use Sample Data":
        if st.sidebar.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                sample_file = create_enhanced_test_data()
                st.sidebar.success("Sample data created!")
                uploaded_file = sample_file
    
    # Processing options
    st.sidebar.header("🎛️ Processing Options")
    
    advanced_mode = st.sidebar.checkbox("Advanced Mode", help="Show detailed preprocessing steps")
    return_statistics = st.sidebar.checkbox("Include Statistics", value=True)
    
    # Main content area
    if uploaded_file is not None:
        
        # Handle file upload vs string path
        if isinstance(uploaded_file, str):
            file_path = uploaded_file
            df_preview, error = load_and_preview_data(file_path)
        else:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name
            df_preview, error = load_and_preview_data(file_path)
        
        if error:
            st.error(f"Error loading file: {error}")
            return
        
        if df_preview is not None:
            
            # Data preview section
            st.header("📊 Data Preview")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Dataset Overview")
                st.dataframe(df_preview.head(10), use_container_width=True)
                
                # Basic info
                buffer = io.StringIO()
                df_preview.info(buf=buffer)
                info_str = buffer.getvalue()
                
                with st.expander("📋 Dataset Info"):
                    st.text(info_str)
            
            with col2:
                st.subheader("Quick Stats")
                st.metric("Rows", f"{len(df_preview):,}")
                st.metric("Columns", len(df_preview.columns))
                st.metric("Missing Values", f"{df_preview.isnull().sum().sum():,}")
                st.metric("Duplicates", f"{df_preview.duplicated().sum():,}")
                
                # Data quality score
                missing_pct = (df_preview.isnull().sum().sum() / (len(df_preview) * len(df_preview.columns))) * 100
                duplicate_pct = (df_preview.duplicated().sum() / len(df_preview)) * 100
                quality_score = max(0, 100 - missing_pct - duplicate_pct)
                
                st.metric(
                    "Data Quality Score", 
                    f"{quality_score:.1f}%",
                    delta=f"{quality_score - 70:.1f}%" if quality_score > 70 else None
                )
            
            # Data visualization
            st.subheader("📈 Data Overview")
            
            try:
                fig_missing, fig_dtypes = create_data_overview_plots(df_preview)
                
                col1, col2 = st.columns(2)
                with col1:
                    if fig_missing:
                        st.plotly_chart(fig_missing, use_container_width=True)
                    else:
                        st.success("🎉 No missing data found!")
                
                with col2:
                    st.plotly_chart(fig_dtypes, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate overview plots: {str(e)}")
            
            # Target column selection
            st.sidebar.header("🎯 Target Variable")
            target_options = ["None"] + list(df_preview.columns)
            target_column = st.sidebar.selectbox(
                "Select target column (for ML):",
                target_options,
                help="Select the column you want to predict (optional)"
            )
            if target_column == "None":
                target_column = None
            
            # Processing button
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Start Preprocessing", type="primary", use_container_width=True):
                    
                    # Run preprocessing
                    with st.spinner("🔄 Running IntelliClean preprocessing..."):
                        try:
                            results = complete_preprocessing_pipeline(
                                file_path, 
                                target_column=target_column, 
                                return_stats=return_statistics
                            )
                            
                            # Store results in session state
                            st.session_state['preprocessing_results'] = results
                            st.session_state['original_df'] = df_preview
                            st.session_state['file_processed'] = True
                            
                            st.success("✅ Preprocessing completed successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error during preprocessing: {str(e)}")
                            st.exception(e)
    
    # Display results if available
    if 'file_processed' in st.session_state and st.session_state['file_processed']:
        display_preprocessing_results(target_column)

def display_preprocessing_results(target_column=None):
    """Display comprehensive preprocessing results."""
    
    results = st.session_state['preprocessing_results']
    original_df = st.session_state['original_df']
    processed_df = results['processed_data']
    characterization = results['characterization']
    preprocessing_plan = results['preprocessing_plan']
    
    st.markdown("---")
    st.header("🎉 Preprocessing Results")
    
    # Summary metrics
    st.subheader("📊 Transformation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Rows", 
            f"{len(processed_df):,}",
            delta=f"{len(processed_df) - len(original_df):+,}"
        )
    
    with col2:
        st.metric(
            "Features", 
            len(processed_df.columns),
            delta=f"{len(processed_df.columns) - len(original_df.columns):+}"
        )
    
    with col3:
        original_missing = original_df.isnull().sum().sum()
        processed_missing = processed_df.isnull().sum().sum()
        st.metric(
            "Missing Values", 
            f"{processed_missing:,}",
            delta=f"{processed_missing - original_missing:+,}"
        )
    
    with col4:
        memory_before = original_df.memory_usage(deep=True).sum() / 1024**2
        memory_after = processed_df.memory_usage(deep=True).sum() / 1024**2
        st.metric(
            "Memory (MB)", 
            f"{memory_after:.1f}",
            delta=f"{memory_after - memory_before:+.1f}"
        )
    
    # Characterization results
    if st.checkbox("Show Data Characterization", value=True):
        display_characterization_results(characterization)
    
    # Preprocessing plan
    if st.checkbox("Show Preprocessing Plan"):
        display_preprocessing_plan(preprocessing_plan)
    
    # Before/After comparison
    st.subheader("📈 Before vs After Comparison")
    
    try:
        fig_comparison = create_preprocessing_summary_plots(original_df, processed_df, characterization)
        st.plotly_chart(fig_comparison, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate comparison plots: {str(e)}")
    
    # Data preview tabs
    st.subheader("🔍 Data Preview")
    
    tab1, tab2, tab3 = st.tabs(["Original Data", "Processed Data", "Column Mapping"])
    
    with tab1:
        st.dataframe(original_df.head(20), use_container_width=True)
        
        # Original data stats
        if st.checkbox("Show Original Data Statistics"):
            st.write("**Descriptive Statistics:**")
            st.dataframe(original_df.describe(), use_container_width=True)
    
    with tab2:
        st.dataframe(processed_df.head(20), use_container_width=True)
        
        # Processed data stats
        if st.checkbox("Show Processed Data Statistics"):
            st.write("**Descriptive Statistics:**")
            st.dataframe(processed_df.describe(), use_container_width=True)
    
    with tab3:
        # Column mapping
        mapping_data = []
        
        original_cols = set(original_df.columns)
        processed_cols = set(processed_df.columns)
        dropped_cols = set(results.get('dropped_columns', []))
        
        # Original columns that remain
        for col in original_cols:
            if col in processed_cols:
                mapping_data.append({
                    'Original Column': col,
                    'Status': 'Kept',
                    'New Columns': col,
                    'Transformation': 'None or In-place'
                })
            elif col in dropped_cols:
                mapping_data.append({
                    'Original Column': col,
                    'Status': 'Dropped',
                    'New Columns': '-',
                    'Transformation': 'Removed'
                })
        
        # New columns created
        new_cols = processed_cols - original_cols
        for col in new_cols:
            base_col = col.split('_')[0] if '_' in col else 'Unknown'
            mapping_data.append({
                'Original Column': base_col,
                'Status': 'Engineered',
                'New Columns': col,
                'Transformation': 'Feature Engineering'
            })
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
    
    # Download section
    st.subheader("💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download processed data
        csv_buffer = io.StringIO()
        processed_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Processed Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name="processed_data.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download preprocessing report
        report_buffer = io.StringIO()
        report_buffer.write("IntelliClean AutoML Preprocessing Report\n")
        report_buffer.write("=" * 50 + "\n\n")
        
        report = results.get('preprocessing_report', {})
        if report:
            report_buffer.write(f"Original Shape: {report['summary']['original_shape']}\n")
            report_buffer.write(f"Processed Shape: {report['summary']['processed_shape']}\n")
            report_buffer.write(f"Columns Dropped: {report['summary']['columns_dropped']}\n")
            report_buffer.write(f"Features Created: {report['summary']['features_created']}\n\n")
            
            report_buffer.write("Transformations Applied:\n")
            trans = report['transformations_applied']
            report_buffer.write(f"  - Numerical: {trans['numerical_transformations']}\n")
            report_buffer.write(f"  - Categorical: {trans['categorical_encodings']}\n")
            report_buffer.write(f"  - Feature Engineering: {trans['feature_engineering']}\n")
        
        st.download_button(
            label="📋 Download Report (TXT)",
            data=report_buffer.getvalue(),
            file_name="preprocessing_report.txt",
            mime="text/plain"
        )
    
    with col3:
        # ML-ready data option
        if target_column:
            if st.button("🎯 Prepare for ML Training"):
                with st.spinner("Preparing ML-ready datasets..."):
                    try:
                        ml_data = preprocess_for_ml(file_path, target_column)
                        st.session_state['ml_ready_data'] = ml_data
                        st.success("✅ ML datasets prepared!")
                        
                        # Show ML data info
                        st.write("**ML Dataset Info:**")
                        st.write(f"Training set: {ml_data['X_train'].shape}")
                        st.write(f"Test set: {ml_data['X_test'].shape}")
                        st.write(f"Features: {len(ml_data['feature_names'])}")
                        
                    except Exception as e:
                        st.error(f"Error preparing ML data: {str(e)}")

# ==================== ADVANCED FEATURES ====================
def advanced_analysis_page():
    """Advanced analysis page for power users."""
    
    st.header("🔬 Advanced Analysis")
    
    if 'preprocessing_results' not in st.session_state:
        st.warning("Please run preprocessing first from the main page.")
        return
    
    results = st.session_state['preprocessing_results']
    processed_df = results['processed_data']
    characterization = results['characterization']
    
    # Feature correlation analysis
    st.subheader("🔗 Feature Correlation Analysis")
    
    numerical_cols = processed_df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        corr_matrix = processed_df[numerical_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # High correlation warnings
        high_corr_pairs = characterization['data_quality_issues']['highly_correlated_features']
        if high_corr_pairs:
            st.warning(f"⚠️ Found {len(high_corr_pairs)} highly correlated feature pairs that might indicate data leakage:")
            for pair in high_corr_pairs:
                st.write(f"  • {pair[0]} ↔ {pair[1]}")
    
    # Distribution analysis
    st.subheader("📊 Feature Distribution Analysis")
    
    if len(numerical_cols) > 0:
        selected_feature = st.selectbox("Select feature to analyze:", numerical_cols)
        
        if selected_feature:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig_hist = px.histogram(
                    processed_df, 
                    x=selected_feature,
                    title=f"Distribution of {selected_feature}",
                    nbins=30
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot
                fig_box = px.box(
                    processed_df, 
                    y=selected_feature,
                    title=f"Box Plot of {selected_feature}"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Feature statistics
            feature_stats = processed_df[selected_feature].describe()
            st.write("**Feature Statistics:**")
            st.dataframe(feature_stats.to_frame().T, use_container_width=True)

# ==================== APP NAVIGATION ====================
def main_app():
    """Main application with navigation."""
    
    # Navigation
    page = st.sidebar.selectbox(
        "🧭 Navigation",
        ["🏠 Home", "🔬 Advanced Analysis", "📚 Help & Documentation"]
    )
    
    if page == "🏠 Home":
        main()
    elif page == "🔬 Advanced Analysis":
        advanced_analysis_page()
    elif page == "📚 Help & Documentation":
        show_documentation()

def show_documentation():
    """Show help and documentation."""
    st.header("📚 IntelliClean AutoML Documentation")
    
    st.markdown("""
    ## 🎯 What is IntelliClean AutoML?
    
    IntelliClean AutoML is an intelligent data preprocessing pipeline that automatically:
    - Analyzes your data characteristics
    - Applies appropriate preprocessing strategies
    - Handles missing data intelligently
    - Encodes categorical variables optimally
    - Engineers useful features
    - Prepares data for machine learning
    
    ## 🚀 How to Use
    
    1. **Upload your data** or use sample data
    2. **Select target column** (if doing supervised ML)
    3. **Click "Start Preprocessing"**
    4. **Review results** and download processed data
    
    ## 🔧 Features
    
    ### Automatic Data Characterization
    - Dataset size and sparsity analysis
    - Column type detection (numerical, categorical, binary, datetime, text)
    - Missing data pattern analysis
    - Data quality assessment
    
    ### Intelligent Preprocessing
    - **Missing Data**: KNN imputation, mean/median/mode strategies
    - **Outliers**: Winsorization, clipping based on severity
    - **Scaling**: Standard, robust, or log transformation based on distribution
    - **Encoding**: One-hot, label, target, frequency encoding based on cardinality
    - **Feature Engineering**: Log transforms, datetime feature extraction
    
    ### Data Quality Improvements
    - Duplicate removal
    - Constant column removal
    - High correlation detection
    - Low variance feature removal
    
    ## 📋 Supported File Formats
    
    - CSV (.csv, .tsv, .txt)
    - Excel (.xlsx, .xls, .xlsm, .xlsb)
    - JSON (.json, .jsonl)
    - Parquet (.parquet, .pq)
    - Feather (.feather, .arrow)
    
    ## ⚠️ Best Practices
    
    1. **Always review the characterization results** before accepting preprocessing
    2. **Check the preprocessing plan** to understand what transformations will be applied
    3. **Validate your target column selection** for supervised learning
    4. **Review dropped columns** to ensure no important features are lost
    5. **Use the advanced analysis** to check for data leakage or correlation issues
    
    ## 🎯 Tips for Better ML Performance
    
    - **Provide a target column** for more intelligent preprocessing
    - **Review high correlation warnings** to avoid data leakage
    - **Check feature engineering results** - log transforms can significantly improve model performance
    - **Use the ML-ready export** for seamless integration with scikit-learn
    
    ## 🐛 Troubleshooting
    
    **File loading issues:**
    - Ensure file format is supported
    - Check for encoding issues (try UTF-8 or Latin-1)
    - Verify file is not corrupted
    
    **Preprocessing errors:**
    - Check for extremely high cardinality columns
    - Verify sufficient data for chosen strategies
    - Review target column selection
    
    **Memory issues:**
    - Use smaller samples for very large datasets
    - Consider feature selection for high-dimensional data
    """)

# ==================== RUN APP ====================
if __name__ == "__main__":
    # Initialize session state
    if 'file_processed' not in st.session_state:
        st.session_state['file_processed'] = False
    
    # Run main app
    main_app()

# ==================== ADDITIONAL UTILITY COMPONENTS ====================
def create_download_section():
    """Create download section with multiple format options."""
    
    if 'preprocessing_results' not in st.session_state:
        return
    
    results = st.session_state['preprocessing_results']
    processed_df = results['processed_data']
    
    st.subheader("💾 Download Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # CSV download
        csv_buffer = io.StringIO()
        processed_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "📄 CSV",
            csv_buffer.getvalue(),
            "processed_data.csv",
            "text/csv"
        )
    
    with col2:
        # Excel download
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            processed_df.to_excel(writer, sheet_name='Processed_Data', index=False)
        
        st.download_button(
            "📊 Excel",
            excel_buffer.getvalue(),
            "processed_data.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        # JSON download
        json_str = processed_df.to_json(orient='records', indent=2)
        st.download_button(
            "🗂️ JSON",
            json_str,
            "processed_data.json",
            "application/json"
        )
    
    with col4:
        # Parquet download
        parquet_buffer = io.BytesIO()
        processed_df.to_parquet(parquet_buffer, index=False)
        st.download_button(
            "🗜️ Parquet",
            parquet_buffer.getvalue(),
            "processed_data.parquet",
            "application/octet-stream"
        )