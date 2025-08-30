# improved_intelliclean_automl.py
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy.stats.mstats import winsorize
import category_encoders as ce  # pip install category-encoders
import warnings
warnings.filterwarnings('ignore')

# ==================== FILE TYPE CATEGORIZATION ====================
def categorize_dataset(file_path):
    """Categorizes a dataset file based on its extension."""
    path = Path(file_path)
    extension = path.suffix.lower()
    
    file_type_mapping = {
        '.csv': 'CSV', '.tsv': 'CSV', '.txt': 'CSV',
        '.xlsx': 'Excel', '.xls': 'Excel', '.xlsm': 'Excel', '.xlsb': 'Excel',
        '.json': 'JSON', '.jsonl': 'JSON',
        '.parquet': 'Parquet', '.pq': 'Parquet',
        '.feather': 'Feather', '.arrow': 'Feather',
    }
    
    return file_type_mapping.get(extension, 'Unknown')

def load_dataset(file_path, file_type):
    """Load a dataset based on its file type with better error handling."""
    try:
        if file_type == 'CSV':
            # Try to auto-detect delimiter and encoding
            try:
                return pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(file_path, encoding='latin-1')
        elif file_type == 'Excel':
            return pd.read_excel(file_path)
        elif file_type == 'JSON':
            return pd.read_json(file_path)
        elif file_type == 'Parquet':
            return pd.read_parquet(file_path)
        elif file_type == 'Feather':
            return pd.read_feather(file_path)
        else:
            # Try CSV as fallback with multiple encodings
            try:
                return pd.read_csv(file_path)
            except UnicodeDecodeError:
                return pd.read_csv(file_path, encoding='latin-1')
    except Exception as e:
        raise ValueError(f"Error loading {file_type} file: {str(e)}")

# ==================== ENHANCED DATA CHARACTERIZATION ====================
def characterize_data(df):
    """Enhanced data characterization with better ML-focused analysis."""
    characterization = {
        'dataset_level': {
            'size_category': '', 
            'sparsity_level': '', 
            'complexity_category': '',
            'feature_count': len(df.columns),
            'sample_count': len(df),
            'missing_data_pattern': ''
        },
        'column_categories': {
            'numerical': {}, 
            'categorical': {}, 
            'high_cardinality': {}, 
            'datetime': {}, 
            'text': {}, 
            'binary': {},
            'constant_columns': [], 
            'high_missing_columns': [],
            'low_variance_columns': []
        },
        'preprocessing_recommendations': {
            'imputation_strategy': {}, 
            'encoding_strategy': {}, 
            'scaling_strategy': {}, 
            'outlier_handling': {}, 
            'feature_selection_needed': False,
            'dimensionality_reduction_needed': False
        },
        'data_quality_issues': {
            'duplicates': 0,
            'potential_data_leakage': [],
            'highly_correlated_features': []
        }
    }
    
    n_rows, n_cols = df.shape
    
    # Enhanced dataset level characterization
    characterization = _characterize_dataset_level(df, characterization, n_rows, n_cols)
    
    # Enhanced column level characterization
    characterization = _characterize_columns_enhanced(df, characterization)
    
    # Data quality assessment
    characterization = _assess_data_quality(df, characterization)
    
    # Generate enhanced recommendations
    characterization = _generate_enhanced_recommendations(characterization, df)
    
    return characterization

def _characterize_dataset_level(df, characterization, n_rows, n_cols):
    """Enhanced dataset level characterization."""
    # Size category
    if n_rows > 100000:
        characterization['dataset_level']['size_category'] = 'LARGE'
    elif n_rows > 10000:
        characterization['dataset_level']['size_category'] = 'MEDIUM'
    else:
        characterization['dataset_level']['size_category'] = 'SMALL'
    
    # Sparsity level with more nuanced categorization
    missing_percentage = (df.isnull().sum().sum() / (n_rows * n_cols)) * 100
    if missing_percentage > 50:
        characterization['dataset_level']['sparsity_level'] = 'EXTREMELY_SPARSE'
    elif missing_percentage > 30:
        characterization['dataset_level']['sparsity_level'] = 'VERY_SPARSE'
    elif missing_percentage > 10:
        characterization['dataset_level']['sparsity_level'] = 'SPARSE'
    elif missing_percentage > 2:
        characterization['dataset_level']['sparsity_level'] = 'MODERATE'
    else:
        characterization['dataset_level']['sparsity_level'] = 'DENSE'
    
    # Missing data pattern analysis
    missing_cols = df.isnull().sum()
    if missing_cols.sum() == 0:
        characterization['dataset_level']['missing_data_pattern'] = 'NO_MISSING'
    elif (missing_cols > 0).sum() == 1:
        characterization['dataset_level']['missing_data_pattern'] = 'SINGLE_COLUMN'
    elif missing_cols.std() / missing_cols.mean() > 1.5:
        characterization['dataset_level']['missing_data_pattern'] = 'IRREGULAR'
    else:
        characterization['dataset_level']['missing_data_pattern'] = 'DISTRIBUTED'
    
    return characterization

def _characterize_columns_enhanced(df, characterization):
    """Enhanced column characterization with better type detection."""
    for col in df.columns:
        col_data = df[col]
        nunique = col_data.nunique()
        null_count = col_data.isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        
        # Check for constant or near-constant columns
        if nunique <= 1:
            characterization['column_categories']['constant_columns'].append(col)
            continue
        
        # Check for low variance columns (only 2-3 unique values in large datasets)
        if len(df) > 1000 and nunique <= 3:
            characterization['column_categories']['low_variance_columns'].append(col)
            
        # Check for high missing columns
        if null_percentage > 70:  # More strict threshold
            characterization['column_categories']['high_missing_columns'].append(col)
            continue
        
        # Enhanced type detection
        if pd.api.types.is_datetime64_any_dtype(col_data) or _is_datetime_string(col_data):
            characterization['column_categories']['datetime'][col] = {
                'null_percentage': null_percentage
            }
        elif pd.api.types.is_numeric_dtype(col_data):
            # Check if it's actually binary (0/1 or True/False)
            unique_vals = col_data.dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                characterization['column_categories']['binary'][col] = {
                    'null_percentage': null_percentage
                }
            else:
                outlier_info = _enhanced_outlier_analysis(col_data)
                characterization['column_categories']['numerical'][col] = {
                    'nunique': nunique, 
                    'null_percentage': null_percentage,
                    'has_outliers': outlier_info['has_outliers'],
                    'outlier_percentage': outlier_info['outlier_percentage'],
                    'skewness': stats.skew(col_data.dropna()),
                    'distribution_type': _detect_distribution(col_data),
                    'zero_inflation': (col_data == 0).sum() / len(col_data) > 0.1
                }
        else:
            # String/categorical analysis
            if _is_text_column(col_data):
                characterization['column_categories']['text'][col] = {
                    'avg_length': col_data.dropna().astype(str).str.len().mean(),
                    'null_percentage': null_percentage
                }
            elif nunique > 50 or nunique / len(df) > 0.5:  # High cardinality
                characterization['column_categories']['high_cardinality'][col] = {
                    'nunique': nunique, 
                    'null_percentage': null_percentage,
                    'cardinality_ratio': nunique / len(df)
                }
            else:  # Regular categorical
                characterization['column_categories']['categorical'][col] = {
                    'nunique': nunique, 
                    'null_percentage': null_percentage,
                    'is_ordinal': _detect_ordinal(col_data)
                }
    
    return characterization

def _is_datetime_string(series):
    """Check if string column contains datetime data."""
    sample = series.dropna().head(100)
    try:
        pd.to_datetime(sample, errors='raise')
        return True
    except:
        return False

def _is_text_column(series):
    """Check if column contains text data (vs categorical)."""
    sample = series.dropna().astype(str)
    avg_length = sample.str.len().mean()
    return avg_length > 20  # Assume text if average length > 20 chars

def _detect_ordinal(series):
    """Detect if categorical column might be ordinal."""
    unique_vals = series.dropna().unique()
    # Simple heuristic: check for common ordinal patterns
    ordinal_patterns = [
        ['low', 'medium', 'high'],
        ['bad', 'good', 'excellent'],
        ['small', 'medium', 'large'],
        ['never', 'rarely', 'sometimes', 'often', 'always']
    ]
    
    vals_lower = [str(v).lower() for v in unique_vals]
    for pattern in ordinal_patterns:
        if all(v in vals_lower for v in pattern):
            return True
    return False

def _enhanced_outlier_analysis(series):
    """Enhanced outlier detection with multiple methods."""
    clean_series = series.dropna()
    if len(clean_series) < 10:
        return {'has_outliers': False, 'outlier_percentage': 0}
    
    # IQR method
    Q1 = clean_series.quantile(0.25)
    Q3 = clean_series.quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = ((clean_series < (Q1 - 1.5 * IQR)) | (clean_series > (Q3 + 1.5 * IQR))).sum()
    
    # Z-score method
    z_scores = np.abs(stats.zscore(clean_series))
    zscore_outliers = (z_scores > 3).sum()
    
    # Use the more conservative estimate
    outlier_count = min(iqr_outliers, zscore_outliers)
    outlier_percentage = outlier_count / len(clean_series)
    
    return {
        'has_outliers': outlier_percentage > 0.05,
        'outlier_percentage': outlier_percentage
    }

def _detect_distribution(series):
    """Detect the likely distribution of numerical data."""
    clean_series = series.dropna()
    if len(clean_series) < 30:
        return 'UNKNOWN'
    
    # Test for normality
    _, p_normal = stats.normaltest(clean_series)
    if p_normal > 0.05:
        return 'NORMAL'
    
    # Check for exponential/log-normal
    if (clean_series > 0).all() and stats.skew(clean_series) > 1:
        return 'EXPONENTIAL'
    
    # Check for uniform
    _, p_uniform = stats.kstest(clean_series, 'uniform')
    if p_uniform > 0.05:
        return 'UNIFORM'
    
    return 'SKEWED'

def _assess_data_quality(df, characterization):
    """Assess data quality issues that affect ML performance."""
    # Check for duplicates
    characterization['data_quality_issues']['duplicates'] = df.duplicated().sum()
    
    # Check for potential data leakage (columns with too perfect correlation)
    numeric_cols = list(characterization['column_categories']['numerical'].keys())
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        characterization['data_quality_issues']['highly_correlated_features'] = high_corr_pairs
    
    return characterization

def _generate_enhanced_recommendations(characterization, df):
    """Generate enhanced preprocessing recommendations based on ML best practices."""
    rec = characterization['preprocessing_recommendations']
    
    # Enhanced numerical column recommendations
    for col, info in characterization['column_categories']['numerical'].items():
        # Better imputation strategy based on distribution and missing pattern
        if info['null_percentage'] > 20:
            rec['imputation_strategy'][col] = 'knn'  # KNN for high missing
        elif info['distribution_type'] == 'SKEWED' or info['has_outliers']:
            rec['imputation_strategy'][col] = 'median'
        else:
            rec['imputation_strategy'][col] = 'mean'
        
        # Better scaling strategy
        if info['distribution_type'] == 'EXPONENTIAL':
            rec['scaling_strategy'][col] = 'log_transform'
        elif info['has_outliers'] or info['distribution_type'] == 'SKEWED':
            rec['scaling_strategy'][col] = 'robust'
        else:
            rec['scaling_strategy'][col] = 'standard'
        
        # Enhanced outlier handling
        if info['outlier_percentage'] > 0.1:
            rec['outlier_handling'][col] = 'winsorize'
        elif info['outlier_percentage'] > 0.05:
            rec['outlier_handling'][col] = 'clip'
        else:
            rec['outlier_handling'][col] = 'none'
    
    # Enhanced categorical recommendations
    for col, info in characterization['column_categories']['categorical'].items():
        rec['imputation_strategy'][col] = 'mode' if info['null_percentage'] < 30 else 'constant'
        
        # Better encoding strategy based on cardinality and ordinality
        if info.get('is_ordinal', False):
            rec['encoding_strategy'][col] = 'ordinal'
        elif info['nunique'] <= 5:
            rec['encoding_strategy'][col] = 'one_hot'
        elif info['nunique'] <= 15:
            rec['encoding_strategy'][col] = 'target'  # Target encoding for medium cardinality
        else:
            rec['encoding_strategy'][col] = 'frequency'
    
    # Binary columns
    for col in characterization['column_categories']['binary']:
        rec['imputation_strategy'][col] = 'mode'
        rec['encoding_strategy'][col] = 'binary'  # Keep as is, just ensure 0/1
    
    # High cardinality recommendations
    for col, info in characterization['column_categories']['high_cardinality'].items():
        if info['cardinality_ratio'] > 0.8:  # Almost unique
            rec['encoding_strategy'][col] = 'drop'  # Usually not useful
        else:
            rec['imputation_strategy'][col] = 'constant'
            rec['encoding_strategy'][col] = 'target'  # Target encoding often better than frequency
    
    # Feature selection recommendations
    total_features = len(df.columns)
    if total_features > 50:
        rec['feature_selection_needed'] = True
    
    if total_features > 100 or len(df) < total_features * 10:
        rec['dimensionality_reduction_needed'] = True
    
    return characterization

# ==================== ENHANCED PREPROCESSING ROUTER ====================
def smart_preprocessing_router(df, characterization, target_column=None):
    """Enhanced preprocessing router with ML-focused strategies."""
    preprocessing_plan = {
        'data_cleaning': {
            'drop_constant': characterization['column_categories']['constant_columns'],
            'drop_high_missing': characterization['column_categories']['high_missing_columns'],
            'drop_low_variance': characterization['column_categories']['low_variance_columns'],
            'handle_duplicates': characterization['data_quality_issues']['duplicates'] > 0,
            'drop_high_cardinality': []
        },
        'strategy_groups': {
            'numerical': {'imputation': {}, 'scaling': {}, 'outlier': {}},
            'categorical': {'encoding': {}, 'imputation': {}},
            'binary': {'imputation': {}},
            'datetime': {'extraction': {}}
        },
        'feature_engineering': {
            'log_transform': [],
            'polynomial_features': [],
            'interaction_features': []
        },
        'post_processing': {
            'feature_selection': characterization['preprocessing_recommendations']['feature_selection_needed'],
            'dimensionality_reduction': characterization['preprocessing_recommendations']['dimensionality_reduction_needed']
        }
    }
    
    # Organize strategies more efficiently
    rec = characterization['preprocessing_recommendations']
    
    # Group strategies by type for batch processing
    _organize_strategies(rec, preprocessing_plan, characterization)
    
    # Identify columns to drop due to high cardinality
    for col, strategy in rec['encoding_strategy'].items():
        if strategy == 'drop':
            preprocessing_plan['data_cleaning']['drop_high_cardinality'].append(col)
    
    return preprocessing_plan

def _organize_strategies(rec, preprocessing_plan, characterization):
    """Organize preprocessing strategies by type for efficient batch processing."""
    
    # Numerical strategies
    for col, strategy in rec['imputation_strategy'].items():
        if col in characterization['column_categories']['numerical']:
            preprocessing_plan['strategy_groups']['numerical']['imputation'].setdefault(strategy, []).append(col)
    
    for col, strategy in rec['scaling_strategy'].items():
        if col in characterization['column_categories']['numerical']:
            if strategy == 'log_transform':
                preprocessing_plan['feature_engineering']['log_transform'].append(col)
            else:
                preprocessing_plan['strategy_groups']['numerical']['scaling'].setdefault(strategy, []).append(col)
    
    for col, strategy in rec['outlier_handling'].items():
        if col in characterization['column_categories']['numerical']:
            preprocessing_plan['strategy_groups']['numerical']['outlier'].setdefault(strategy, []).append(col)
    
    # Categorical strategies
    for col, strategy in rec['encoding_strategy'].items():
        if col in characterization['column_categories']['categorical']:
            preprocessing_plan['strategy_groups']['categorical']['encoding'].setdefault(strategy, []).append(col)
        elif col in characterization['column_categories']['high_cardinality'] and strategy != 'drop':
            preprocessing_plan['strategy_groups']['categorical']['encoding'].setdefault(strategy, []).append(col)
    
    for col, strategy in rec['imputation_strategy'].items():
        if col in characterization['column_categories']['categorical'] or col in characterization['column_categories']['high_cardinality']:
            preprocessing_plan['strategy_groups']['categorical']['imputation'].setdefault(strategy, []).append(col)
    
    # Binary strategies
    for col in characterization['column_categories']['binary']:
        strategy = rec['imputation_strategy'].get(col, 'mode')
        preprocessing_plan['strategy_groups']['binary']['imputation'].setdefault(strategy, []).append(col)

# ==================== ENHANCED ADAPTIVE PREPROCESSOR ====================
def adaptive_preprocessor(df, preprocessing_plan, target_column=None):
    """Enhanced adaptive preprocessing with better ML practices."""
    processed_df = df.copy()
    dropped_columns = []
    
    print("🧹 Starting data cleaning...")
    
    # Step 1: Handle duplicates
    if preprocessing_plan['data_cleaning']['handle_duplicates']:
        before_count = len(processed_df)
        processed_df = processed_df.drop_duplicates()
        print(f"   Removed {before_count - len(processed_df)} duplicate rows")
    
    # Step 2: Collect all columns to drop
    cols_to_drop = list(set(
        preprocessing_plan['data_cleaning']['drop_constant'] + 
        preprocessing_plan['data_cleaning']['drop_high_missing'] +
        preprocessing_plan['data_cleaning']['drop_low_variance'] +
        preprocessing_plan['data_cleaning']['drop_high_cardinality']
    ))
    
    # Don't drop target column if specified
    if target_column and target_column in cols_to_drop:
        cols_to_drop.remove(target_column)
    
    dropped_columns.extend(cols_to_drop)
    
    # Step 3: Apply feature engineering first (before imputation)
    processed_df = _apply_feature_engineering(processed_df, preprocessing_plan, cols_to_drop)
    
    # Step 4: Apply preprocessing strategies
    processed_df = _process_numerical_data_enhanced(processed_df, preprocessing_plan, cols_to_drop)
    processed_df = _process_categorical_data_enhanced(processed_df, preprocessing_plan, cols_to_drop)
    processed_df = _process_binary_data(processed_df, preprocessing_plan, cols_to_drop)
    processed_df = _process_datetime_data(processed_df, preprocessing_plan, cols_to_drop)
    
    # Step 5: Drop identified columns
    if cols_to_drop:
        existing_cols_to_drop = [col for col in cols_to_drop if col in processed_df.columns]
        if existing_cols_to_drop:
            print(f"🗑️ Dropping columns: {existing_cols_to_drop}")
            processed_df = processed_df.drop(columns=existing_cols_to_drop)
    
    # Step 6: Post-processing
    processed_df = _apply_post_processing(processed_df, preprocessing_plan, target_column)
    
    return processed_df, dropped_columns

def _apply_feature_engineering(df, preprocessing_plan, cols_to_drop):
    """Apply feature engineering transformations."""
    processed_df = df.copy()
    
    # Log transform for skewed numerical data
    for col in preprocessing_plan['feature_engineering']['log_transform']:
        if col not in cols_to_drop and col in processed_df.columns:
            # Add small constant to handle zeros
            processed_df[f'{col}_log'] = np.log1p(processed_df[col].fillna(0))
            print(f"   Applied log transform to {col}")
    
    return processed_df

def _process_numerical_data_enhanced(df, preprocessing_plan, cols_to_drop):
    """Enhanced numerical data processing."""
    processed_df = df.copy()
    strategies = preprocessing_plan['strategy_groups']['numerical']
    
    # Enhanced imputation
    for strategy, columns in strategies['imputation'].items():
        if columns:
            valid_columns = [col for col in columns if col not in cols_to_drop and col in processed_df.columns]
            if not valid_columns:
                continue
                
            if strategy == 'knn':
                # Use KNN imputation for better results with high missing data
                imputer = KNNImputer(n_neighbors=5)
                processed_df[valid_columns] = imputer.fit_transform(processed_df[valid_columns])
                print(f"   Applied KNN imputation to: {valid_columns}")
            elif strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
                processed_df[valid_columns] = imputer.fit_transform(processed_df[valid_columns])
                print(f"   Applied mean imputation to: {valid_columns}")
            elif strategy == 'median':
                imputer = SimpleImputer(strategy='median')
                processed_df[valid_columns] = imputer.fit_transform(processed_df[valid_columns])
                print(f"   Applied median imputation to: {valid_columns}")
    
    # Enhanced outlier handling
    for strategy, columns in strategies['outlier'].items():
        valid_columns = [col for col in columns if col not in cols_to_drop and col in processed_df.columns]
        if strategy == 'winsorize' and valid_columns:
            for col in valid_columns:
                processed_df[col] = winsorize(processed_df[col], limits=[0.05, 0.05])
            print(f"   Applied winsorization to: {valid_columns}")
        elif strategy == 'clip' and valid_columns:
            for col in valid_columns:
                Q1 = processed_df[col].quantile(0.25)
                Q3 = processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
            print(f"   Applied clipping to: {valid_columns}")
    
    # Enhanced scaling
    for strategy, columns in strategies['scaling'].items():
        valid_columns = [col for col in columns if col not in cols_to_drop and col in processed_df.columns]
        if not valid_columns:
            continue
            
        if strategy == 'standard':
            scaler = StandardScaler()
        elif strategy == 'robust':
            scaler = RobustScaler()
        elif strategy == 'minmax':
            scaler = MinMaxScaler()
        else:
            continue
            
        processed_df[valid_columns] = scaler.fit_transform(processed_df[valid_columns])
        print(f"   Applied {strategy} scaling to: {valid_columns}")
    
    return processed_df

def _process_categorical_data_enhanced(df, preprocessing_plan, cols_to_drop):
    """Enhanced categorical data processing with better encoding strategies."""
    processed_df = df.copy()
    strategies = preprocessing_plan['strategy_groups']['categorical']
    
    # Enhanced imputation
    for strategy, columns in strategies['imputation'].items():
        valid_columns = [col for col in columns if col not in cols_to_drop and col in processed_df.columns]
        if not valid_columns:
            continue
            
        if strategy == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
            processed_df[valid_columns] = imputer.fit_transform(processed_df[valid_columns])
            print(f"   Applied mode imputation to: {valid_columns}")
        elif strategy == 'constant':
            imputer = SimpleImputer(strategy='constant', fill_value='MISSING')
            processed_df[valid_columns] = imputer.fit_transform(processed_df[valid_columns])
            print(f"   Applied constant imputation to: {valid_columns}")
    
    # Enhanced encoding
    for strategy, columns in strategies['encoding'].items():
        valid_columns = [col for col in columns if col not in cols_to_drop and col in processed_df.columns]
        if not valid_columns:
            continue
            
        if strategy == 'one_hot':
            processed_df = pd.get_dummies(processed_df, columns=valid_columns, prefix=valid_columns, drop_first=True)
            print(f"   Applied one-hot encoding to: {valid_columns}")
        elif strategy == 'label':
            for col in valid_columns:
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            print(f"   Applied label encoding to: {valid_columns}")
        elif strategy == 'frequency':
            for col in valid_columns:
                freq_encoding = processed_df[col].value_counts().to_dict()
                processed_df[col] = processed_df[col].map(freq_encoding)
            print(f"   Applied frequency encoding to: {valid_columns}")
        elif strategy == 'target':
            # Note: Target encoding requires target variable - placeholder for now
            print(f"   Target encoding planned for: {valid_columns} (requires target variable)")
        elif strategy == 'ordinal':
            # Simple ordinal encoding - could be enhanced with proper ordering
            for col in valid_columns:
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            print(f"   Applied ordinal encoding to: {valid_columns}")
    
    return processed_df

def _process_binary_data(df, preprocessing_plan, cols_to_drop):
    """Process binary columns."""
    processed_df = df.copy()
    strategies = preprocessing_plan['strategy_groups']['binary']
    
    for strategy, columns in strategies['imputation'].items():
        valid_columns = [col for col in columns if col not in cols_to_drop and col in processed_df.columns]
        if valid_columns and strategy == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
            processed_df[valid_columns] = imputer.fit_transform(processed_df[valid_columns])
            print(f"   Applied mode imputation to binary columns: {valid_columns}")
    
    return processed_df

def _process_datetime_data(df, preprocessing_plan, cols_to_drop):
    """Process datetime columns by extracting useful features."""
    processed_df = df.copy()
    
    datetime_cols = [col for col in preprocessing_plan['strategy_groups']['datetime'].get('extraction', {}) 
                    if col not in cols_to_drop and col in processed_df.columns]
    
    for col in datetime_cols:
        if col in processed_df.columns:
            # Convert to datetime
            processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
            
            # Extract features
            processed_df[f'{col}_year'] = processed_df[col].dt.year
            processed_df[f'{col}_month'] = processed_df[col].dt.month
            processed_df[f'{col}_dayofweek'] = processed_df[col].dt.dayofweek
            processed_df[f'{col}_quarter'] = processed_df[col].dt.quarter
            
            # Drop original datetime column
            processed_df = processed_df.drop(columns=[col])
            print(f"   Extracted datetime features from {col}")
    
    return processed_df

def _apply_post_processing(df, preprocessing_plan, target_column):
    """Apply post-processing steps like feature selection."""
    processed_df = df.copy()
    
    # Remove low variance features
    if preprocessing_plan['post_processing']['feature_selection']:
        # Only apply to numerical columns to avoid issues with categorical
        numerical_cols = processed_df.select_dtypes(include=[np.number]).columns
        if target_column:
            numerical_cols = [col for col in numerical_cols if col != target_column]
        
        if len(numerical_cols) > 0:
            selector = VarianceThreshold(threshold=0.01)  # Remove features with very low variance
            try:
                mask = selector.fit(processed_df[numerical_cols]).get_support()
                cols_to_keep = numerical_cols[mask]
                cols_to_remove = numerical_cols[~mask]
                
                if len(cols_to_remove) > 0:
                    processed_df = processed_df.drop(columns=cols_to_remove)
                    print(f"   Removed low variance features: {list(cols_to_remove)}")
            except:
                print("   Skipped variance-based feature selection due to data issues")
    
    return processed_df

# ==================== ENHANCED MAIN PIPELINE ====================
def complete_preprocessing_pipeline(file_path, target_column=None, return_stats=False):
    """Complete enhanced pipeline from file to preprocessed data."""
    print("🚀 Starting Enhanced IntelliClean AutoML Pipeline")
    print("=" * 60)
    
    try:
        # 1. Load data
        print("📂 Loading data...")
        file_type = categorize_dataset(file_path)
        df = load_dataset(file_path, file_type)
        print(f"   Loaded {file_type} file with shape: {df.shape}")
        
        # 2. Enhanced data characterization
        print("🔍 Characterizing data...")
        characterization = characterize_data(df)
        print(f"   Dataset size: {characterization['dataset_level']['size_category']}")
        print(f"   Data sparsity: {characterization['dataset_level']['sparsity_level']}")
        print(f"   Missing pattern: {characterization['dataset_level']['missing_data_pattern']}")
        
        # Print data quality issues
        quality_issues = characterization['data_quality_issues']
        if quality_issues['duplicates'] > 0:
            print(f"   ⚠️ Found {quality_issues['duplicates']} duplicate rows")
        if quality_issues['highly_correlated_features']:
            print(f"   ⚠️ Found {len(quality_issues['highly_correlated_features'])} highly correlated feature pairs")
        
        # 3. Create enhanced preprocessing plan
        print("📋 Creating preprocessing plan...")
        preprocessing_plan = smart_preprocessing_router(df, characterization, target_column)
        
        # 4. Apply enhanced adaptive preprocessing
        print("⚙️ Applying adaptive preprocessing...")
        processed_df, dropped_columns = adaptive_preprocessor(df, preprocessing_plan, target_column)
        
        print("✅ Preprocessing completed successfully!")
        print(f"   Original shape: {df.shape}")
        print(f"   Processed shape: {processed_df.shape}")
        print(f"   Columns dropped: {len(dropped_columns)}")
        
        # Generate preprocessing report
        report = _generate_preprocessing_report(df, processed_df, characterization, preprocessing_plan, dropped_columns)
        
        results = {
            'original_data': df,
            'processed_data': processed_df,
            'characterization': characterization,
            'preprocessing_plan': preprocessing_plan,
            'dropped_columns': dropped_columns,
            'preprocessing_report': report
        }
        
        if return_stats:
            results['statistics'] = _calculate_preprocessing_stats(df, processed_df)
        
        return results
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        raise

def _generate_preprocessing_report(original_df, processed_df, characterization, preprocessing_plan, dropped_columns):
    """Generate a comprehensive preprocessing report."""
    report = {
        'summary': {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'columns_dropped': len(dropped_columns),
            'features_created': processed_df.shape[1] - original_df.shape[1] + len(dropped_columns)
        },
        'transformations_applied': {
            'numerical_transformations': len([col for cols in preprocessing_plan['strategy_groups']['numerical']['scaling'].values() for col in cols]),
            'categorical_encodings': len([col for cols in preprocessing_plan['strategy_groups']['categorical']['encoding'].values() for col in cols]),
            'feature_engineering': len(preprocessing_plan['feature_engineering']['log_transform'])
        },
        'data_quality_improvements': {
            'duplicates_removed': characterization['data_quality_issues']['duplicates'],
            'constant_columns_removed': len(preprocessing_plan['data_cleaning']['drop_constant']),
            'high_missing_columns_removed': len(preprocessing_plan['data_cleaning']['drop_high_missing'])
        }
    }
    return report

def _calculate_preprocessing_stats(original_df, processed_df):
    """Calculate statistics comparing original vs processed data."""
    stats = {
        'missing_data_reduction': {
            'original_missing': original_df.isnull().sum().sum(),
            'processed_missing': processed_df.isnull().sum().sum()
        },
        'feature_count_change': {
            'original_features': original_df.shape[1],
            'processed_features': processed_df.shape[1]
        }
    }
    return stats

# ==================== ENHANCED TEST FUNCTION ====================
def create_enhanced_test_data():
    """Create more realistic test data with various ML challenges."""
    np.random.seed(42)
    n_rows = 1000
    
    # Create realistic test data
    test_data = pd.DataFrame({
        # Numerical with different characteristics
        'age': np.random.normal(35, 12, n_rows),
        'income': np.random.lognormal(10, 1, n_rows),  # Log-normal distribution
        'score': np.random.beta(2, 5, n_rows) * 100,  # Skewed data
        'rating': np.random.choice([1, 2, 3, 4, 5], n_rows, p=[0.1, 0.15, 0.3, 0.35, 0.1]),
        
        # Categorical with different cardinalities
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Sales'], n_rows),
        'country': np.random.choice([f'Country_{i}' for i in range(50)], n_rows),  # High cardinality
        
        # Binary
        'is_active': np.random.choice([0, 1], n_rows, p=[0.3, 0.7]),
        'has_premium': np.random.choice([True, False], n_rows),
        
        # Text-like
        'product_id': [f'PROD_{i:06d}' for i in np.random.randint(1, 10000, n_rows)],
        
        # Constant column
        'constant_col': [1] * n_rows,
        
        # Datetime
        'created_date': pd.date_range('2020-01-01', periods=n_rows, freq='D')
    })
    
    # Add realistic missing data patterns
    # Random missing
    test_data.loc[np.random.choice(n_rows, int(n_rows * 0.1), False), 'age'] = np.nan
    test_data.loc[np.random.choice(n_rows, int(n_rows * 0.05), False), 'income'] = np.nan
    
    # Systematic missing (missing not at random)
    high_income_mask = test_data['income'] > test_data['income'].quantile(0.9)
    test_data.loc[high_income_mask, 'department'] = np.nan  # High earners don't report department
    
    # High missing column
    test_data.loc[np.random.choice(n_rows, int(n_rows * 0.8), False), 'score'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(n_rows, 20, False)
    test_data.loc[outlier_indices, 'age'] = np.random.uniform(100, 120, 20)
    
    # Add some duplicates
    duplicate_rows = test_data.sample(50)
    test_data = pd.concat([test_data, duplicate_rows], ignore_index=True)
    
    # Save test data
    test_data.to_csv('enhanced_test_dataset.csv', index=False)
    print("📁 Created enhanced_test_dataset.csv")
    return 'enhanced_test_dataset.csv'

# ==================== VALIDATION AND COMPARISON ====================
def validate_preprocessing(original_df, processed_df, characterization):
    """Validate that preprocessing was applied correctly."""
    validation_results = {
        'missing_data_handled': processed_df.isnull().sum().sum() == 0,
        'no_constant_columns': processed_df.nunique().min() > 1,
        'all_numeric_scaled': True,  # Will check this
        'categorical_encoded': True,  # Will check this
        'issues_found': []
    }
    
    # Check if all categorical columns are properly encoded
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object':
            validation_results['categorical_encoded'] = False
            validation_results['issues_found'].append(f"Column '{col}' still contains object dtype")
    
    # Check for infinite values
    if np.isinf(processed_df.select_dtypes(include=[np.number])).any().any():
        validation_results['issues_found'].append("Infinite values detected")
    
    return validation_results

def compare_preprocessing_approaches(file_path, target_column=None):
    """Compare original vs enhanced preprocessing approaches."""
    print("🔄 Comparing preprocessing approaches...")
    
    # Load data
    file_type = categorize_dataset(file_path)
    df = load_dataset(file_path, file_type)
    
    # Original approach (simplified)
    print("\n📊 Original approach results:")
    original_char = characterize_data(df)
    original_plan = smart_preprocessing_router(df, original_char, target_column)
    
    # Enhanced approach
    print("\n🚀 Enhanced approach results:")
    enhanced_results = complete_preprocessing_pipeline(file_path, target_column, return_stats=True)
    
    # Validation
    validation = validate_preprocessing(df, enhanced_results['processed_data'], enhanced_results['characterization'])
    
    print("\n📋 Validation Results:")
    print(f"   Missing data handled: {validation['missing_data_handled']}")
    print(f"   No constant columns: {validation['no_constant_columns']}")
    print(f"   Categorical data encoded: {validation['categorical_encoded']}")
    if validation['issues_found']:
        print(f"   ⚠️ Issues found: {validation['issues_found']}")
    else:
        print("   ✅ No issues found")
    
    return {
        'enhanced_results': enhanced_results,
        'validation': validation
    }

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("Enhanced IntelliClean AutoML Pipeline")
    print("=" * 60)
    
    # Create enhanced test data
    test_file = create_enhanced_test_data()
    
    # Run comparison
    comparison_results = compare_preprocessing_approaches(test_file)
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced Pipeline Results Summary:")
    print("=" * 60)
    
    results = comparison_results['enhanced_results']
    
    print(f"Original data shape: {results['original_data'].shape}")
    print(f"Processed data shape: {results['processed_data'].shape}")
    print(f"Columns dropped: {len(results['dropped_columns'])}")
    
    print("\n📊 Sample of processed data:")
    print(results['processed_data'].head())
    
    print("\n💡 Key improvements made:")
    report = results['preprocessing_report']
    print(f"   • {report['data_quality_improvements']['duplicates_removed']} duplicates removed")
    print(f"   • {report['data_quality_improvements']['constant_columns_removed']} constant columns removed")
    print(f"   • {report['transformations_applied']['numerical_transformations']} numerical transformations applied")
    print(f"   • {report['transformations_applied']['categorical_encodings']} categorical encodings applied")
    
    if results.get('statistics'):
        stats = results['statistics']
        print(f"\n📈 Data Quality Improvement:")
        print(f"   • Missing values: {stats['missing_data_reduction']['original_missing']} → {stats['missing_data_reduction']['processed_missing']}")
        print(f"   • Feature count: {stats['feature_count_change']['original_features']} → {stats['feature_count_change']['processed_features']}")

# ==================== UTILITY FUNCTIONS FOR INTEGRATION ====================
def preprocess_for_ml(file_path, target_column=None, test_size=0.2, random_state=42):
    """Convenience function for ML pipeline integration."""
    from sklearn.model_selection import train_test_split
    
    # Run preprocessing
    results = complete_preprocessing_pipeline(file_path, target_column, return_stats=True)
    processed_df = results['processed_data']
    
    if target_column and target_column in processed_df.columns:
        # Split features and target
        X = processed_df.drop(columns=[target_column])
        y = processed_df[target_column]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) < 20 else None
        )
        
        return {
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'feature_names': list(X.columns), 'preprocessing_results': results
        }
    else:
        return {
            'processed_data': processed_df,
            'feature_names': list(processed_df.columns),
            'preprocessing_results': results
        }

def get_feature_importance_ready_data(file_path, target_column):
    """Prepare data specifically for feature importance analysis."""
    ml_ready = preprocess_for_ml(file_path, target_column)
    
    # Additional feature engineering for interpretability
    X_train = ml_ready['X_train']
    
    # Create feature importance mapping
    feature_mapping = {
        'original_features': [],
        'engineered_features': [],
        'encoded_features': []
    }
    
    for col in X_train.columns:
        if '_log' in col:
            feature_mapping['engineered_features'].append(col)
        elif any(prefix in col for prefix in ['_year', '_month', '_dayofweek', '_quarter']):
            feature_mapping['engineered_features'].append(col)
        elif '_' in col and any(cat in col for cat in ['category', 'department', 'country']):
            feature_mapping['encoded_features'].append(col)
        else:
            feature_mapping['original_features'].append(col)
    
    ml_ready['feature_mapping'] = feature_mapping
    return ml_ready 