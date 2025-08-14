import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import norm, t, chi2, f
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit
st.set_page_config(
    page_title="Statistics Learning Hub",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin: 1rem 0;
    }
    .definition-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .simple-explanation {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .formula-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_datasets():
    """Load and prepare sample datasets for analysis"""
    datasets = {}
    
    try:
        # Load Iris dataset
        iris = load_iris()
        datasets['iris'] = pd.DataFrame(iris.data, columns=iris.feature_names)
        datasets['iris']['species'] = [iris.target_names[i] for i in iris.target]
        
        # Load Wine dataset
        wine = load_wine()
        datasets['wine'] = pd.DataFrame(wine.data, columns=wine.feature_names)
        datasets['wine']['wine_class'] = wine.target
        
        # Load Breast Cancer dataset
        cancer = load_breast_cancer()
        datasets['cancer'] = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        datasets['cancer']['diagnosis'] = [cancer.target_names[i] for i in cancer.target]
        
        # Load Diabetes dataset
        diabetes = load_diabetes()
        datasets['diabetes'] = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        datasets['diabetes']['target'] = diabetes.target
        
        # Load Seaborn datasets
        try:
            datasets['tips'] = sns.load_dataset('tips')
        except:
            # Create synthetic tips dataset
            np.random.seed(42)
            n = 244
            datasets['tips'] = pd.DataFrame({
                'total_bill': np.random.normal(20, 8, n),
                'tip': np.random.normal(3, 1.5, n),
                'sex': np.random.choice(['Male', 'Female'], n),
                'smoker': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
                'day': np.random.choice(['Thur', 'Fri', 'Sat', 'Sun'], n),
                'time': np.random.choice(['Lunch', 'Dinner'], n),
                'size': np.random.choice([1, 2, 3, 4, 5, 6], n)
            })
        
        try:
            datasets['titanic'] = sns.load_dataset('titanic')
        except:
            # Create synthetic titanic dataset
            np.random.seed(42)
            n = 891
            datasets['titanic'] = pd.DataFrame({
                'survived': np.random.choice([0, 1], n),
                'pclass': np.random.choice([1, 2, 3], n),
                'sex': np.random.choice(['male', 'female'], n),
                'age': np.random.normal(30, 12, n),
                'sibsp': np.random.choice([0, 1, 2, 3, 4], n),
                'parch': np.random.choice([0, 1, 2, 3], n),
                'fare': np.random.exponential(15, n),
                'embarked': np.random.choice(['C', 'Q', 'S'], n)
            })
            
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
        
    return datasets

def calculate_statistics(data):
    """Calculate comprehensive descriptive statistics"""
    try:
        data_clean = pd.Series(data).dropna()
        if len(data_clean) == 0:
            return {}
        
        stats_dict = {
            'count': len(data_clean),
            'mean': np.mean(data_clean),
            'median': np.median(data_clean),
            'mode': stats.mode(data_clean, keepdims=True).mode[0] if len(data_clean) > 0 else np.nan,
            'std': np.std(data_clean, ddof=1),
            'var': np.var(data_clean, ddof=1),
            'min': np.min(data_clean),
            'max': np.max(data_clean),
            'range': np.max(data_clean) - np.min(data_clean),
            'q1': np.percentile(data_clean, 25),
            'q3': np.percentile(data_clean, 75),
            'iqr': np.percentile(data_clean, 75) - np.percentile(data_clean, 25),
            'skewness': stats.skew(data_clean),
            'kurtosis': stats.kurtosis(data_clean)
        }
        
        if stats_dict['mean'] != 0:
            stats_dict['cv'] = (stats_dict['std'] / stats_dict['mean']) * 100
        else:
            stats_dict['cv'] = 0
            
        return stats_dict
    except Exception as e:
        st.error(f"Error calculating statistics: {str(e)}")
        return {}

# ===============================
# 1. DESCRIPTIVE STATISTICS
# ===============================

def show_descriptive_statistics():
    st.markdown('<h1 class="main-header">Descriptive Statistics</h1>', unsafe_allow_html=True)
    
    descriptive_topics = [
        "Data Types Explorer",
        "Central Tendency", 
        "Dispersion Measures",
        "Quantiles and Percentiles"
    ]
    
    selected_topic = st.selectbox("Select Topic:", descriptive_topics)
    
    if selected_topic == "Data Types Explorer":
        show_data_types()
    elif selected_topic == "Central Tendency":
        show_central_tendency()
    elif selected_topic == "Dispersion Measures":
        show_dispersion_measures()
    elif selected_topic == "Quantiles and Percentiles":
        show_quantiles_percentiles()

def show_data_types():
    st.markdown('<h2 class="sub-header">Data Types Explorer</h2>', unsafe_allow_html=True)
    
    datasets = load_sample_datasets()
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> Data types classify information to determine appropriate statistical methods and visualizations.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simple-explanation">
    <strong>Simple Explanation:</strong> Just like organizing items in different boxes based on their properties, 
    we organize data based on whether it's numbers, categories, or other types.
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset selection
    dataset_name = st.selectbox("Choose a dataset:", list(datasets.keys()))
    df = datasets[dataset_name]
    
    if not df.empty:
        st.subheader(f"Dataset: {dataset_name.title()}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Overview:**")
            st.write(f"Shape: {df.shape}")
            st.dataframe(df.head())
            
        with col2:
            st.write("**Data Types:**")
            dtype_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(dtype_info)
        
        # Visualize data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            st.subheader("Numerical Data Analysis")
            selected_num_col = st.selectbox("Select numerical column:", numeric_cols)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            ax1.hist(df[selected_num_col].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'Distribution of {selected_num_col}')
            ax1.set_xlabel(selected_num_col)
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(df[selected_num_col].dropna())
            ax2.set_title(f'Box Plot of {selected_num_col}')
            ax2.set_ylabel(selected_num_col)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        if len(categorical_cols) > 0:
            st.subheader("Categorical Data Analysis")
            selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
            
            if selected_cat_col in df.columns:
                value_counts = df[selected_cat_col].value_counts()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Bar plot
                value_counts.plot(kind='bar', ax=ax1, color='lightcoral')
                ax1.set_title(f'Distribution of {selected_cat_col}')
                ax1.set_xlabel(selected_cat_col)
                ax1.set_ylabel('Count')
                ax1.tick_params(axis='x', rotation=45)
                
                # Pie chart
                ax2.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
                ax2.set_title(f'Proportion of {selected_cat_col}')
                
                plt.tight_layout()
                st.pyplot(fig)

def show_central_tendency():
    st.markdown('<h2 class="sub-header">Central Tendency Measures</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> Central tendency measures describe the center or typical value of a dataset.
    The three main measures are mean, median, and mode.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simple-explanation">
    <strong>Simple Explanation:</strong> Imagine you want to describe a typical person's height in your class. 
    Central tendency helps you find that "typical" value in different ways.
    </div>
    """, unsafe_allow_html=True)
    
    datasets = load_sample_datasets()
    dataset_name = st.selectbox("Choose dataset:", list(datasets.keys()), key="central_dataset")
    df = datasets[dataset_name]
    
    if not df.empty:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column:", numeric_cols, key="central_col")
            data = df[selected_col].dropna()
            
            if len(data) > 0:
                stats_dict = calculate_statistics(data)
                
                if stats_dict:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean", f"{stats_dict['mean']:.3f}")
                        st.markdown("""
                        <div class="formula-box">
                        <strong>Formula:</strong> μ = Σx / n<br>
                        <strong>When to use:</strong> Symmetric data, no extreme outliers
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Median", f"{stats_dict['median']:.3f}")
                        st.markdown("""
                        <div class="formula-box">
                        <strong>Formula:</strong> Middle value when sorted<br>
                        <strong>When to use:</strong> Skewed data or with outliers
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.metric("Mode", f"{stats_dict['mode']:.3f}")
                        st.markdown("""
                        <div class="formula-box">
                        <strong>Formula:</strong> Most frequent value<br>
                        <strong>When to use:</strong> Categorical data or finding most common value
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Histogram with central tendency lines
                    ax1.hist(data, bins=30, alpha=0.7, color='lightblue', edgecolor='black', density=True)
                    ax1.axvline(stats_dict['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats_dict["mean"]:.2f}')
                    ax1.axvline(stats_dict['median'], color='green', linestyle='-', linewidth=2, label=f'Median: {stats_dict["median"]:.2f}')
                    ax1.axvline(stats_dict['mode'], color='blue', linestyle=':', linewidth=2, label=f'Mode: {stats_dict["mode"]:.2f}')
                    ax1.set_title(f'Distribution of {selected_col} with Central Tendency Measures')
                    ax1.set_xlabel(selected_col)
                    ax1.set_ylabel('Density')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Box plot
                    ax2.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightcoral', alpha=0.7))
                    ax2.axvline(stats_dict['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats_dict["mean"]:.2f}')
                    ax2.axvline(stats_dict['median'], color='green', linestyle='-', linewidth=2, label=f'Median: {stats_dict["median"]:.2f}')
                    ax2.set_title('Box Plot View')
                    ax2.set_xlabel(selected_col)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Interpretation
                    st.subheader("Interpretation")
                    skewness = stats_dict['skewness']
                    if abs(skewness) < 0.5:
                        interpretation = "approximately symmetric - mean and median are similar"
                    elif skewness > 0.5:
                        interpretation = "right-skewed - mean > median due to high outliers"
                    else:
                        interpretation = "left-skewed - mean < median due to low outliers"
                    
                    st.write(f"**Distribution Shape:** {interpretation}")
                    st.write(f"**Skewness:** {skewness:.3f}")

def show_dispersion_measures():
    st.markdown('<h2 class="sub-header">Dispersion Measures</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> Dispersion measures describe how spread out or scattered the data points are 
    from the central tendency.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simple-explanation">
    <strong>Simple Explanation:</strong> If central tendency tells us the "typical" value, 
    dispersion tells us how much individual values differ from that typical value.
    </div>
    """, unsafe_allow_html=True)
    
    datasets = load_sample_datasets()
    dataset_name = st.selectbox("Choose dataset:", list(datasets.keys()), key="dispersion_dataset")
    df = datasets[dataset_name]
    
    if not df.empty:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column:", numeric_cols, key="dispersion_col")
            data = df[selected_col].dropna()
            
            if len(data) > 0:
                stats_dict = calculate_statistics(data)
                
                if stats_dict:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Standard Deviation", f"{stats_dict['std']:.3f}")
                    with col2:
                        st.metric("Variance", f"{stats_dict['var']:.3f}")
                    with col3:
                        st.metric("Range", f"{stats_dict['range']:.3f}")
                    with col4:
                        st.metric("IQR", f"{stats_dict['iqr']:.3f}")
                    
                    # Formulas and explanations
                    st.markdown("""
                    <div class="formula-box">
                    <strong>Standard Deviation:</strong> σ = √(Σ(x - μ)² / N)<br>
                    <strong>Variance:</strong> σ² = Σ(x - μ)² / N<br>
                    <strong>Range:</strong> Max - Min<br>
                    <strong>IQR:</strong> Q3 - Q1
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visualization
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    
                    # Histogram with std dev bands
                    mean_val = stats_dict['mean']
                    std_val = stats_dict['std']
                    
                    axes[0,0].hist(data, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
                    axes[0,0].axvline(mean_val, color='red', linestyle='-', linewidth=2, label='Mean')
                    axes[0,0].axvline(mean_val - std_val, color='orange', linestyle='--', label='-1 SD')
                    axes[0,0].axvline(mean_val + std_val, color='orange', linestyle='--', label='+1 SD')
                    axes[0,0].set_title('Distribution with Standard Deviation')
                    axes[0,0].legend()
                    axes[0,0].grid(True, alpha=0.3)
                    
                    # Box plot
                    axes[0,1].boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightgreen', alpha=0.7))
                    axes[0,1].set_title('Box Plot showing IQR')
                    axes[0,1].grid(True, alpha=0.3)
                    
                    # Range visualization
                    axes[1,0].plot([stats_dict['min'], stats_dict['max']], [0, 0], 'ro-', linewidth=3, markersize=8)
                    axes[1,0].set_title(f'Range: {stats_dict["range"]:.2f}')
                    axes[1,0].set_xlabel('Values')
                    axes[1,0].grid(True, alpha=0.3)
                    
                    # Comparison of measures
                    measures = ['Std Dev', 'Variance', 'Range', 'IQR']
                    values = [stats_dict['std'], stats_dict['var'], stats_dict['range'], stats_dict['iqr']]
                    axes[1,1].bar(measures, values, color=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow'])
                    axes[1,1].set_title('Comparison of Dispersion Measures')
                    axes[1,1].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

def show_quantiles_percentiles():
    st.markdown('<h2 class="sub-header">Quantiles and Percentiles</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> Quantiles divide a dataset into equal parts. 
    Percentiles are quantiles that divide data into 100 parts.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simple-explanation">
    <strong>Simple Explanation:</strong> Like dividing a pizza into equal slices, 
    quantiles help us understand what values separate different portions of our data.
    </div>
    """, unsafe_allow_html=True)
    
    datasets = load_sample_datasets()
    dataset_name = st.selectbox("Choose dataset:", list(datasets.keys()), key="quantile_dataset")
    df = datasets[dataset_name]
    
    if not df.empty:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column:", numeric_cols, key="quantile_col")
            data = df[selected_col].dropna()
            
            if len(data) > 0:
                # Calculate quantiles
                quartiles = [np.percentile(data, q) for q in [25, 50, 75]]
                deciles = [np.percentile(data, q) for q in range(10, 100, 10)]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Quartiles")
                    for i, q in enumerate(quartiles, 1):
                        st.write(f"Q{i} ({i*25}th percentile): {q:.3f}")
                    
                    st.subheader("Key Percentiles")
                    key_percentiles = [10, 25, 50, 75, 90, 95, 99]
                    for p in key_percentiles:
                        value = np.percentile(data, p)
                        st.write(f"P{p}: {value:.3f}")
                
                with col2:
                    # Interactive percentile calculator
                    st.subheader("Percentile Calculator")
                    percentile_input = st.slider("Select percentile:", 1, 99, 50)
                    calculated_percentile = np.percentile(data, percentile_input)
                    st.metric(f"{percentile_input}th Percentile", f"{calculated_percentile:.3f}")
                    st.write(f"**Meaning:** {percentile_input}% of values are below {calculated_percentile:.3f}")
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Histogram with quartile lines
                ax1.hist(data, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
                colors = ['red', 'orange', 'green']
                for i, q in enumerate(quartiles):
                    ax1.axvline(q, color=colors[i], linestyle='--', linewidth=2, label=f'Q{i+1}: {q:.2f}')
                ax1.set_title('Distribution with Quartiles')
                ax1.set_xlabel(selected_col)
                ax1.set_ylabel('Frequency')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Box plot with percentile overlay
                ax2.hist(data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                ax2.axvline(calculated_percentile, color='red', linestyle='--', linewidth=3, 
                           label=f'P{percentile_input}: {calculated_percentile:.2f}')
                ax2.set_title(f'Distribution with {percentile_input}th Percentile')
                ax2.set_xlabel(selected_col)
                ax2.set_ylabel('Frequency')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)

# ===============================
# 2. INFERENTIAL STATISTICS
# ===============================

def show_inferential_statistics():
    st.markdown('<h1 class="main-header">Inferential Statistics</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> Inferential statistics involves making predictions, generalizations, 
    and decisions about a population based on sample data.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simple-explanation">
    <strong>Simple Explanation:</strong> Like a detective using clues (sample data) to solve a mystery 
    (understand the whole population), inferential statistics helps us make educated guesses about 
    things we can't measure directly.
    </div>
    """, unsafe_allow_html=True)
    
    methods = [
        "Hypothesis Testing",
        "Estimation and Confidence Intervals", 
        "Regression Analysis",
        "Analysis of Variance (ANOVA)",
        "Chi-Square Tests",
        "Non-Parametric Methods",
        "Bayesian Inference"
    ]
    
    selected_method = st.selectbox("Select Inferential Method:", methods)
    
    if selected_method == "Hypothesis Testing":
        show_hypothesis_testing()
    elif selected_method == "Estimation and Confidence Intervals":
        show_estimation_confidence()
    elif selected_method == "Regression Analysis":
        show_regression_analysis()
    elif selected_method == "Analysis of Variance (ANOVA)":
        show_anova()
    elif selected_method == "Chi-Square Tests":
        show_chi_square_tests()
    elif selected_method == "Non-Parametric Methods":
        show_nonparametric_methods()
    elif selected_method == "Bayesian Inference":
        show_bayesian_inference()

def show_hypothesis_testing():
    st.markdown('<h2 class="sub-header">Hypothesis Testing</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> Hypothesis testing is a statistical procedure used to test assumptions 
    (hypotheses) about a population parameter using sample data.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simple-explanation">
    <strong>Simple Explanation:</strong> Imagine you suspect a coin is unfair. You flip it many times 
    and see if the results are so unusual that your suspicion is likely correct. That's hypothesis testing!
    </div>
    """, unsafe_allow_html=True)
    
    test_type = st.selectbox("Select Test Type:", ["One-Sample t-test", "Two-Sample t-test", "Z-test"])
    
    if test_type == "One-Sample t-test":
        st.subheader("One-Sample t-test")
        st.write("**Use case:** Test if a sample mean differs significantly from a known population mean")
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.slider("Sample size:", 10, 100, 30)
            population_mean = st.number_input("Hypothesized population mean:", value=0.0)
        with col2:
            sample_mean = st.number_input("Sample mean:", value=1.0)
            sample_std = st.number_input("Sample standard deviation:", value=2.0, min_value=0.1)
        
        # Calculate test statistic
        if sample_std > 0:
            t_statistic = (sample_mean - population_mean) / (sample_std / np.sqrt(sample_size))
            df = sample_size - 1
            p_value = 2 * (1 - t.cdf(abs(t_statistic), df))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("t-statistic", f"{t_statistic:.4f}")
            with col2:
                st.metric("p-value", f"{p_value:.4f}")
            with col3:
                st.metric("Degrees of Freedom", df)
            
            # Interpretation
            alpha = 0.05
            if p_value < alpha:
                conclusion = f"Reject H₀ (p < {alpha}): Significant difference exists"
                color = "red"
            else:
                conclusion = f"Fail to reject H₀ (p ≥ {alpha}): No significant difference"
                color = "green"
            
            st.markdown(f"**Conclusion:** <span style='color: {color}'>{conclusion}</span>", unsafe_allow_html=True)
            
            # Visualization
            x = np.linspace(-4, 4, 1000)
            y = t.pdf(x, df)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, y, 'b-', linewidth=2, label=f't-distribution (df={df})')
            ax.axvline(t_statistic, color='red', linestyle='--', linewidth=2, 
                      label=f't-statistic = {t_statistic:.3f}')
            
            # Shade rejection regions
            critical_value = t.ppf(0.975, df)
            x_left = x[x <= -critical_value]
            x_right = x[x >= critical_value]
            ax.fill_between(x_left, t.pdf(x_left, df), alpha=0.3, color='red', label='Rejection Region')
            ax.fill_between(x_right, t.pdf(x_right, df), alpha=0.3, color='red')
            
            ax.set_title('One-Sample t-test Distribution')
            ax.set_xlabel('t-value')
            ax.set_ylabel('Probability Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    elif test_type == "Two-Sample t-test":
        st.subheader("Two-Sample t-test")
        st.write("**Use case:** Compare means of two independent groups")
        
        # Generate sample data
        datasets = load_sample_datasets()
        
        if 'tips' in datasets and not datasets['tips'].empty:
            df = datasets['tips']
            
            # Example: Compare tips by gender
            if 'sex' in df.columns and 'tip' in df.columns:
                group1 = df[df['sex'] == 'Male']['tip'].dropna()
                group2 = df[df['sex'] == 'Female']['tip'].dropna()
                
                if len(group1) > 0 and len(group2) > 0:
                    # Perform t-test
                    t_stat, p_val = stats.ttest_ind(group1, group2)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Group Statistics:**")
                        st.write(f"Male tips: Mean = {np.mean(group1):.3f}, SD = {np.std(group1):.3f}, N = {len(group1)}")
                        st.write(f"Female tips: Mean = {np.mean(group2):.3f}, SD = {np.std(group2):.3f}, N = {len(group2)}")
                    
                    with col2:
                        st.metric("t-statistic", f"{t_stat:.4f}")
                        st.metric("p-value", f"{p_val:.4f}")
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Box plots
                    ax1.boxplot([group1, group2], labels=['Male', 'Female'])
                    ax1.set_title('Tip Distribution by Gender')
                    ax1.set_ylabel('Tip Amount')
                    ax1.grid(True, alpha=0.3)
                    
                    # Histograms
                    ax2.hist(group1, alpha=0.5, label='Male', bins=15)
                    ax2.hist(group2, alpha=0.5, label='Female', bins=15)
                    ax2.set_title('Tip Distribution Comparison')
                    ax2.set_xlabel('Tip Amount')
                    ax2.set_ylabel('Frequency')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    elif test_type == "Z-test":
        st.subheader("Z-test")
        st.write("**Use case:** Test hypotheses when population standard deviation is known and sample size is large")
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.slider("Sample size:", 30, 1000, 100)
            sample_mean = st.number_input("Sample mean:", value=105.0)
        with col2:
            population_mean = st.number_input("Population mean (H₀):", value=100.0)
            population_std = st.number_input("Population standard deviation:", value=15.0, min_value=0.1)
        
        # Calculate Z-statistic
        if population_std > 0:
            z_statistic = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))
            p_value = 2 * (1 - norm.cdf(abs(z_statistic)))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Z-statistic", f"{z_statistic:.4f}")
            with col2:
                st.metric("p-value", f"{p_value:.4f}")
            
            # Visualization
            x = np.linspace(-4, 4, 1000)
            y = norm.pdf(x)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, y, 'b-', linewidth=2, label='Standard Normal Distribution')
            ax.axvline(z_statistic, color='red', linestyle='--', linewidth=2, 
                      label=f'Z-statistic = {z_statistic:.3f}')
            
            # Shade rejection regions
            x_left = x[x <= -1.96]
            x_right = x[x >= 1.96]
            ax.fill_between(x_left, norm.pdf(x_left), alpha=0.3, color='red', label='Rejection Region (α=0.05)')
            ax.fill_between(x_right, norm.pdf(x_right), alpha=0.3, color='red')
            
            ax.set_title('Z-test Distribution')
            ax.set_xlabel('Z-value')
            ax.set_ylabel('Probability Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

def show_estimation_confidence():
    st.markdown('<h2 class="sub-header">Estimation and Confidence Intervals</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> Estimation involves using sample data to estimate population parameters. 
    Confidence intervals provide a range of plausible values for the parameter.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simple-explanation">
    <strong>Simple Explanation:</strong> Instead of guessing an exact number (like saying "the average height is exactly 5'8\""), 
    we give a range (like "we're 95% confident the average height is between 5'7\" and 5'9\"").
    </div>
    """, unsafe_allow_html=True)
    
    estimation_type = st.selectbox("Select Estimation Type:", ["Mean Estimation", "Proportion Estimation"])
    
    if estimation_type == "Mean Estimation":
        st.subheader("Confidence Interval for Mean")
        
        # Use real data
        datasets = load_sample_datasets()
        df = datasets['iris']
        
        if not df.empty:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            selected_col = st.selectbox("Select variable:", numeric_cols)
            confidence_level = st.slider("Confidence Level:", 0.80, 0.99, 0.95, 0.01)
            
            data = df[selected_col].dropna()
            
            if len(data) > 0:
                n = len(data)
                mean_estimate = np.mean(data)
                std_error = stats.sem(data)
                
                # Calculate confidence interval
                alpha = 1 - confidence_level
                t_critical = t.ppf(1 - alpha/2, n - 1)
                margin_error = t_critical * std_error
                ci_lower = mean_estimate - margin_error
                ci_upper = mean_estimate + margin_error
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sample Mean", f"{mean_estimate:.4f}")
                with col2:
                    st.metric("Standard Error", f"{std_error:.4f}")
                with col3:
                    st.metric("Margin of Error", f"{margin_error:.4f}")
                
                st.markdown(f"""
                **{confidence_level*100:.0f}% Confidence Interval:** [{ci_lower:.4f}, {ci_upper:.4f}]
                
                **Interpretation:** We are {confidence_level*100:.0f}% confident that the true population mean 
                of {selected_col} lies between {ci_lower:.4f} and {ci_upper:.4f}.
                """)
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Sample distribution
                ax1.hist(data, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
                ax1.axvline(mean_estimate, color='red', linestyle='--', linewidth=2, label=f'Sample Mean: {mean_estimate:.3f}')
                ax1.set_title(f'Sample Distribution of {selected_col}')
                ax1.set_xlabel(selected_col)
                ax1.set_ylabel('Frequency')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Confidence interval visualization
                x_ci = np.linspace(ci_lower - margin_error, ci_upper + margin_error, 1000)
                y_ci = norm.pdf(x_ci, mean_estimate, std_error)
                
                ax2.plot(x_ci, y_ci, 'b-', linewidth=2, label='Sampling Distribution')
                ax2.fill_between(x_ci[(x_ci >= ci_lower) & (x_ci <= ci_upper)], 
                                y_ci[(x_ci >= ci_lower) & (x_ci <= ci_upper)], 
                                alpha=0.3, color='green', label=f'{confidence_level*100:.0f}% CI')
                ax2.axvline(mean_estimate, color='red', linestyle='--', linewidth=2, label='Sample Mean')
                ax2.axvline(ci_lower, color='green', linestyle=':', linewidth=2)
                ax2.axvline(ci_upper, color='green', linestyle=':', linewidth=2)
                ax2.set_title(f'{confidence_level*100:.0f}% Confidence Interval')
                ax2.set_xlabel('Value')
                ax2.set_ylabel('Probability Density')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)

def show_regression_analysis():
    st.markdown('<h2 class="sub-header">Regression Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> Regression analysis is used to model and analyze the relationship 
    between a dependent variable and one or more independent variables.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simple-explanation">
    <strong>Simple Explanation:</strong> Like finding the best line through scattered points on a graph 
    to predict one thing based on another (like predicting house prices based on size).
    </div>
    """, unsafe_allow_html=True)
    
    datasets = load_sample_datasets()
    
    # Use tips dataset for regression example
    if 'tips' in datasets and not datasets['tips'].empty:
        df = datasets['tips']
        
        if 'total_bill' in df.columns and 'tip' in df.columns:
            st.subheader("Example: Predicting Tip Amount from Total Bill")
            
            # Prepare data
            X = df['total_bill'].dropna()
            y = df['tip'].dropna()
            
            # Ensure same length
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len].values.reshape(-1, 1)
            y = y.iloc[:min_len].values
            
            # Perform regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            slope = model.coef_[0]
            intercept = model.intercept_
            r2 = r2_score(y, y_pred)
            correlation = np.corrcoef(X.flatten(), y)[0, 1]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Slope", f"{slope:.4f}")
            with col2:
                st.metric("Intercept", f"{intercept:.4f}")
            with col3:
                st.metric("R-squared", f"{r2:.4f}")
            with col4:
                st.metric("Correlation", f"{correlation:.4f}")
            
            st.markdown(f"""
            **Regression Equation:** Tip = {intercept:.3f} + {slope:.3f} × Total Bill
            
            **Interpretation:** 
            - For every $1 increase in total bill, tip increases by ${slope:.3f}
            - {r2*100:.1f}% of the variation in tip amount is explained by total bill
            """)
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot with regression line
            ax1.scatter(X, y, alpha=0.6, color='blue', label='Data Points')
            ax1.plot(X, y_pred, color='red', linewidth=2, label=f'Regression Line (R² = {r2:.3f})')
            ax1.set_xlabel('Total Bill ($)')
            ax1.set_ylabel('Tip ($)')
            ax1.set_title('Regression: Tip vs Total Bill')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Residuals plot
            residuals = y - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Predicted Tip ($)')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals Plot')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Prediction tool
            st.subheader("Make a Prediction")
            bill_amount = st.number_input("Enter total bill amount ($):", min_value=0.0, value=25.0, step=0.5)
            predicted_tip = intercept + slope * bill_amount
            st.write(f"**Predicted tip:** ${predicted_tip:.2f}")

def show_anova():
    st.markdown('<h2 class="sub-header">Analysis of Variance (ANOVA)</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> ANOVA tests whether there are significant differences between 
    the means of three or more groups.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simple-explanation">
    <strong>Simple Explanation:</strong> Like comparing test scores from three different schools 
    to see if one school performs significantly better than the others.
    </div>
    """, unsafe_allow_html=True)
    
    datasets = load_sample_datasets()
    
    # Use iris dataset for ANOVA example
    df = datasets['iris']
    
    if not df.empty and 'species' in df.columns:
        st.subheader("Example: Comparing Petal Length across Iris Species")
        
        # Prepare data
        species_groups = []
        species_names = df['species'].unique()
        
        for species in species_names:
            group_data = df[df['species'] == species]['petal length (cm)'].dropna()
            if len(group_data) > 0:
                species_groups.append(group_data)
        
        if len(species_groups) >= 3:
            # Perform ANOVA
            f_statistic, p_value = stats.f_oneway(*species_groups)
            
            # Calculate group statistics
            group_stats = []
            for i, group in enumerate(species_groups):
                group_stats.append({
                    'Species': species_names[i],
                    'Mean': np.mean(group),
                    'Std': np.std(group),
                    'Count': len(group)
                })
            
            stats_df = pd.DataFrame(group_stats)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Group Statistics:**")
                st.dataframe(stats_df)
            
            with col2:
                st.metric("F-statistic", f"{f_statistic:.4f}")
                st.metric("p-value", f"{p_value:.6f}")
                
                if p_value < 0.05:
                    st.success("Significant differences exist between groups (p < 0.05)")
                else:
                    st.info("No significant differences between groups (p ≥ 0.05)")
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Box plots
            data_for_boxplot = [group.values for group in species_groups]
            ax1.boxplot(data_for_boxplot, labels=species_names)
            ax1.set_title('Petal Length by Species')
            ax1.set_ylabel('Petal Length (cm)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Violin plots
            positions = range(1, len(species_groups) + 1)
            parts = ax2.violinplot(data_for_boxplot, positions=positions, widths=0.6)
            ax2.set_xticks(positions)
            ax2.set_xticklabels(species_names, rotation=45)
            ax2.set_title('Distribution of Petal Length by Species')
            ax2.set_ylabel('Petal Length (cm)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Post-hoc analysis note
            if p_value < 0.05:
                st.write("""
                **Note:** Since ANOVA detected significant differences, post-hoc tests 
                (like Tukey's HSD) could be performed to determine which specific groups differ.
                """)

def show_chi_square_tests():
    st.markdown('<h2 class="sub-header">Chi-Square Tests</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> Chi-square tests examine relationships between categorical variables 
    or test if observed frequencies match expected frequencies.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simple-explanation">
    <strong>Simple Explanation:</strong> Like checking if there's a relationship between gender and 
    favorite ice cream flavor, or testing if a dice is fair by comparing actual rolls to expected rolls.
    </div>
    """, unsafe_allow_html=True)
    
    test_type = st.selectbox("Select Chi-Square Test:", 
                           ["Test of Independence", "Goodness of Fit Test"])
    
    datasets = load_sample_datasets()
    
    if test_type == "Test of Independence":
        st.subheader("Chi-Square Test of Independence")
        
        # Use titanic dataset if available
        if 'titanic' in datasets and not datasets['titanic'].empty:
            df = datasets['titanic']
            
            if 'survived' in df.columns and 'sex' in df.columns:
                st.write("**Example:** Testing independence between Gender and Survival on Titanic")
                
                # Create contingency table
                contingency_table = pd.crosstab(df['sex'], df['survived'])
                
                st.write("**Contingency Table:**")
                st.dataframe(contingency_table)
                
                # Perform chi-square test
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chi-Square Statistic", f"{chi2_stat:.4f}")
                with col2:
                    st.metric("p-value", f"{p_value:.6f}")
                with col3:
                    st.metric("Degrees of Freedom", dof)
                
                # Effect size (Cramér's V)
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                st.metric("Cramér's V (Effect Size)", f"{cramers_v:.4f}")
                
                # Interpretation
                if p_value < 0.05:
                    st.success("Significant association exists between variables (p < 0.05)")
                else:
                    st.info("No significant association between variables (p ≥ 0.05)")
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Heatmap of observed frequencies
                sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', ax=ax1)
                ax1.set_title('Observed Frequencies')
                
                # Heatmap of expected frequencies
                expected_df = pd.DataFrame(expected, 
                                         index=contingency_table.index, 
                                         columns=contingency_table.columns)
                sns.heatmap(expected_df, annot=True, fmt='.1f', cmap='Reds', ax=ax2)
                ax2.set_title('Expected Frequencies')
                
                plt.tight_layout()
                st.pyplot(fig)
    
    elif test_type == "Goodness of Fit Test":
        st.subheader("Chi-Square Goodness of Fit Test")
        st.write("**Example:** Testing if a die is fair")
        
        # Simulate dice rolls
        rolls = st.slider("Number of dice rolls:", 60, 600, 120)
        observed = np.random.multinomial(rolls, [1/6]*6)
        expected = np.array([rolls/6]*6)
        
        # Perform goodness of fit test
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chi-Square Statistic", f"{chi2_stat:.4f}")
        with col2:
            st.metric("p-value", f"{p_value:.6f}")
        
        # Create results table
        results_df = pd.DataFrame({
            'Face': [1, 2, 3, 4, 5, 6],
            'Observed': observed,
            'Expected': expected,
            'Difference': observed - expected
        })
        
        st.dataframe(results_df)
        
        if p_value < 0.05:
            st.error("Die appears to be unfair (p < 0.05)")
        else:
            st.success("Die appears to be fair (p ≥ 0.05)")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(1, 7)
        width = 0.35
        
        ax.bar(x - width/2, observed, width, label='Observed', alpha=0.7, color='skyblue')
        ax.bar(x + width/2, expected, width, label='Expected', alpha=0.7, color='lightcoral')
        
        ax.set_xlabel('Die Face')
        ax.set_ylabel('Frequency')
        ax.set_title('Dice Roll Results: Observed vs Expected')
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

def show_nonparametric_methods():
    st.markdown('<h2 class="sub-header">Non-Parametric Methods</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> Non-parametric methods are statistical techniques that don't assume 
    a specific distribution for the data and are often based on ranks rather than actual values.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simple-explanation">
    <strong>Simple Explanation:</strong> When your data doesn't follow a normal bell curve or you have 
    small samples, these methods work with the order/ranking of your data instead of exact values.
    </div>
    """, unsafe_allow_html=True)
    
    method = st.selectbox("Select Non-Parametric Method:", 
                         ["Mann-Whitney U Test", "Wilcoxon Signed-Rank Test", "Kruskal-Wallis Test"])
    
    if method == "Mann-Whitney U Test":
        st.subheader("Mann-Whitney U Test")
        st.write("**Use case:** Compare two independent groups (non-parametric alternative to t-test)")
        
        datasets = load_sample_datasets()
        
        # Use tips data
        if 'tips' in datasets and not datasets['tips'].empty:
            df = datasets['tips']
            
            if 'sex' in df.columns and 'tip' in df.columns:
                st.write("**Example:** Comparing tip amounts between males and females")
                
                # Prepare data
                male_tips = df[df['sex'] == 'Male']['tip'].dropna()
                female_tips = df[df['sex'] == 'Female']['tip'].dropna()
                
                if len(male_tips) > 0 and len(female_tips) > 0:
                    # Perform Mann-Whitney U test
                    statistic, p_value = stats.mannwhitneyu(male_tips, female_tips, alternative='two-sided')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Group Statistics:**")
                        st.write(f"Male tips: Median = {np.median(male_tips):.3f}, N = {len(male_tips)}")
                        st.write(f"Female tips: Median = {np.median(female_tips):.3f}, N = {len(female_tips)}")
                    
                    with col2:
                        st.metric("U-statistic", f"{statistic:.0f}")
                        st.metric("p-value", f"{p_value:.6f}")
                    
                    if p_value < 0.05:
                        st.success("Significant difference between groups (p < 0.05)")
                    else:
                        st.info("No significant difference between groups (p ≥ 0.05)")
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Box plots
                    ax1.boxplot([male_tips, female_tips], labels=['Male', 'Female'])
                    ax1.set_title('Tip Distribution by Gender')
                    ax1.set_ylabel('Tip Amount ($)')
                    ax1.grid(True, alpha=0.3)
                    
                    # Histograms
                    ax2.hist(male_tips, alpha=0.5, label='Male', bins=15, density=True)
                    ax2.hist(female_tips, alpha=0.5, label='Female', bins=15, density=True)
                    ax2.axvline(np.median(male_tips), color='blue', linestyle='--', label=f'Male Median: {np.median(male_tips):.2f}')
                    ax2.axvline(np.median(female_tips), color='orange', linestyle='--', label=f'Female Median: {np.median(female_tips):.2f}')
                    ax2.set_title('Tip Distribution Comparison')
                    ax2.set_xlabel('Tip Amount ($)')
                    ax2.set_ylabel('Density')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

def show_bayesian_inference():
    st.markdown('<h2 class="sub-header">Bayesian Inference</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> Bayesian inference uses Bayes' theorem to update probabilities 
    as new evidence becomes available, treating parameters as random variables with probability distributions.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simple-explanation">
    <strong>Simple Explanation:</strong> Like being a detective who starts with a hunch (prior belief) 
    and updates it as new clues (data) are found, becoming more confident or changing the theory entirely.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Bayes' Theorem")
    st.latex(r"P(H|E) = \frac{P(E|H) \times P(H)}{P(E)}")
    
    st.markdown("""
    Where:
    - **P(H|E)** = Posterior probability (updated belief after seeing evidence)
    - **P(E|H)** = Likelihood (probability of evidence given hypothesis)
    - **P(H)** = Prior probability (initial belief)
    - **P(E)** = Evidence (probability of observing the data)
    """)
    
    st.subheader("Interactive Example: Medical Diagnosis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Scenario Parameters:**")
        disease_prevalence = st.slider("Disease prevalence in population (%):", 0.1, 10.0, 1.0, 0.1)
        test_sensitivity = st.slider("Test sensitivity (% of sick people testing positive):", 80.0, 99.9, 95.0, 0.1)
        test_specificity = st.slider("Test specificity (% of healthy people testing negative):", 80.0, 99.9, 90.0, 0.1)
    
    with col2:
        # Calculate Bayesian probabilities
        prior = disease_prevalence / 100
        sensitivity = test_sensitivity / 100
        specificity = test_specificity / 100
        
        # P(Test+|Disease) = sensitivity
        # P(Test+|No Disease) = 1 - specificity
        # P(Test+) = P(Test+|Disease) * P(Disease) + P(Test+|No Disease) * P(No Disease)
        
        p_test_pos = sensitivity * prior + (1 - specificity) * (1 - prior)
        
        # Bayes' theorem: P(Disease|Test+)
        posterior = (sensitivity * prior) / p_test_pos
        
        st.write("**Results:**")
        st.metric("Prior Probability", f"{prior*100:.2f}%")
        st.metric("Posterior Probability", f"{posterior*100:.2f}%")
        st.write(f"**Interpretation:** If someone tests positive, there's a {posterior*100:.1f}% chance they actually have the disease.")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prior vs Posterior comparison
    categories = ['Has Disease', "Doesn't Have Disease"]
    prior_probs = [prior, 1 - prior]
    posterior_probs = [posterior, 1 - posterior]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, prior_probs, width, label='Prior (Before Test)', alpha=0.7, color='lightblue')
    ax1.bar(x + width/2, posterior_probs, width, label='Posterior (After Positive Test)', alpha=0.7, color='lightcoral')
    ax1.set_ylabel('Probability')
    ax1.set_title('Bayesian Update: Prior vs Posterior')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sensitivity analysis
    prevalence_range = np.linspace(0.001, 0.1, 100)
    posterior_range = []
    
    for prev in prevalence_range:
        p_test_pos_temp = sensitivity * prev + (1 - specificity) * (1 - prev)
        posterior_temp = (sensitivity * prev) / p_test_pos_temp
        posterior_range.append(posterior_temp)
    
    ax2.plot(prevalence_range * 100, np.array(posterior_range) * 100, linewidth=2, color='green')
    ax2.axvline(disease_prevalence, color='red', linestyle='--', label=f'Current Prevalence: {disease_prevalence}%')
    ax2.axhline(posterior * 100, color='red', linestyle=':', label=f'Current Posterior: {posterior*100:.1f}%')
    ax2.set_xlabel('Disease Prevalence (%)')
    ax2.set_ylabel('Posterior Probability (%)')
    ax2.set_title('Sensitivity Analysis: How Prevalence Affects Posterior')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **Key Insights:**
    - Even with a highly accurate test, if a disease is rare, most positive results may be false positives
    - The posterior probability depends heavily on the prior probability (base rate)
    - This demonstrates why medical tests are often followed up with additional testing
    """)

# ===============================
# 3. EXPLORATORY DATA ANALYSIS
# ===============================

def show_exploratory_data_analysis():
    st.markdown('<h1 class="main-header">Exploratory Data Analysis (EDA)</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> EDA is the process of analyzing and visualizing data to understand 
    patterns, relationships, anomalies, and insights before formal statistical modeling.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simple-explanation">
    <strong>Simple Explanation:</strong> Like exploring a new city before making plans - you look around, 
    get familiar with the landscape, and discover interesting places before deciding what to do.
    </div>
    """, unsafe_allow_html=True)
    
    analysis_type = st.selectbox("Select Analysis Type:", ["Univariate Analysis", "Bivariate Analysis"])
    
    if analysis_type == "Univariate Analysis":
        show_univariate_analysis()
    elif analysis_type == "Bivariate Analysis":
        show_bivariate_analysis()

def show_univariate_analysis():
    st.markdown('<h2 class="sub-header">Univariate Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> Univariate analysis examines one variable at a time to understand 
    its distribution, central tendency, variability, and identify outliers.
    </div>
    """, unsafe_allow_html=True)
    
    datasets = load_sample_datasets()
    dataset_name = st.selectbox("Choose dataset:", list(datasets.keys()), key="uni_dataset")
    df = datasets[dataset_name]
    
    if not df.empty:
        variable_type = st.radio("Select variable type:", ["Numerical", "Categorical"])
        
        if variable_type == "Numerical":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select numerical variable:", numeric_cols, key="uni_num_col")
                data = df[selected_col].dropna()
                
                if len(data) > 0:
                    # Calculate statistics
                    stats_dict = calculate_statistics(data)
                    
                    # Display summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{stats_dict['mean']:.3f}")
                        st.metric("Std Dev", f"{stats_dict['std']:.3f}")
                    with col2:
                        st.metric("Median", f"{stats_dict['median']:.3f}")
                        st.metric("IQR", f"{stats_dict['iqr']:.3f}")
                    with col3:
                        st.metric("Min", f"{stats_dict['min']:.3f}")
                        st.metric("Max", f"{stats_dict['max']:.3f}")
                    with col4:
                        st.metric("Skewness", f"{stats_dict['skewness']:.3f}")
                        st.metric("Kurtosis", f"{stats_dict['kurtosis']:.3f}")
                    
                    # Multiple visualizations
                    plot_type = st.selectbox("Select plot type:", 
                                           ["Histogram", "Box Plot", "Violin Plot", "Q-Q Plot", "ECDF"])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if plot_type == "Histogram":
                        bins = st.slider("Number of bins:", 10, 50, 25)
                        ax.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
                        ax.axvline(stats_dict['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats_dict["mean"]:.2f}')
                        ax.axvline(stats_dict['median'], color='green', linestyle='-', linewidth=2, label=f'Median: {stats_dict["median"]:.2f}')
                        ax.set_title(f'Histogram of {selected_col}')
                        ax.set_xlabel(selected_col)
                        ax.set_ylabel('Frequency')
                        ax.legend()
                        
                    elif plot_type == "Box Plot":
                        ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightcoral', alpha=0.7))
                        ax.set_title(f'Box Plot of {selected_col}')
                        ax.set_ylabel(selected_col)
                        
                    elif plot_type == "Violin Plot":
                        parts = ax.violinplot([data], positions=[1], widths=[0.6])
                        for pc in parts['bodies']:
                            pc.set_facecolor('lightgreen')
                            pc.set_alpha(0.7)
                        ax.set_title(f'Violin Plot of {selected_col}')
                        ax.set_ylabel(selected_col)
                        
                    elif plot_type == "Q-Q Plot":
                        stats.probplot(data, dist="norm", plot=ax)
                        ax.set_title(f'Q-Q Plot of {selected_col}')
                        
                    elif plot_type == "ECDF":
                        sorted_data = np.sort(data)
                        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                        ax.plot(sorted_data, y, marker='.', linestyle='none', markersize=3)
                        ax.set_title(f'Empirical CDF of {selected_col}')
                        ax.set_xlabel(selected_col)
                        ax.set_ylabel('Cumulative Probability')
                    
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        else:  # Categorical
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                selected_col = st.selectbox("Select categorical variable:", categorical_cols, key="uni_cat_col")
                
                if selected_col in df.columns:
                    value_counts = df[selected_col].value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Summary Statistics:**")
                        st.write(f"Number of categories: {len(value_counts)}")
                        st.write(f"Most common: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)")
                        st.write(f"Total observations: {value_counts.sum()}")
                    
                    with col2:
                        st.write("**Frequency Table:**")
                        freq_df = pd.DataFrame({
                            'Category': value_counts.index,
                            'Count': value_counts.values,
                            'Percentage': (value_counts.values / value_counts.sum() * 100).round(2)
                        })
                        st.dataframe(freq_df)
                    
                    # Visualization
                    plot_type = st.selectbox("Select plot type:", ["Bar Chart", "Pie Chart", "Horizontal Bar"])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if plot_type == "Bar Chart":
                        value_counts.plot(kind='bar', ax=ax, color='lightcoral', alpha=0.7)
                        ax.set_title(f'Bar Chart of {selected_col}')
                        ax.set_xlabel(selected_col)
                        ax.set_ylabel('Count')
                        plt.xticks(rotation=45)
                        
                    elif plot_type == "Pie Chart":
                        ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                        ax.set_title(f'Pie Chart of {selected_col}')
                        
                    elif plot_type == "Horizontal Bar":
                        value_counts.plot(kind='barh', ax=ax, color='lightgreen', alpha=0.7)
                        ax.set_title(f'Horizontal Bar Chart of {selected_col}')
                        ax.set_xlabel('Count')
                        ax.set_ylabel(selected_col)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

def show_bivariate_analysis():
    st.markdown('<h2 class="sub-header">Bivariate Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Definition:</strong> Bivariate analysis examines the relationship between two variables 
    to understand correlation, association, or dependence patterns.
    </div>
    """, unsafe_allow_html=True)
    
    datasets = load_sample_datasets()
    dataset_name = st.selectbox("Choose dataset:", list(datasets.keys()), key="bi_dataset")
    df = datasets[dataset_name]
    
    if not df.empty:
        relationship_type = st.selectbox("Select relationship type:", 
                                       ["Numerical vs Numerical", "Categorical vs Numerical", "Categorical vs Categorical"])
        
        if relationship_type == "Numerical vs Numerical":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("Select X variable:", numeric_cols, key="bi_x_var")
                with col2:
                    y_var = st.selectbox("Select Y variable:", 
                                       [col for col in numeric_cols if col != x_var], key="bi_y_var")
                
                # Clean data
                clean_df = df[[x_var, y_var]].dropna()
                
                if len(clean_df) > 0:
                    # Calculate correlation
                    correlation = clean_df[x_var].corr(clean_df[y_var])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pearson Correlation", f"{correlation:.4f}")
                    with col2:
                        if abs(correlation) > 0.7:
                            strength = "Strong"
                        elif abs(correlation) > 0.3:
                            strength = "Moderate"
                        else:
                            strength = "Weak"
                        st.write(f"**Strength:** {strength}")
                    with col3:
                        direction = "Positive" if correlation > 0 else "Negative"
                        st.write(f"**Direction:** {direction}")
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Scatter plot with regression line
                    ax1.scatter(clean_df[x_var], clean_df[y_var], alpha=0.6, color='blue')
                    
                    # Add regression line
                    z = np.polyfit(clean_df[x_var], clean_df[y_var], 1)
                    p = np.poly1d(z)
                    ax1.plot(clean_df[x_var], p(clean_df[x_var]), "r--", alpha=0.8, linewidth=2)
                    
                    ax1.set_xlabel(x_var)
                    ax1.set_ylabel(y_var)
                    ax1.set_title(f'Scatter Plot: {x_var} vs {y_var}')
                    ax1.grid(True, alpha=0.3)
                    
                    # Hexbin plot for density
                    ax2.hexbin(clean_df[x_var], clean_df[y_var], gridsize=20, cmap='Blues')
                    ax2.set_xlabel(x_var)
                    ax2.set_ylabel(y_var)
                    ax2.set_title(f'Density Plot: {x_var} vs {y_var}')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        elif relationship_type == "Categorical vs Numerical":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    cat_var = st.selectbox("Select categorical variable:", categorical_cols, key="bi_cat_var")
                with col2:
                    num_var = st.selectbox("Select numerical variable:", numeric_cols, key="bi_num_var")
                
                # Clean data
                clean_df = df[[cat_var, num_var]].dropna()
                
                if len(clean_df) > 0:
                    # Group statistics
                    group_stats = clean_df.groupby(cat_var)[num_var].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
                    st.write("**Group Statistics:**")
                    st.dataframe(group_stats)
                    
                    # Statistical test
                    groups = [group[num_var].values for name, group in clean_df.groupby(cat_var)]
                    if len(groups) >= 2:
                        if len(groups) == 2:
                            t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
                            test_name = "T-test"
                        else:
                            f_stat, p_val = stats.f_oneway(*groups)
                            test_name = "ANOVA"
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Test Statistic", f"{f_stat if len(groups) > 2 else t_stat:.4f}")
                        with col2:
                            st.metric("p-value", f"{p_val:.6f}")
                        
                        if p_val < 0.05:
                            st.success(f"Significant difference between groups (p < 0.05) - {test_name}")
                        else:
                            st.info(f"No significant difference between groups (p ≥ 0.05) - {test_name}")
                    
                    # Visualization
                    plot_choice = st.selectbox("Choose visualization:", ["Box Plot", "Violin Plot", "Strip Plot"])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if plot_choice == "Box Plot":
                        clean_df.boxplot(column=num_var, by=cat_var, ax=ax)
                        ax.set_title(f'Box Plot: {num_var} by {cat_var}')
                    elif plot_choice == "Violin Plot":
                        # Manual violin plot since pandas doesn't have direct support
                        unique_cats = clean_df[cat_var].unique()
                        data_by_cat = [clean_df[clean_df[cat_var] == cat][num_var].values for cat in unique_cats]
                        parts = ax.violinplot(data_by_cat, positions=range(len(unique_cats)))
                        ax.set_xticks(range(len(unique_cats)))
                        ax.set_xticklabels(unique_cats)
                        ax.set_title(f'Violin Plot: {num_var} by {cat_var}')
                        ax.set_ylabel(num_var)
                    else:  # Strip Plot
                        for i, cat in enumerate(clean_df[cat_var].unique()):
                            cat_data = clean_df[clean_df[cat_var] == cat][num_var]
                            y_jitter = np.random.normal(i, 0.1, len(cat_data))
                            ax.scatter(y_jitter, cat_data, alpha=0.6, label=cat)
                        ax.set_title(f'Strip Plot: {num_var} by {cat_var}')
                        ax.set_xlabel(cat_var)
                        ax.set_ylabel(num_var)
                        ax.legend()
                    
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
        
        elif relationship_type == "Categorical vs Categorical":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    cat_var1 = st.selectbox("Select first categorical variable:", categorical_cols, key="bi_cat1")
                with col2:
                    cat_var2 = st.selectbox("Select second categorical variable:", 
                                          [col for col in categorical_cols if col != cat_var1], key="bi_cat2")
                
                # Create contingency table
                clean_df = df[[cat_var1, cat_var2]].dropna()
                
                if len(clean_df) > 0:
                    crosstab = pd.crosstab(clean_df[cat_var1], clean_df[cat_var2])
                    
                    st.write("**Contingency Table:**")
                    st.dataframe(crosstab)
                    
                    # Chi-square test
                    if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
                        chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Chi-Square", f"{chi2:.4f}")
                        with col2:
                            st.metric("p-value", f"{p_value:.6f}")
                        with col3:
                            st.metric("Degrees of Freedom", dof)
                        
                        if p_value < 0.05:
                            st.success("Significant association between variables (p < 0.05)")
                        else:
                            st.info("No significant association between variables (p ≥ 0.05)")
                    
                    # Visualization
                    plot_choice = st.selectbox("Choose visualization:", ["Heatmap", "Stacked Bar", "Grouped Bar"])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if plot_choice == "Heatmap":
                        sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_title(f'Heatmap: {cat_var1} vs {cat_var2}')
                    elif plot_choice == "Stacked Bar":
                        crosstab.plot(kind='bar', stacked=True, ax=ax)
                        ax.set_title(f'Stacked Bar: {cat_var1} vs {cat_var2}')
                        ax.legend(title=cat_var2, bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.xticks(rotation=45)
                    else:  # Grouped Bar
                        crosstab.plot(kind='bar', ax=ax)
                        ax.set_title(f'Grouped Bar: {cat_var1} vs {cat_var2}')
                        ax.legend(title=cat_var2, bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

# ===============================
# 4. STATISTICAL TERMINOLOGIES
# ===============================

def show_statistical_terminologies():
    st.markdown('<h1 class="main-header">Statistical Terminologies</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="definition-box">
    <strong>Purpose:</strong> This comprehensive glossary covers essential statistical terms 
    from both descriptive and inferential statistics, with simple explanations and examples.
    </div>
    """, unsafe_allow_html=True)
    
    # Categories of terms
    term_categories = {
        "Basic Concepts": {
            "Population": {
                "definition": "The entire group of individuals or items being studied.",
                "simple": "Everyone or everything you're interested in learning about.",
                "example": "All students in a university when studying average GPA."
            },
            "Sample": {
                "definition": "A subset of the population selected for study.",
                "simple": "A smaller group chosen to represent the whole population.",
                "example": "Surveying 500 students out of 10,000 to estimate average GPA."
            },
            "Parameter": {
                "definition": "A numerical characteristic of a population.",
                "simple": "A number that describes something about the entire population.",
                "example": "The true average height of all adults in a country."
            },
            "Statistic": {
                "definition": "A numerical characteristic calculated from sample data.",
                "simple": "A number calculated from your sample to estimate the population parameter.",
                "example": "The average height calculated from 1,000 sampled adults."
            }
        },
        
        "Descriptive Statistics": {
            "Mean": {
                "definition": "The arithmetic average of all values in a dataset.",
                "simple": "Add all numbers and divide by how many numbers you have.",
                "example": "Test scores 80, 85, 90. Mean = (80+85+90)/3 = 85."
            },
            "Median": {
                "definition": "The middle value when data is arranged in ascending order.",
                "simple": "The value in the middle when you line up all numbers from smallest to largest.",
                "example": "In 1, 3, 5, 7, 9, the median is 5."
            },
            "Mode": {
                "definition": "The value that appears most frequently in a dataset.",
                "simple": "The number that shows up most often.",
                "example": "In 1, 2, 2, 3, 2, 4, the mode is 2."
            },
            "Standard Deviation": {
                "definition": "A measure of how spread out data points are from the mean.",
                "simple": "Shows how much the numbers typically differ from the average.",
                "example": "Low SD means numbers are close to average; high SD means they're spread out."
            },
            "Variance": {
                "definition": "The average of squared differences from the mean.",
                "simple": "Like standard deviation but squared. Shows how much data varies.",
                "example": "If SD = 5, then variance = 25."
            }
        },
        
        "Inferential Statistics": {
            "Hypothesis Testing": {
                "definition": "A procedure to test assumptions about population parameters using sample data.",
                "simple": "Using sample data to test whether a belief about the population is likely true.",
                "example": "Testing if a new drug is more effective than existing treatment."
            },
            "P-value": {
                "definition": "Probability of observing data as extreme as observed, assuming null hypothesis is true.",
                "simple": "How surprising your results would be if nothing special was happening.",
                "example": "p = 0.03 means 3% chance of these results if there's really no effect."
            },
            "Confidence Interval": {
                "definition": "A range of values likely to contain the true population parameter.",
                "simple": "A range where we're pretty sure the true answer lies.",
                "example": "95% confident the average height is between 5'6\" and 5'10\"."
            },
            "Type I Error": {
                "definition": "Rejecting a true null hypothesis (false positive).",
                "simple": "Thinking you found something when you really didn't.",
                "example": "Concluding a drug works when it actually doesn't."
            },
            "Type II Error": {
                "definition": "Accepting a false null hypothesis (false negative).",
                "simple": "Missing a real effect when it actually exists.",
                "example": "Concluding a drug doesn't work when it actually does."
            },
            "Statistical Power": {
                "definition": "Probability of correctly rejecting a false null hypothesis.",
                "simple": "Your ability to detect a real effect when it exists.",
                "example": "80% power means 80% chance of finding an effect if it's really there."
            }
        },
        
        "Probability and Distributions": {
            "Normal Distribution": {
                "definition": "A symmetric, bell-shaped probability distribution.",
                "simple": "The classic bell curve where most values cluster around the middle.",
                "example": "Heights, test scores, and many natural phenomena follow this pattern."
            },
            "Standard Normal Distribution": {
                "definition": "Normal distribution with mean = 0 and standard deviation = 1.",
                "simple": "The standard bell curve used for comparison and calculations.",
                "example": "Z-scores are based on this distribution."
            },
            "Skewness": {
                "definition": "Measure of asymmetry in a distribution.",
                "simple": "Shows if your data leans more to one side like a lopsided hill.",
                "example": "Income data is right-skewed (most people earn less, few earn much more)."
            },
            "Kurtosis": {
                "definition": "Measure of the 'tailedness' of a distribution.",
                "simple": "Shows if your data has heavy tails (more extreme values) or light tails.",
                "example": "High kurtosis means more outliers than normal distribution."
            }
        },
        
        "Correlation and Regression": {
            "Correlation Coefficient": {
                "definition": "Measures strength and direction of linear relationship between two variables.",
                "simple": "Shows how two things move together. +1 = perfect positive, -1 = perfect negative, 0 = no relationship.",
                "example": "Height vs Weight might have r = 0.7 (strong positive correlation)."
            },
            "R-squared": {
                "definition": "Proportion of variance in dependent variable explained by independent variable(s).",
                "simple": "Percentage of variation in Y that's explained by X.",
                "example": "R² = 0.64 means 64% of variation in house prices is explained by size."
            },
            "Regression": {
                "definition": "Statistical method for modeling relationship between variables.",
                "simple": "Finding the best line through data points to predict one thing from another.",
                "example": "Predicting house price based on size, location, and age."
            }
        },
        
        "Non-Parametric Statistics": {
            "Mann-Whitney U Test": {
                "definition": "Non-parametric test comparing two independent groups.",
                "simple": "Compares two groups without assuming normal distribution.",
                "example": "Comparing test scores between two schools when scores aren't normally distributed."
            },
            "Chi-Square Test": {
                "definition": "Tests relationships between categorical variables.",
                "simple": "Checks if categories are related or independent.",
                "example": "Testing if gender is related to product preference."
            },
            "Wilcoxon Test": {
                "definition": "Non-parametric test for paired samples.",
                "simple": "Compares before/after measurements without assuming normal distribution.",
                "example": "Comparing blood pressure before and after treatment."
            }
        }
    }
    
    # Display terms by category
    category = st.selectbox("Select Category:", list(term_categories.keys()))
    
    st.markdown(f'<h2 class="sub-header">{category}</h2>', unsafe_allow_html=True)
    
    for term, info in term_categories[category].items():
        with st.expander(f" {term}"):
            st.markdown(f"""
            <div class="definition-box">
            <strong>Definition:</strong> {info['definition']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="simple-explanation">
            <strong>Simple Explanation:</strong> {info['simple']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Example:** {info['example']}")
    
    # Search functionality
    st.markdown("## Search Terms")
    search_query = st.text_input("Search for a statistical term:", placeholder="e.g., mean, correlation, p-value")
    
    if search_query:
        search_results = []
        search_lower = search_query.lower()
        
        for category_name, terms in term_categories.items():
            for term, info in terms.items():
                if (search_lower in term.lower() or 
                    search_lower in info['definition'].lower() or 
                    search_lower in info['simple'].lower() or 
                    search_lower in info['example'].lower()):
                    search_results.append((category_name, term, info))
        
        if search_results:
            st.markdown(f"### Found {len(search_results)} result(s):")
            for category_name, term, info in search_results:
                st.markdown(f"""
                **{term}** (*{category_name}*)
                - **Definition:** {info['definition']}
                - **Simple:** {info['simple']}
                - **Example:** {info['example']}
                ---
                """)
        else:
            st.write("No terms found. Try different keywords.")

# ===============================
# 5. FEEDBACK
# ===============================

def show_feedback():
    st.markdown('<h1 class="main-header">Feedback and Suggestions</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Contact Information
    
    For any feedback or suggestions for the next versions, please mail to: **ma24m012@smail.iitm.ac.in**
    
    Your feedback helps improve this educational platform!
    """)

# ===============================
# MAIN APPLICATION
# ===============================

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    sections = [
        "Descriptive Statistics",
        "Inferential Statistics", 
        "Exploratory Data Analysis",
        "Statistical Terminologies",
        "Feedback"
    ]
    
    selected_section = st.sidebar.radio("Go to:", sections)
    
    # Route to selected section
    if selected_section == "Descriptive Statistics":
        show_descriptive_statistics()
    elif selected_section == "Inferential Statistics":
        show_inferential_statistics()
    elif selected_section == "Exploratory Data Analysis":
        show_exploratory_data_analysis()
    elif selected_section == "Statistical Terminologies":
        show_statistical_terminologies()
    elif selected_section == "Feedback":
        show_feedback()

if __name__ == "__main__":
    main()
