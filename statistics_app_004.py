

""" Original file is located at
    https://colab.research.google.com/drive/1dXKlxA49KN11urPw60ECmFUi3ys8othS
"""

# Comprehensive Statistics Visualization Platform
# This application provides interactive learning for statistical concepts
# with real-world examples and visualizations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
import warnings
warnings.filterwarnings('ignore')  # Suppress warning messages for cleaner output

# Configure Streamlit page settings for better user experience
st.set_page_config(
    page_title="Statistics Visualization Hub",
    page_icon="",
    layout="wide",  # Use full width of browser
    initial_sidebar_state="expanded"  # Show sidebar by default
)

# Custom CSS for enhanced styling and user experience
st.markdown("""
<style>
    /* Main header styling with gradient effect */
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }

    /* Subheader styling */
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin: 1rem 0;
    }

    /* Highlight box for important information */
    .highlight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }

    /* Container for metric displays */
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }

    /* Formula display box */
    .formula-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }

    /* Definition box styling */
    .definition-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Simple explanation box styling */
    .simple-explanation {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data  # Cache data to improve performance by avoiding repeated loading
def load_sample_datasets():
    """
    Load and prepare sample datasets from sklearn and seaborn for statistical analysis.

    This function creates multiple datasets with both numerical and categorical variables
    to demonstrate different types of statistical analysis.

    Returns:
        dict: Dictionary containing multiple prepared datasets
    """
    datasets = {}

    # Load Iris dataset - classic dataset for classification and EDA
    iris = load_iris()
    datasets['iris'] = pd.DataFrame(iris.data, columns=iris.feature_names)
    datasets['iris']['species'] = iris.target_names[iris.target]

    # Load Wine dataset - good for multivariate analysis
    wine = load_wine()
    datasets['wine'] = pd.DataFrame(wine.data, columns=wine.feature_names)
    datasets['wine']['wine_class'] = wine.target

    # Load Breast Cancer dataset - important for medical statistics
    cancer = load_breast_cancer()
    datasets['cancer'] = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    datasets['cancer']['diagnosis'] = cancer.target_names[cancer.target]

    # Load Diabetes dataset - regression analysis example
    diabetes = load_diabetes()
    datasets['diabetes'] = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    datasets['diabetes']['target'] = diabetes.target

    # Load Tips dataset from seaborn - excellent for bivariate analysis
    try:
        tips = sns.load_dataset('tips')
        datasets['tips'] = tips
    except:
        # Create synthetic tips dataset if seaborn dataset fails
        np.random.seed(42)
        n_samples = 244
        datasets['tips'] = pd.DataFrame({
            'total_bill': np.random.normal(20, 8, n_samples),
            'tip': np.random.normal(3, 1.5, n_samples),
            'sex': np.random.choice(['Male', 'Female'], n_samples),
            'smoker': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'day': np.random.choice(['Thur', 'Fri', 'Sat', 'Sun'], n_samples),
            'time': np.random.choice(['Lunch', 'Dinner'], n_samples, p=[0.4, 0.6]),
            'size': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.1, 0.4, 0.2, 0.2, 0.08, 0.02])
        })

    # Load Titanic dataset - great for categorical analysis
    try:
        titanic = sns.load_dataset('titanic')
        datasets['titanic'] = titanic
    except:
        # Create synthetic titanic dataset if seaborn dataset fails
        np.random.seed(42)
        n_samples = 891
        datasets['titanic'] = pd.DataFrame({
            'survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.2, 0.6]),
            'sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'age': np.random.normal(30, 12, n_samples),
            'sibsp': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.7, 0.2, 0.05, 0.03, 0.02]),
            'parch': np.random.choice([0, 1, 2, 3], n_samples, p=[0.8, 0.15, 0.03, 0.02]),
            'fare': np.random.exponential(15, n_samples),
            'embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.2, 0.1, 0.7])
        })

    return datasets

def calculate_statistics(data):
    """
    Calculate comprehensive descriptive statistics for a dataset.

    This function computes all major measures of central tendency and dispersion,
    with proper error handling for edge cases.

    Args:
        data: Array-like data for statistical calculation

    Returns:
        dict: Dictionary containing calculated statistics
    """
    try:
        # Clean the data by removing missing values
        data = pd.Series(data).dropna()
        if len(data) == 0:
            return {}

        # Calculate mode using scipy.stats with proper handling
        mode_result = stats.mode(data, keepdims=True)
        mode_value = mode_result.mode[0] if len(mode_result.mode) > 0 else np.nan

        # Return comprehensive statistics dictionary
        return {
            'mean': np.mean(data),           # Average value
            'median': np.median(data),       # Middle value when sorted
            'mode': mode_value,              # Most frequent value
            'std': np.std(data),             # Standard deviation
            'var': np.var(data),             # Variance
            'range': np.max(data) - np.min(data),  # Difference between max and min
            'q1': np.percentile(data, 25),   # First quartile
            'q3': np.percentile(data, 75),   # Third quartile
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),  # Interquartile range
            'cv': (np.std(data) / np.mean(data)) * 100 if np.mean(data) != 0 else 0  # Coefficient of variation
        }
    except Exception as e:
        st.error(f"Error calculating statistics: {str(e)}")
        return {}

def create_navigation_buttons():
    """
    Create interactive navigation buttons for section selection.

    This function generates a grid of buttons that allow users to navigate
    between different sections of the application.
    """
    sections = [
        (" Descriptive Statistics", "descriptive"),
        (" Data Types Explorer", "data_types"),
        (" Central Tendency", "central_tendency"),
        (" Dispersion Measures", "dispersion"),
        (" Univariate Analysis", "univariate"),
        (" Bivariate Analysis", "bivariate"),
        (" Quantiles & Percentiles", "quantiles"),
        (" Statistical Terminology", "terminology"),
        (" Feedback", "feedback")
    ]

    st.markdown("###  Choose Your Learning Journey")

    # Create a grid layout for navigation buttons
    for i in range(0, len(sections), 3):
        cols = st.columns(3)
        for j, (title, key) in enumerate(sections[i:i+3]):
            if j < len(cols):
                with cols[j]:
                    # Create button with session state management
                    if st.button(title, key=f"nav_{key}", use_container_width=True):
                        st.session_state.current_section = key
                        st.rerun()

def main():
    """
    Main application function that handles routing between different sections.

    This function manages the overall application flow and displays content
    based on the current section selected by the user.
    """
    # Initialize session state for navigation tracking
    if 'current_section' not in st.session_state:
        st.session_state.current_section = 'home'

    # Route to appropriate section based on user selection
    if st.session_state.current_section == 'home':
        show_home_page()
    elif st.session_state.current_section == 'descriptive':
        show_descriptive_stats()
    elif st.session_state.current_section == 'data_types':
        show_data_types()
    elif st.session_state.current_section == 'central_tendency':
        show_central_tendency()
    elif st.session_state.current_section == 'dispersion':
        show_dispersion()
    elif st.session_state.current_section == 'univariate':
        show_univariate_analysis()
    elif st.session_state.current_section == 'bivariate':
        show_bivariate_analysis()
    elif st.session_state.current_section == 'quantiles':
        show_quantiles()
    elif st.session_state.current_section == 'terminology':
        show_terminology()
    elif st.session_state.current_section == 'feedback':
        show_feedback_form()

def show_home_page():
    """
    Display the main landing page with welcome message and navigation options.

    This page provides an overview of the platform and showcases the real-world
    impact of statistics across various industries.
    """
    # Main header with custom styling
    st.markdown('<h1 class="main-header">Statistics...What?...Why?...Where?</h1>', unsafe_allow_html=True)

    # Welcome message with platform description
    st.markdown("""
    <div class="highlight-box">
        <h2> Welcome to the Interactive Statistics Learning Platform</h2>
        <p>Discover the power of statistics through real-world examples and interactive visualizations.
        Statistics drives billions of dollars in business decisions and shapes critical research worldwide.</p>
    </div>
    """, unsafe_allow_html=True)

    # Place navigation buttons prominently after welcome message
    create_navigation_buttons()

    # Showcase real-world impact of statistics with concrete examples
    st.markdown("##  Statistics in Action: Real-World Impact")

    # Create four columns to display different industry applications
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        ** Business Intelligence**
        - **$35B+** - Amazon's recommendation revenue annually
        - **10-20 hours** saved daily by HelloFresh through analytics
        - **Marketing ROI** improvements generating millions in revenue
        - **A/B testing** increases conversion rates by 20-30%
        """)

    with col2:
        st.markdown("""
        ** Healthcare Revolution**
        - **$200B+** - Annual pharmaceutical research budget
        - **Clinical trials** save millions of lives worldwide
        - **COVID-19 modeling** influenced trillion-dollar policies
        - **Personalized medicine** improves treatment success rates
        """)

    with col3:
        st.markdown("""
        ** Financial Markets**
        - **$100T+** - Global assets managed using statistical models
        - **$6T+** - Insurance premiums based on risk statistics
        - **High-frequency trading** generates billions through algorithms
        - **Credit scoring** affects millions of loan decisions daily
        """)

    with col4:
        st.markdown("""
        ** Sports Analytics**
        - **Multi-billion** dollar sports analytics industry
        - **Player performance** optimization saves teams millions
        - **Draft strategies** based on statistical player analysis
        - **Injury prevention** through statistical health monitoring
        """)

    # Platform statistics to show comprehensive coverage
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(" Datasets", "6+", delta="Real-world data")
    with col2:
        st.metric(" Visualizations", "25+", delta="Interactive plots")
    with col3:
        st.metric(" Statistical Measures", "20+", delta="Key concepts")
    with col4:
        st.metric(" Learning Sections", "9", delta="Comprehensive coverage")

def show_central_tendency():
    """
    Display interactive central tendency measures with visualizations.

    This section teaches mean, median, and mode concepts with real-world applications,
    but excludes mode calculation and plotting as requested.
    """
    st.markdown('<h1 class="main-header"> Measures of Central Tendency</h1>', unsafe_allow_html=True)

    # Back button for navigation
    if st.button(" Back to Home", key="back_central"):
        st.session_state.current_section = 'home'
        st.rerun()

    # Interactive controls in sidebar for parameter adjustment
    st.sidebar.subheader(" Interactive Controls")
    distribution_type = st.sidebar.selectbox("Choose distribution type:",
                                           ["Normal", "Right Skewed", "Left Skewed", "Bimodal", "Uniform"])
    sample_size = st.sidebar.slider("Sample size:", 100, 2000, 500)

    # Generate data based on user selection with different distribution patterns
    np.random.seed(42)  # Ensure reproducible results
    if distribution_type == "Normal":
        data = np.random.normal(50, 15, sample_size)
        description = "Normal distribution: symmetric, bell-shaped"
    elif distribution_type == "Right Skewed":
        data = np.random.exponential(2, sample_size) * 10 + 20
        description = "Right skewed: tail extends to the right"
    elif distribution_type == "Left Skewed":
        data = 100 - np.random.exponential(2, sample_size) * 10
        description = "Left skewed: tail extends to the left"
    elif distribution_type == "Uniform":
        data = np.random.uniform(20, 80, sample_size)
        description = "Uniform distribution: all values equally likely"
    else:  # Bimodal
        data1 = np.random.normal(30, 8, sample_size//2)
        data2 = np.random.normal(70, 8, sample_size//2)
        data = np.concatenate([data1, data2])
        description = "Bimodal distribution: two peaks"

    # Calculate only mean and median (mode excluded as requested)
    stats_dict = calculate_statistics(data)
    if not stats_dict:
        return

    mean_val = stats_dict['mean']
    median_val = stats_dict['median']

    st.markdown(f"**Distribution Type:** {distribution_type} - {description}")

    # Display measures with comprehensive explanations
    st.markdown("##  Central Tendency Measures")

    col1, col2, col3 = st.columns(3)

    # Mean explanation with business context
    with col1:
        st.markdown(f"""
        ** Mean (Average): {mean_val:.2f}**

        **Formula:** μ = Σx / n

        **When to use:** With symmetric data, no extreme outliers

        **Business Applications:**
        - Salary negotiations and benchmarking
        - Production target setting
        - Budget planning and forecasting
        - Performance bonus calculations

        **Simple Explanation:** Add all numbers and divide by how many numbers you have.
        Like finding the average test score in your class.
        """)

    # Median explanation with business context
    with col2:
        st.markdown(f"""
        ** Median (Middle Value): {median_val:.2f}**

        **Formula:** Middle value when data is sorted

        **When to use:** With skewed data or when outliers are present

        **Business Applications:**
        - Real estate pricing analysis
        - Income and wage studies
        - Government policy decisions
        - Market research insights

        **Simple Explanation:** Line up all numbers from smallest to largest and pick the middle one.
        Like finding the middle person's height when everyone stands in order.
        """)

    # Mode explanation without calculation (as requested)
    with col3:
        st.markdown(f"""
        ** Mode (Most Common)**

        **Formula:** Most frequently occurring value

        **When to use:** With categorical data or to find most popular choice

        **Business Applications:**
        - Inventory planning (most popular sizes)
        - Product preference analysis
        - Service optimization
        - Customer behavior patterns

        **Simple Explanation:** The number that appears most often.
        Like finding the most popular pizza topping ordered at a restaurant.
        """)

    # Enhanced visualization with only mean and median (mode excluded from plot)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Main distribution plot without mode line
    ax1.hist(data, bins=30, alpha=0.7, color='lightblue', edgecolor='black', density=True)
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=3, label=f'Mean: {mean_val:.2f}')
    ax1.axvline(median_val, color='green', linestyle='--', linewidth=3, label=f'Median: {median_val:.2f}')

    ax1.set_title(f'{distribution_type} Distribution - Central Tendency Measures', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Values')
    ax1.set_ylabel('Density')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Box plot for additional insight
    ax2.boxplot(data, vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightcoral', alpha=0.7))
    ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax2.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    ax2.set_title('Box Plot View')
    ax2.set_xlabel('Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Interpretation guide for understanding results
    st.markdown("## Interpretation Guide")

    skewness = stats.skew(data)
    if abs(skewness) < 0.5:
        skew_interpretation = "approximately symmetric (mean ≈ median)"
        skew_advice = "Either mean or median is appropriate"
    elif skewness > 0.5:
        skew_interpretation = "right-skewed (mean > median)"
        skew_advice = "Median is preferred as it's less affected by outliers"
    else:
        skew_interpretation = "left-skewed (mean < median)"
        skew_advice = "Median is preferred as it's less affected by outliers"

    st.markdown(f"""
    **Distribution Analysis:**
    - **Skewness:** {skewness:.3f} - This distribution is {skew_interpretation}
    - **Recommendation:** {skew_advice}
    - **Sample Size:** {sample_size} observations
    """)

def show_data_types():
    """
    Display comprehensive data types exploration with interactive examples.

    This section covers all major data types including ordinal, nominal,
    discrete, continuous, observational, and experimental data.
    """
    st.markdown('<h1 class="main-header"> Data Types Explorer</h1>', unsafe_allow_html=True)

    # Back button for navigation
    if st.button(" Back to Home", key="back_data_types"):
        st.session_state.current_section = 'home'
        st.rerun()

    st.markdown("""
    ## Understanding Data Types

    Data types are the foundation of statistical analysis. Understanding them helps you choose the right analytical techniques and visualizations.
    """)

    # Enhanced data type tabs with comprehensive coverage
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        " Quantitative Data",
        " Qualitative Data",
        " Discrete vs Continuous",
        " Nominal vs Ordinal",
        " Observational vs Experimental",
        " Time Series"
    ])

    # Quantitative data explanation with examples
    with tab1:
        st.subheader(" Quantitative Data (Numerical)")

        st.markdown("""
        **Definition:** Data that represents quantities and can be measured numerically.
        Mathematical operations (addition, subtraction, etc.) can be performed on this data.

        **Simple Explanation:** Think of quantitative data as anything you can count or measure -
        like your height, the temperature outside, or how much money you have.
        """)

        # Generate interactive sample for demonstration
        np.random.seed(42)
        sample_size = st.slider("Sample size:", 100, 2000, 1000, key="quant_sample")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Real-world examples:**
            - **Manufacturing:** Product weights, dimensions, defect rates
            - **Finance:** Stock prices, trading volumes, profit margins
            - **Healthcare:** Patient vital signs, test results, medication dosages
            - **Sports:** Player statistics, game scores, performance metrics
            - **Education:** Test scores, GPA, study hours
            """)

        with col2:
            # Manufacturing quality control example with visualization
            manufacturing_data = np.random.normal(100, 5, sample_size)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(manufacturing_data, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            ax.axvline(np.mean(manufacturing_data), color='red', linestyle='--',
                      label=f'Mean: {np.mean(manufacturing_data):.2f}g')
            ax.set_title('Manufacturing: Product Weight Distribution')
            ax.set_xlabel('Weight (grams)')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)

    # Qualitative data explanation with examples
    with tab2:
        st.subheader(" Qualitative Data (Categorical)")

        st.markdown("""
        **Definition:** Data that represents categories, groups, or characteristics that cannot be measured numerically.
        Used for labeling or describing attributes.

        **Simple Explanation:** Qualitative data describes qualities or characteristics -
        like your favorite color, the brand of your phone, or your satisfaction level.
        """)

        # Interactive categorical data example
        categories = ['Excellent', 'Good', 'Fair', 'Poor']
        sample_size_qual = st.slider("Number of responses:", 100, 1000, 500, key="qual_sample")
        customer_satisfaction = np.random.choice(categories, sample_size_qual, p=[0.3, 0.4, 0.2, 0.1])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Real-world examples:**
            - **Customer Feedback:** Excellent, Good, Fair, Poor
            - **Demographics:** Gender, Race, Education Level
            - **Business:** Department, Job Title, Product Category
            - **Geography:** Country, City, Region
            - **Preferences:** Brand loyalty, Political affiliation
            """)

        with col2:
            # Customer satisfaction visualization
            satisfaction_counts = pd.Series(customer_satisfaction).value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
            satisfaction_counts.plot(kind='bar', ax=ax, color=colors)
            ax.set_title('Customer Satisfaction Survey Results')
            ax.set_xlabel('Rating')
            ax.set_ylabel('Count')
            ax.set_xticklabels(satisfaction_counts.index, rotation=45)

            # Add percentage labels on bars for better understanding
            for i, v in enumerate(satisfaction_counts.values):
                ax.text(i, v + 5, f'{v/sample_size_qual*100:.1f}%', ha='center')

            st.pyplot(fig)

    # Discrete vs Continuous data comparison
    with tab3:
        st.subheader(" Discrete vs Continuous Data")

        col1, col2 = st.columns(2)

        # Discrete data explanation and visualization
        with col1:
            st.markdown("""
            **Discrete Data:** Can only take specific, countable values with gaps between them.
            Usually obtained by counting.

            **Simple Explanation:** Discrete data comes in whole numbers you can count -
            like the number of students in a class (you can't have 23.5 students!).

            **Examples:**
            - Number of children in a family
            - Number of cars sold
            - Dice roll outcomes
            - Number of books on a shelf
            - Customer complaints per day
            """)

            # Discrete data visualization (Poisson distribution)
            discrete_data = np.random.poisson(3, 500)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(discrete_data, bins=range(0, max(discrete_data)+2), alpha=0.7, color='lightcoral', edgecolor='black')
            ax.set_title('Discrete Data: Number of Daily Customer Calls')
            ax.set_xlabel('Number of Calls')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        # Continuous data explanation and visualization
        with col2:
            st.markdown("""
            **Continuous Data:** Can take any value within a range, including decimal places.
            Usually obtained by measuring.

            **Simple Explanation:** Continuous data can be any number with decimals -
            like your height (you could be 5.7834 feet tall).

            **Examples:**
            - Height and weight
            - Temperature
            - Time duration
            - Distance traveled
            - Stock prices
            """)

            # Continuous data visualization (Normal distribution)
            continuous_data = np.random.normal(70, 10, 500)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(continuous_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.set_title('Continuous Data: Student Heights (inches)')
            ax.set_xlabel('Height (inches)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

    # Nominal vs Ordinal data comparison
    with tab4:
        st.subheader(" Nominal vs Ordinal Data")

        col1, col2 = st.columns(2)

        # Nominal data explanation and visualization
        with col1:
            st.markdown("""
            **Nominal Data:** Categories with no inherent order or ranking.
            Categories are just different, not better or worse.

            **Simple Explanation:** Nominal data is like different flavors of ice cream -
            vanilla isn't "better" than chocolate, they're just different categories.

            **Examples:**
            - Eye color (Blue, Brown, Green)
            - Gender (Male, Female, Other)
            - Marital status (Single, Married, Divorced)
            - Blood type (A, B, AB, O)
            - Product categories (Electronics, Clothing, Books)
            """)

            # Nominal data visualization
            eye_colors = ['Brown', 'Blue', 'Green', 'Hazel', 'Gray']
            eye_color_counts = np.random.multinomial(400, [0.35, 0.25, 0.15, 0.20, 0.05])

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#8B4513', '#4169E1', '#228B22', '#CD853F', '#808080']
            bars = ax.bar(eye_colors, eye_color_counts, color=colors)
            ax.set_title('Nominal Data: Eye Color Distribution')
            ax.set_ylabel('Count')
            st.pyplot(fig)

        # Ordinal data explanation and visualization
        with col2:
            st.markdown("""
            **Ordinal Data:** Categories with a meaningful order or ranking.
            You can say one category is "higher" or "better" than another.

            **Simple Explanation:** Ordinal data is like movie ratings - 5 stars is definitely
            better than 1 star, and there's a clear order from worst to best.

            **Examples:**
            - Education level (High School < Bachelor's < Master's < PhD)
            - Customer satisfaction (Poor < Fair < Good < Excellent)
            - Movie ratings (1 star < 2 stars < 3 stars < 4 stars < 5 stars)
            - Income brackets (Low < Medium < High)
            - Survey responses (Strongly Disagree < Disagree < Neutral < Agree < Strongly Agree)
            """)

            # Ordinal data visualization
            satisfaction_levels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
            satisfaction_counts = [50, 100, 200, 250, 150]

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
            bars = ax.bar(satisfaction_levels, satisfaction_counts, color=colors)
            ax.set_title('Ordinal Data: Customer Satisfaction Levels')
            ax.set_ylabel('Count')
            ax.set_xticklabels(satisfaction_levels, rotation=45)
            st.pyplot(fig)

    # Observational vs Experimental data comparison
    with tab5:
        st.subheader(" Observational vs Experimental Data")

        col1, col2 = st.columns(2)

        # Observational data explanation
        with col1:
            st.markdown("""
            **Observational Data:** Data collected by observing subjects without manipulating
            or controlling any variables. Researchers observe what naturally occurs.

            **Simple Explanation:** Like being a detective - you watch and record what happens
            naturally without interfering. Like observing how many people wear masks in a store.

            **Characteristics:**
            - No manipulation of variables
            - Cannot establish causation (only correlation)
            - Reflects real-world conditions
            - May have confounding variables

            **Examples:**
            - Survey responses about shopping habits
            - Medical records analysis
            - Social media behavior tracking
            - Weather pattern observations
            - Traffic flow studies

            **Advantages:**
            -  Ethical (no manipulation)
            -  Real-world applicability
            -  Large sample sizes possible
            -  Cost-effective

            **Disadvantages:**
            -  Cannot prove causation
            -  Confounding variables
            -  Less control over conditions
            """)

        # Experimental data explanation
        with col2:
            st.markdown("""
            **Experimental Data:** Data collected through controlled experiments where researchers
            manipulate one or more variables to observe their effects.

            **Simple Explanation:** Like a science experiment - you change one thing and see what happens.
            Like testing if a new medicine works by giving it to some patients and not others.

            **Characteristics:**
            - Deliberate manipulation of variables
            - Can establish causation
            - Control groups used
            - Random assignment of subjects

            **Examples:**
            - Clinical drug trials
            - A/B testing for websites
            - Educational intervention studies
            - Marketing campaign experiments
            - Laboratory experiments

            **Advantages:**
            -  Can establish causation
            -  High internal validity
            -  Control over variables
            -  Reproducible results

            **Disadvantages:**
            -  May be unethical
            -  Artificial conditions
            -  Expensive and time-consuming
            -  Limited sample sizes
            """)

        # Comparison table for better understanding
        st.markdown("###  Key Differences Summary")

        comparison_data = {
            'Aspect': ['Control over variables', 'Causation', 'Cost', 'Ethics', 'Real-world applicability'],
            'Observational': ['Low', 'Cannot establish', 'Low', 'High', 'High'],
            'Experimental': ['High', 'Can establish', 'High', 'May be limited', 'May be limited']
        }

        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)

    # Time series data explanation and visualization
    with tab6:
        st.subheader(" Time Series Data")

        st.markdown("""
        **Definition:** Data points collected or recorded at successive time intervals.
        The order of observations matters, and time is a crucial component.

        **Simple Explanation:** Time series data is like a diary of numbers - it shows how something
        changes over time, like tracking your daily steps or a company's monthly sales.
        """)

        # Interactive time series parameters
        days = st.slider("Number of days to display:", 30, 365, 100, key="time_series_days")
        dates = pd.date_range('2023-01-01', periods=days, freq='D')

        # Different time series pattern options
        trend_type = st.selectbox("Select pattern type:",
                                 ["Upward Trend", "Seasonal Pattern", "Random Walk", "Cyclical Pattern"])

        # Generate different patterns based on selection
        if trend_type == "Upward Trend":
            # Sales growth pattern
            base_sales = 1000
            growth_rate = 0.01
            noise = np.random.normal(0, 50, days)
            sales_data = [base_sales * (1 + growth_rate) ** i + noise[i] for i in range(days)]
            title = "Daily Sales Revenue (Upward Trend)"
            ylabel = "Sales ($)"

        elif trend_type == "Seasonal Pattern":
            # Temperature with seasonal variation
            base_temp = 60
            seasonal = 20 * np.sin(2 * np.pi * np.arange(days) / 365)
            noise = np.random.normal(0, 3, days)
            sales_data = base_temp + seasonal + noise
            title = "Daily Temperature (Seasonal Pattern)"
            ylabel = "Temperature (°F)"

        elif trend_type == "Random Walk":
            # Stock price random walk
            sales_data = np.cumsum(np.random.randn(days)) + 100
            title = "Stock Price (Random Walk)"
            ylabel = "Price ($)"

        else:  # Cyclical Pattern
            # Business cycle pattern
            cycle = 500 * np.sin(2 * np.pi * np.arange(days) / 60)
            trend = np.arange(days) * 2
            noise = np.random.normal(0, 100, days)
            sales_data = 2000 + cycle + trend + noise
            title = "Monthly Revenue (Cyclical Pattern)"
            ylabel = "Revenue ($)"

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Key Characteristics:**
            - **Temporal ordering:** Order of observations matters
            - **Trends:** Long-term increases or decreases
            - **Seasonality:** Regular patterns that repeat
            - **Cycles:** Irregular fluctuations
            - **Autocorrelation:** Current values depend on past values

            **Real-world examples:**
            - **Business:** Daily sales, quarterly profits
            - **Finance:** Stock prices, currency exchange rates
            - **Healthcare:** Patient vital signs over time
            - **Environment:** Temperature, rainfall patterns
            - **Technology:** Website traffic, app usage
            """)

        with col2:
            # Time series visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(dates, sales_data, linewidth=2, color='#2ecc71')
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Time series components explanation
        st.markdown("###  Time Series Components")

        components_info = {
            'Component': ['Trend', 'Seasonality', 'Cyclical', 'Irregular (Noise)'],
            'Description': [
                'Long-term increase or decrease in data',
                'Regular, predictable patterns (daily, monthly, yearly)',
                'Fluctuations with no fixed period',
                'Random variation that cannot be explained'
            ],
            'Example': [
                'Population growth over decades',
                'Ice cream sales higher in summer',
                'Economic boom and bust cycles',
                'Unexpected events affecting sales'
            ]
        }

        components_df = pd.DataFrame(components_info)
        st.table(components_df)

def show_univariate_analysis():
    """
    Display comprehensive univariate analysis with multiple plot types.

    This section provides various visualization options for both numerical
    and categorical variables with detailed analysis capabilities.
    """
    st.markdown('<h1 class="main-header"> Univariate Analysis</h1>', unsafe_allow_html=True)

    # Back button for navigation
    if st.button(" Back to Home", key="back_univariate"):
        st.session_state.current_section = 'home'
        st.rerun()

    # Load datasets for analysis
    datasets = load_sample_datasets()

    # User controls for dataset and analysis selection
    dataset_choice = st.selectbox("Choose a dataset:", list(datasets.keys()))
    selected_data = datasets[dataset_choice]

    # Separate numeric and categorical columns for appropriate analysis
    numeric_cols = selected_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = selected_data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Add object columns that might be categorical
    for col in selected_data.columns:
        if selected_data[col].dtype == 'object' and col not in categorical_cols:
            categorical_cols.append(col)

    analysis_type = st.radio("Select analysis type:", ["Numerical Variable", "Categorical Variable"])

    # Numerical variable analysis with multiple plot options
    if analysis_type == "Numerical Variable" and numeric_cols:
        selected_column = st.selectbox("Select numerical column:", numeric_cols)
        data = selected_data[selected_column].dropna()

        # Interactive controls for customization
        col1, col2 = st.columns(2)
        with col1:
            bins = st.slider("Number of bins:", 10, 50, 30)
        with col2:
            plot_type = st.selectbox("Plot type:", [
                "Histogram", "Density Plot", "Box Plot", "Violin Plot",
                "Q-Q Plot", "Strip Plot", "Rug Plot", "ECDF Plot"
            ])

        # Create enhanced visualizations based on user selection
        fig, ax = plt.subplots(figsize=(12, 6))

        if plot_type == "Histogram":
            # Standard histogram with statistical overlays
            n, bins_used, patches = ax.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'Histogram of {selected_column}')
            ax.set_xlabel(selected_column)
            ax.set_ylabel('Frequency')

            # Add mean and median lines for reference
            mean_val = np.mean(data)
            median_val = np.median(data)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            ax.legend()

        elif plot_type == "Density Plot":
            # Kernel density estimation plot
            sns.histplot(data, kde=True, ax=ax, stat='density')
            ax.set_title(f'Density Plot of {selected_column}')

        elif plot_type == "Box Plot":
            # Box plot with custom styling
            box_plot = ax.boxplot(data, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightcoral')
            ax.set_title(f'Box Plot of {selected_column}')
            ax.set_ylabel(selected_column)

        elif plot_type == "Violin Plot":
            # Violin plot showing distribution shape
            parts = ax.violinplot(data, positions=[1], widths=[0.6])
            for pc in parts['bodies']:
                pc.set_facecolor('lightgreen')
                pc.set_alpha(0.7)
            ax.set_title(f'Violin Plot of {selected_column}')
            ax.set_ylabel(selected_column)

        elif plot_type == "Q-Q Plot":
            # Quantile-quantile plot for normality assessment
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot of {selected_column}')

        elif plot_type == "Strip Plot":
            # Strip plot with jittering
            y_pos = np.random.normal(0, 0.1, len(data))
            ax.scatter(data, y_pos, alpha=0.6, color='orange')
            ax.set_title(f'Strip Plot of {selected_column}')
            ax.set_xlabel(selected_column)
            ax.set_ylabel('Jittered Y')

        elif plot_type == "Rug Plot":
            # Histogram with rug plot overlay
            ax.hist(data, bins=bins, alpha=0.7, color='lightblue', edgecolor='black')
            ax2 = ax.twinx()
            ax2.set_ylim(0, 1)
            # Sample data points for performance with large datasets
            for value in data[::max(1, len(data)//100)]:
                ax2.axvline(value, ymin=0, ymax=0.1, color='red', alpha=0.3)
            ax.set_title(f'Histogram with Rug Plot of {selected_column}')
            ax2.set_ylabel('Rug')

        else:  # ECDF Plot
            # Empirical Cumulative Distribution Function
            sorted_data = np.sort(data)
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, y, marker='.', linestyle='none')
            ax.set_title(f'Empirical CDF of {selected_column}')
            ax.set_xlabel(selected_column)
            ax.set_ylabel('Cumulative Probability')
            ax.grid(True, alpha=0.3)

        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Enhanced distribution analysis with multiple statistical measures
        st.subheader(" Distribution Analysis")

        stats_dict = calculate_statistics(data)
        if stats_dict:
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)

            col1, col2, col3, col4 = st.columns(4)

            # Skewness interpretation
            with col1:
                st.metric("Skewness", f"{skewness:.3f}")
                if skewness > 0.5:
                    st.write("🔴 Right-skewed")
                elif skewness < -0.5:
                    st.write("🔵 Left-skewed")
                else:
                    st.write("🟢 Approximately symmetric")

            # Kurtosis interpretation
            with col2:
                st.metric("Kurtosis", f"{kurtosis:.3f}")
                if kurtosis > 0:
                    st.write(" Heavy-tailed")
                else:
                    st.write(" Light-tailed")

            # Normality test results
            with col3:
                _, p_value = stats.normaltest(data)
                st.metric("Normality p-value", f"{p_value:.4f}")
                if p_value > 0.05:
                    st.write(" Likely normal")
                else:
                    st.write(" Not normal")

            # Outlier detection using IQR method
            with col4:
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
                st.metric("Outliers", len(outliers))
                st.write(f"({len(outliers)/len(data)*100:.1f}% of data)")

    # Categorical variable analysis with multiple chart options
    elif analysis_type == "Categorical Variable" and categorical_cols:
        selected_column = st.selectbox("Select categorical column:", categorical_cols)
        data = selected_data[selected_column].value_counts()

        # Visualization options for categorical data
        plot_type = st.selectbox("Plot type:", ["Bar Chart", "Horizontal Bar", "Pie Chart", "Donut Chart"])

        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "Bar Chart":
            # Vertical bar chart with value labels
            bars = data.plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title(f'Bar Chart of {selected_column}')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)

            # Add value labels on bars for clarity
            for i, v in enumerate(data.values):
                ax.text(i, v + max(data.values) * 0.01, str(v), ha='center', va='bottom')

        elif plot_type == "Horizontal Bar":
            # Horizontal bar chart
            data.plot(kind='barh', ax=ax, color='lightgreen')
            ax.set_title(f'Horizontal Bar Chart of {selected_column}')
            ax.set_xlabel('Count')

        elif plot_type == "Pie Chart":
            # Standard pie chart
            wedges, texts, autotexts = ax.pie(data.values, labels=data.index, autopct='%1.1f%%')
            ax.set_title(f'Pie Chart of {selected_column}')

        else:  # Donut Chart
            # Donut chart (pie chart with center removed)
            wedges, texts, autotexts = ax.pie(data.values, labels=data.index, autopct='%1.1f%%')
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            ax.add_artist(centre_circle)
            ax.set_title(f'Donut Chart of {selected_column}')

        plt.tight_layout()
        st.pyplot(fig)

        # Categorical analysis with summary statistics
        st.subheader(" Categorical Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Category Statistics:**")
            st.write(f"**Number of categories:** {len(data)}")
            st.write(f"**Most common:** {data.index[0]} ({data.iloc[0]} occurrences)")
            st.write(f"**Least common:** {data.index[-1]} ({data.iloc[-1]} occurrences)")
            st.write(f"**Total observations:** {data.sum()}")

        with col2:
            st.markdown("**Frequency Table:**")
            freq_table = pd.DataFrame({
                'Category': data.index,
                'Count': data.values,
                'Percentage': (data.values / data.sum() * 100).round(2)
            })
            st.dataframe(freq_table)

def show_bivariate_analysis():
    """
    Display comprehensive bivariate analysis for different variable type combinations.

    This section handles numerical-numerical, categorical-numerical, and
    categorical-categorical relationships with appropriate visualizations and tests.
    """
    st.markdown('<h1 class="main-header"> Bivariate Analysis</h1>', unsafe_allow_html=True)

    # Back button for navigation
    if st.button(" Back to Home", key="back_bivariate"):
        st.session_state.current_section = 'home'
        st.rerun()

    st.markdown("""
    ## Understanding Bivariate Analysis

    Bivariate analysis examines the relationship between two variables. The type of analysis depends on
    the data types of both variables.
    """)

    # Load datasets with good categorical-numerical combinations
    datasets = load_sample_datasets()

    # Dataset selection
    dataset_choice = st.selectbox("Choose a dataset:", list(datasets.keys()))
    selected_data = datasets[dataset_choice]

    # Identify variable types for appropriate analysis
    numeric_cols = selected_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = selected_data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Ensure object columns are treated as categorical
    for col in selected_data.columns:
        if selected_data[col].dtype == 'object' and col not in categorical_cols:
            categorical_cols.append(col)

    analysis_type = st.selectbox("Select relationship type:",
                                ["Numerical vs Numerical", "Categorical vs Numerical", "Categorical vs Categorical"])

    # Numerical vs Numerical analysis
    if analysis_type == "Numerical vs Numerical" and len(numeric_cols) >= 2:
        st.subheader(" Numerical vs Numerical Analysis")

        st.markdown("""
        **What it means:** Analyzing the relationship between two continuous variables to understand
        how they change together. This helps identify patterns, trends, and correlations.

        **Simple Explanation:** Like checking if taller people tend to weigh more,
        or if students who study more hours get better grades.
        """)

        # Variable selection for correlation analysis
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Select X variable:", numeric_cols, key="x_var_num")
        with col2:
            y_var = st.selectbox("Select Y variable:", [col for col in numeric_cols if col != x_var], key="y_var_num")

        # Create enhanced scatter plot with regression analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Clean data for analysis
        x_data = selected_data[x_var].dropna()
        y_data = selected_data[y_var].dropna()

        # Ensure same length for correlation analysis
        min_len = min(len(x_data), len(y_data))
        if min_len > 0:
            x_data = x_data.iloc[:min_len]
            y_data = y_data.iloc[:min_len]

            # Scatter plot with regression line
            ax1.scatter(x_data, y_data, alpha=0.6, color='blue')

            # Add regression line for trend visualization
            try:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax1.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
                ax1.set_xlabel(x_var)
                ax1.set_ylabel(y_var)
                ax1.set_title(f'Scatter Plot: {x_var} vs {y_var}')
                ax1.grid(True, alpha=0.3)

                # Correlation calculation and interpretation
                correlation = np.corrcoef(x_data, y_data)[0, 1]

                # Hexbin plot for density visualization
                hb = ax2.hexbin(x_data, y_data, gridsize=20, cmap='Blues')
                ax2.set_xlabel(x_var)
                ax2.set_ylabel(y_var)
                ax2.set_title(f'Density Plot: {x_var} vs {y_var}')
                plt.colorbar(hb, ax=ax2)

                plt.tight_layout()
                st.pyplot(fig)

                # Correlation interpretation with statistical significance
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Pearson Correlation", f"{correlation:.3f}")

                with col2:
                    # Correlation strength interpretation
                    if abs(correlation) > 0.7:
                        strength = "Strong"
                        color = "🔴"
                    elif abs(correlation) > 0.3:
                        strength = "Moderate"
                        color = "🟡"
                    else:
                        strength = "Weak"
                        color = "🟢"
                    st.write(f"{color} {strength} correlation")

                with col3:
                    direction = "Positive" if correlation > 0 else "Negative"
                    st.write(f" {direction} relationship")

                # Detailed interpretation for understanding
                st.markdown("### 🔍 Interpretation:")
                if correlation > 0.7:
                    interpretation = f"Strong positive relationship: As {x_var} increases, {y_var} tends to increase significantly."
                elif correlation > 0.3:
                    interpretation = f"Moderate positive relationship: As {x_var} increases, {y_var} tends to increase somewhat."
                elif correlation > -0.3:
                    interpretation = f"Weak relationship: {x_var} and {y_var} don't show a clear linear relationship."
                elif correlation > -0.7:
                    interpretation = f"Moderate negative relationship: As {x_var} increases, {y_var} tends to decrease somewhat."
                else:
                    interpretation = f"Strong negative relationship: As {x_var} increases, {y_var} tends to decrease significantly."

                st.write(interpretation)

            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")

    # Categorical vs Numerical analysis
    elif analysis_type == "Categorical vs Numerical":
        st.subheader(" Categorical vs Numerical Analysis")

        st.markdown("""
        **What it means:** Comparing numerical values across different categories to understand
        how categories differ in terms of the numerical variable.

        **Simple Explanation:** Like comparing average salaries across different departments,
        or test scores between different schools.
        """)

        if categorical_cols and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                cat_var = st.selectbox("Select categorical variable:", categorical_cols, key="cat_var")
            with col2:
                num_var = st.selectbox("Select numerical variable:", numeric_cols, key="num_var")

            # Enhanced visualizations for categorical-numerical relationships
            plot_choice = st.selectbox("Choose visualization:",
                                     ["Box Plot", "Violin Plot", "Strip Plot", "Bar Plot", "Swarm Plot"])

            fig, ax = plt.subplots(figsize=(12, 6))

            try:
                # Clean data for analysis
                clean_data = selected_data[[cat_var, num_var]].dropna()

                if len(clean_data) > 0:
                    # Generate appropriate plot based on user selection
                    if plot_choice == "Box Plot":
                        sns.boxplot(data=clean_data, x=cat_var, y=num_var, ax=ax)
                        ax.set_title(f'Box Plot: {num_var} by {cat_var}')

                    elif plot_choice == "Violin Plot":
                        sns.violinplot(data=clean_data, x=cat_var, y=num_var, ax=ax)
                        ax.set_title(f'Violin Plot: {num_var} by {cat_var}')

                    elif plot_choice == "Strip Plot":
                        sns.stripplot(data=clean_data, x=cat_var, y=num_var, ax=ax, alpha=0.7)
                        ax.set_title(f'Strip Plot: {num_var} by {cat_var}')

                    elif plot_choice == "Bar Plot":
                        sns.barplot(data=clean_data, x=cat_var, y=num_var, ax=ax)
                        ax.set_title(f'Bar Plot: Average {num_var} by {cat_var}')

                    else:  # Swarm Plot
                        # Use swarm plot for smaller datasets, strip plot for larger ones
                        if len(clean_data) < 1000:
                            sns.swarmplot(data=clean_data, x=cat_var, y=num_var, ax=ax)
                            ax.set_title(f'Swarm Plot: {num_var} by {cat_var}')
                        else:
                            sns.stripplot(data=clean_data, x=cat_var, y=num_var, ax=ax, alpha=0.7)
                            ax.set_title(f'Strip Plot: {num_var} by {cat_var} (Swarm not suitable for large data)')

                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Group statistics for detailed comparison
                    group_stats = clean_data.groupby(cat_var)[num_var].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
                    st.subheader(" Group Statistics")
                    st.dataframe(group_stats)

                    # Statistical significance testing
                    groups = [group[num_var].dropna() for name, group in clean_data.groupby(cat_var)]
                    if len(groups) >= 2 and all(len(group) > 1 for group in groups):
                        if len(groups) == 2:
                            # T-test for two groups
                            statistic, p_value = stats.ttest_ind(groups[0], groups[1])
                            test_name = "T-test"
                        else:
                            # ANOVA for multiple groups
                            statistic, p_value = stats.f_oneway(*groups)
                            test_name = "ANOVA"

                        st.subheader(f" Statistical Test ({test_name})")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Test Statistic", f"{statistic:.4f}")
                        with col2:
                            st.metric("P-value", f"{p_value:.4f}")

                        # Interpretation of statistical significance
                        if p_value < 0.05:
                            st.write(" **Significant difference** between groups (p < 0.05)")
                        else:
                            st.write(" **No significant difference** between groups (p ≥ 0.05)")

            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")

    # Categorical vs Categorical analysis
    elif analysis_type == "Categorical vs Categorical":
        st.subheader(" Categorical vs Categorical Analysis")

        st.markdown("""
        **What it means:** Examining the relationship between two categorical variables to understand
        if they are independent or if there's an association between them.

        **Simple Explanation:** Like checking if gender is related to preferred movie genre,
        or if education level is associated with political preference.
        """)

        if len(categorical_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                cat_var1 = st.selectbox("Select first categorical variable:", categorical_cols, key="cat1")
            with col2:
                cat_var2 = st.selectbox("Select second categorical variable:",
                                       [col for col in categorical_cols if col != cat_var1], key="cat2")

            # Cross-tabulation analysis
            try:
                # Clean data for categorical analysis
                clean_data = selected_data[[cat_var1, cat_var2]].dropna()

                if len(clean_data) > 0:
                    crosstab = pd.crosstab(clean_data[cat_var1], clean_data[cat_var2])

                    st.subheader(" Cross-tabulation Table")
                    st.dataframe(crosstab)

                    # Enhanced visualizations for categorical relationships
                    plot_choice = st.selectbox("Choose visualization:",
                                             ["Heatmap", "Stacked Bar", "Grouped Bar"])

                    fig, ax = plt.subplots(figsize=(12, 6))

                    if plot_choice == "Heatmap":
                        # Heatmap showing relationship strength
                        sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_title(f'Heatmap: {cat_var1} vs {cat_var2}')

                    elif plot_choice == "Stacked Bar":
                        # Stacked bar chart showing proportions
                        crosstab.plot(kind='bar', stacked=True, ax=ax)
                        ax.set_title(f'Stacked Bar Chart: {cat_var1} vs {cat_var2}')
                        ax.legend(title=cat_var2, bbox_to_anchor=(1.05, 1), loc='upper left')

                    else:  # Grouped Bar
                        # Grouped bar chart for comparison
                        crosstab.plot(kind='bar', ax=ax)
                        ax.set_title(f'Grouped Bar Chart: {cat_var1} vs {cat_var2}')
                        ax.legend(title=cat_var2, bbox_to_anchor=(1.05, 1), loc='upper left')

                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Chi-square test for independence
                    if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
                        chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)

                        st.subheader(" Chi-Square Test for Independence")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Chi-Square Statistic", f"{chi2:.4f}")
                        with col2:
                            st.metric("P-value", f"{p_value:.4f}")
                        with col3:
                            st.metric("Degrees of Freedom", dof)

                        # Statistical significance interpretation
                        if p_value < 0.05:
                            st.write(" **Significant association** between variables (p < 0.05)")
                            st.write("The two variables are not independent.")
                        else:
                            st.write(" **No significant association** between variables (p ≥ 0.05)")
                            st.write("The two variables appear to be independent.")

                        # Effect size calculation (Cramér's V)
                        n = crosstab.sum().sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
                        st.metric("Cramér's V (Effect Size)", f"{cramers_v:.4f}")

                        # Effect size interpretation
                        if cramers_v < 0.1:
                            effect_size = "Negligible"
                        elif cramers_v < 0.3:
                            effect_size = "Small"
                        elif cramers_v < 0.5:
                            effect_size = "Medium"
                        else:
                            effect_size = "Large"

                        st.write(f"Effect size: **{effect_size}**")

            except Exception as e:
                st.error(f"Error in categorical analysis: {str(e)}")

def show_quantiles():
    """
    Display comprehensive quantiles and percentiles analysis with interactive examples.

    This section covers quartiles, deciles, quintiles, and percentiles with
    real-world applications and interactive calculations.
    """
    st.markdown('<h1 class="main-header"> Quantiles & Percentiles</h1>', unsafe_allow_html=True)

    # Back button for navigation
    if st.button(" Back to Home", key="back_quantiles"):
        st.session_state.current_section = 'home'
        st.rerun()

    st.markdown("""
    ## Understanding Quantiles and Percentiles

    Quantiles divide a dataset into equal parts, helping us understand the distribution and position of values.
    """)

    # Interactive data generation for demonstration
    np.random.seed(42)
    sample_size = st.slider("Sample size:", 100, 1000, 500)
    distribution_type = st.selectbox("Distribution type:", ["Normal", "Exponential", "Uniform", "Beta"])

    # Generate different distribution types for varied learning
    if distribution_type == "Normal":
        data = np.random.normal(50, 15, sample_size)
    elif distribution_type == "Exponential":
        data = np.random.exponential(2, sample_size) * 10 + 20
    elif distribution_type == "Uniform":
        data = np.random.uniform(10, 90, sample_size)
    else:  # Beta
        data = np.random.beta(2, 5, sample_size) * 100

    # Calculate various quantile types
    quartiles = [np.percentile(data, q) for q in [25, 50, 75]]
    deciles = [np.percentile(data, q) for q in range(10, 100, 10)]
    quintiles = [np.percentile(data, q) for q in [20, 40, 60, 80]]

    # Enhanced quantile information with formulas and applications
    st.markdown("##  Quantile Types and Formulas")

    quantile_tabs = st.tabs([" Quartiles", " Deciles", " Quintiles", " Percentiles"])

    # Quartiles explanation and visualization
    with quantile_tabs[0]:
        st.subheader(" Quartiles")

        st.markdown("""
        **Definition:** Quartiles divide data into four equal parts. Each part contains 25% of the data.

        **Formulas:**
        - Q1 = P25 (25th percentile)
        - Q2 = P50 (50th percentile = Median)
        - Q3 = P75 (75th percentile)
        - IQR = Q3 - Q1 (Interquartile Range)

        **Simple Explanation:** Imagine lining up all students by test score.
        Quartiles tell you the scores that separate the bottom 25%, middle 50%, and top 25%.
        """)

        col1, col2 = st.columns(2)

        with col1:
            # Display quartile values and business applications
            for i, q in enumerate(quartiles, 1):
                st.write(f"**Q{i} ({i*25}th percentile):** {q:.2f}")
            st.write(f"**IQR:** {quartiles[2] - quartiles[0]:.2f}")

            st.markdown("""
            **Business Applications:**
            - **Performance quartiles** for employee evaluation
            - **Customer segmentation** by purchase value
            - **Quality control** using quartile ranges
            - **Risk assessment** in finance
            """)

        with col2:
            # Quartile visualization
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(data, bins=30, alpha=0.7, color='lightblue', edgecolor='black')

            colors = ['red', 'orange', 'green']
            for i, q in enumerate(quartiles):
                ax.axvline(q, color=colors[i], linestyle='--', linewidth=2,
                          label=f'Q{i+1}: {q:.2f}')

            ax.set_title('Distribution with Quartile Lines')
            ax.set_xlabel('Values')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # Deciles explanation and visualization
    with quantile_tabs[1]:
        st.subheader(" Deciles")

        st.markdown("""
        **Definition:** Deciles divide data into ten equal parts. Each part contains 10% of the data.

        **Formula:** Di = P(i×10) where i = 1, 2, ..., 9
        Example: D1 = P10, D5 = P50, D9 = P90

        **Simple Explanation:** Like dividing a pizza into 10 equal slices.
        Deciles tell you the values that mark each slice boundary.
        """)

        col1, col2 = st.columns(2)

        with col1:
            # Display decile values and applications
            for i, decile in enumerate(deciles, 1):
                st.write(f"**D{i} ({i*10}th percentile):** {decile:.2f}")

            st.markdown("""
            **Business Applications:**
            - **Customer lifetime value** ranking
            - **Sales performance** evaluation
            - **Market research** segmentation
            - **Academic grading** systems
            """)

        with col2:
            # Decile visualization
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(data, bins=30, alpha=0.7, color='lightyellow', edgecolor='black')

            # Show selected deciles for clarity
            for i, d in enumerate(deciles[::2]):
                ax.axvline(d, color='purple', linestyle=':', alpha=0.7,
                          label=f'D{(i*2)+1}: {d:.1f}')

            ax.set_title('Distribution with Selected Decile Lines')
            ax.set_xlabel('Values')
            ax.set_ylabel('Frequency')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # Quintiles explanation and visualization
    with quantile_tabs[2]:
        st.subheader(" Quintiles")

        st.markdown("""
        **Definition:** Quintiles divide data into five equal parts. Each part contains 20% of the data.

        **Formula:** Qi = P(i×20) where i = 1, 2, 3, 4
        Example: Q1 = P20, Q2 = P40, Q3 = P60, Q4 = P80

        **Simple Explanation:** Like dividing your class into 5 equal groups based on grades.
        Each group has the same number of students.
        """)

        col1, col2 = st.columns(2)

        with col1:
            # Display quintile values and applications
            for i, quintile in enumerate(quintiles, 1):
                st.write(f"**Quintile {i} ({i*20}th percentile):** {quintile:.2f}")

            st.markdown("""
            **Business Applications:**
            - **Investment portfolio** risk categories
            - **Insurance premium** calculations
            - **Market share** analysis
            - **Product rating** systems
            """)

        with col2:
            # Quintile visualization
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')

            colors = ['blue', 'green', 'orange', 'red']
            for i, q in enumerate(quintiles):
                ax.axvline(q, color=colors[i], linestyle='--', linewidth=2,
                          label=f'Q{i+1}: {q:.2f}')

            ax.set_title('Distribution with Quintile Lines')
            ax.set_xlabel('Values')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # Percentiles explanation and interactive calculator
    with quantile_tabs[3]:
        st.subheader(" Percentiles")

        st.markdown("""
        **Definition:** Percentiles divide data into 100 equal parts. The kth percentile is the value below which k% of data falls.

        **Formula:** Pk = Value at position (k/100) × (n+1)
        where k is the percentile (1-99) and n is sample size

        **Simple Explanation:** If you scored in the 90th percentile on a test,
        you did better than 90% of all test takers.
        """)

        # Interactive percentile calculator
        st.subheader(" Interactive Percentile Calculator")

        col1, col2 = st.columns(2)

        with col1:
            # User-interactive percentile calculation
            percentile_value = st.slider("Select percentile:", 1, 99, 50)
            calculated_percentile = np.percentile(data, percentile_value)

            st.metric(f"{percentile_value}th Percentile", f"{calculated_percentile:.2f}")
            st.write(f"**Meaning:** {percentile_value}% of values are below {calculated_percentile:.2f}")

            # Display common percentiles for reference
            st.markdown("**Common Percentiles:**")
            common_percentiles = [10, 25, 50, 75, 90, 95, 99]
            for p in common_percentiles:
                value = np.percentile(data, p)
                st.write(f"P{p}: {value:.2f}")

        with col2:
            # Visual representation of selected percentile
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.axvline(calculated_percentile, color='red', linestyle='--', linewidth=3,
                      label=f'P{percentile_value}: {calculated_percentile:.2f}')

            ax.set_title(f'Distribution with {percentile_value}th Percentile')
            ax.set_xlabel('Values')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # Real-world applications summary
    st.markdown("##  Real-World Applications Summary")

    # Comprehensive applications across different sectors
    applications = {
        "Education": {
            "Percentiles": "SAT/GRE scores, class rankings",
            "Quartiles": "Grade distribution analysis",
            "Impact": "College admissions, scholarship decisions"
        },
        "Healthcare": {
            "Percentiles": "Growth charts for children, BMI categories",
            "Quartiles": "Blood pressure ranges, diagnostic thresholds",
            "Impact": "Early intervention, treatment protocols"
        },
        "Finance": {
            "Quintiles": "Investment risk categories, credit scores",
            "Deciles": "Wealth distribution analysis",
            "Impact": "Portfolio management, loan approvals"
        },
        "Business": {
            "Quartiles": "Customer segmentation, performance evaluation",
            "Percentiles": "Salary benchmarking, KPI tracking",
            "Impact": "Strategic planning, resource allocation"
        }
    }

    # Display applications in expandable sections
    for sector, info in applications.items():
        with st.expander(f" {sector}"):
            for measure, description in info.items():
                if measure != "Impact":
                    st.write(f"**{measure}:** {description}")
            st.write(f"** Business Impact:** {info['Impact']}")

def show_terminology():
    """
    Display comprehensive statistical terminology with definitions, formulas, and examples.

    This section includes both descriptive and inferential statistics terms
    with plot explanations and advanced statistical concepts.
    """
    st.markdown('<h1 class="main-header"> Statistical Terminology</h1>', unsafe_allow_html=True)

    # Back button for navigation
    if st.button(" Back to Home", key="back_terminology"):
        st.session_state.current_section = 'home'
        st.rerun()

    st.markdown("""
    ## Complete Statistical Reference Guide

    Master statistical concepts with definitions, formulas, and simple explanations.
    """)

    # Plot explanations section with comprehensive coverage
    st.markdown("##  Plot Types and Their Meanings")

    plot_explanations = {
        "Histogram": {
            "meaning": "Shows the frequency distribution of a continuous variable by dividing data into bins",
            "usage": "Best for understanding the shape, center, and spread of a single continuous variable",
            "understanding": "Each bar represents how many data points fall within a specific range of values",
            "example": "Visualizing test scores to see if most students scored high, low, or in the middle"
        },
        "Box Plot": {
            "meaning": "Displays the five-number summary: minimum, Q1, median, Q3, maximum, plus outliers",
            "usage": "Perfect for comparing distributions between groups and identifying outliers",
            "understanding": "The box shows the middle 50% of data, whiskers show the range, dots show outliers",
            "example": "Comparing salary ranges across different departments in a company"
        },
        "Violin Plot": {
            "meaning": "Combines box plot with kernel density estimation to show distribution shape",
            "usage": "Shows both summary statistics and the full distribution shape, especially useful for multimodal data",
            "understanding": "Width at any point shows how many data points exist at that value",
            "example": "Analyzing customer ages where there might be two peaks (young adults and middle-aged)"
        },
        "Density Plot": {
            "meaning": "Shows the probability density function of continuous data using kernel density estimation",
            "usage": "Reveals the underlying distribution shape without the arbitrary binning of histograms",
            "understanding": "The curve shows where data points are most likely to occur",
            "example": "Understanding the distribution of house prices in a city"
        },
        "Q-Q Plot": {
            "meaning": "Quantile-Quantile plot compares sample quantiles with theoretical distribution quantiles",
            "usage": "Tests if data follows a specific distribution (usually normal distribution)",
            "understanding": "Points on diagonal line = data matches expected distribution; deviations show how data differs",
            "example": "Checking if student test scores follow a normal distribution for statistical analysis"
        },
        "ECDF Plot": {
            "meaning": "Empirical Cumulative Distribution Function shows proportion of data below each value",
            "usage": "Shows percentiles and cumulative probabilities without making distribution assumptions",
            "understanding": "Y-axis shows what percentage of data falls below each X-axis value",
            "example": "Understanding what percentage of customers spend less than a certain amount"
        },
        "Strip Plot": {
            "meaning": "Shows individual data points for categorical variables with some jittering to avoid overlap",
            "usage": "Displays all data points while showing the distribution across categories",
            "understanding": "Each dot is an actual data point, jittered horizontally to reduce overlap",
            "example": "Showing individual student scores for each teacher to see both distribution and actual values"
        },
        "Rug Plot": {
            "meaning": "Adds small vertical lines along the axis to show the location of individual data points",
            "usage": "Often combined with other plots to show exact data point locations",
            "understanding": "Each small line represents one data point's exact location",
            "example": "Adding to a histogram to show exactly where each measurement occurred"
        },
        "Heat Map": {
            "meaning": "Uses colors to represent values in a matrix format, with color intensity showing magnitude",
            "usage": "Perfect for showing correlations between variables or patterns in two-dimensional data",
            "understanding": "Darker/brighter colors typically represent higher values, lighter colors represent lower values",
            "example": "Showing correlations between different stock prices or website activity by hour and day"
        },
        "Contingency Table": {
            "meaning": "Cross-tabulation showing frequencies of combinations between two categorical variables",
            "usage": "Analyzes relationships between categorical variables and tests for independence",
            "understanding": "Each cell shows how many observations have that specific combination of categories",
            "example": "Analyzing relationship between gender and product preference in customer surveys"
        }
    }

    # Display plot explanations in expandable format
    for plot_name, info in plot_explanations.items():
        with st.expander(f" {plot_name}"):
            st.markdown(f"""
            **Meaning:** {info['meaning']}

            **Usage:** {info['usage']}

            **Understanding:** {info['understanding']}

            **Example:** {info['example']}
            """)

    # Comprehensive statistical terms organized by category
    terminology_categories = {
        " Descriptive Statistics": {
            "Mean (Average)": {
                "formula": "μ = Σx / n",
                "definition": "The sum of all values divided by the number of values. Most common measure of central tendency.",
                "simple": "Add all numbers and divide by how many numbers you have. Like finding the average test score.",
                "example": "Test scores: 85, 90, 88, 92, 85. Mean = (85+90+88+92+85)/5 = 88"
            },
            "Median": {
                "formula": "Middle value when data is ordered",
                "definition": "The middle value in a dataset when arranged in ascending order. Divides data into two equal halves.",
                "simple": "Line up all numbers from smallest to largest and pick the middle one.",
                "example": "Values: 10, 15, 20, 25, 30. Median = 20 (middle value)"
            },
            "Mode": {
                "formula": "Most frequently occurring value",
                "definition": "The value that appears most often in a dataset. A dataset can have no mode, one mode, or multiple modes.",
                "simple": "The number that shows up most often in your list.",
                "example": "Shoe sizes: 7, 8, 8, 9, 8, 10. Mode = 8 (appears 3 times)"
            },
            "Standard Deviation": {
                "formula": "σ = √(Σ(x - μ)² / N)",
                "definition": "Measures the average distance of data points from the mean. Shows how spread out the data is.",
                "simple": "Tells you how much your numbers typically differ from the average.",
                "example": "Low SD = numbers close to average. High SD = numbers spread out from average."
            },
            "Variance": {
                "formula": "σ² = Σ(x - μ)² / N",
                "definition": "The average of squared differences from the mean. Variance is the square of standard deviation.",
                "simple": "Like standard deviation but squared. Shows how much data varies from the average.",
                "example": "If SD = 5, then Variance = 25. Units are squared (e.g., dollars²)."
            },
            "Range": {
                "formula": "Range = Maximum - Minimum",
                "definition": "The difference between the largest and smallest values in a dataset.",
                "simple": "Subtract the smallest number from the biggest number.",
                "example": "Test scores: 65, 78, 85, 92, 98. Range = 98 - 65 = 33"
            },
            "Interquartile Range (IQR)": {
                "formula": "IQR = Q3 - Q1",
                "definition": "The range of the middle 50% of data. Difference between 75th and 25th percentiles.",
                "simple": "The spread of the middle half of your data, ignoring extreme values.",
                "example": "Q1 = 25, Q3 = 75. IQR = 75 - 25 = 50"
            },
            "Coefficient of Variation": {
                "formula": "CV = (σ / μ) × 100%",
                "definition": "Standard deviation expressed as a percentage of the mean. Allows comparison across different scales.",
                "simple": "Compares variability relative to the average. Useful for comparing different datasets.",
                "example": "Stock A: Mean=$50, SD=$5, CV=10%. Stock B: Mean=$100, SD=$8, CV=8%"
            }
        },
        " Distribution Properties": {
            "Skewness": {
                "formula": "Skewness = E[(X-μ)³] / σ³",
                "definition": "Measures the asymmetry of a distribution. Indicates which tail is longer.",
                "simple": "Shows if your data leans more to one side. Like a lopsided hill.",
                "example": "Positive skew = tail on right (income data). Negative skew = tail on left (test scores)"
            },
            "Kurtosis": {
                "formula": "Kurtosis = E[(X-μ)⁴] / σ⁴ - 3",
                "definition": "Measures the 'tailedness' of a distribution. Indicates presence of outliers.",
                "simple": "Shows if your data has heavy tails (more extreme values) or light tails.",
                "example": "High kurtosis = more outliers. Low kurtosis = fewer extreme values."
            },
            "Normal Distribution": {
                "formula": "f(x) = (1/σ√2π) × e^(-½((x-μ)/σ)²)",
                "definition": "A symmetric, bell-shaped distribution where mean = median = mode. 68-95-99.7 rule applies.",
                "simple": "The classic bell curve. Most values cluster around the middle.",
                "example": "Heights, test scores, measurement errors often follow normal distribution."
            },
            "Standard Normal Distribution": {
                "formula": "Z = (X - μ) / σ",
                "definition": "Normal distribution with mean = 0 and standard deviation = 1. Used for standardization.",
                "simple": "Converting any normal distribution to a standard scale for comparison.",
                "example": "Z-score of 1.5 means 1.5 standard deviations above average."
            }
        },
        " Percentiles & Quantiles": {
            "Percentile": {
                "formula": "Pk = value below which k% of data falls",
                "definition": "Values that divide data into 100 equal parts. The kth percentile has k% of data below it.",
                "simple": "Your position compared to everyone else, expressed as a percentage.",
                "example": "90th percentile = you scored better than 90% of people."
            },
            "Quartile": {
                "formula": "Q1=P25, Q2=P50, Q3=P75",
                "definition": "Values that divide data into four equal parts (25% each).",
                "simple": "Splits your data into four equal groups.",
                "example": "Q1: bottom 25%, Q2: median, Q3: top 25% boundary"
            },
            "Decile": {
                "formula": "Di = P(i×10)",
                "definition": "Values that divide data into ten equal parts (10% each).",
                "simple": "Splits your data into ten equal groups.",
                "example": "D1 = 10th percentile, D5 = 50th percentile"
            },
            "Quintile": {
                "formula": "Qi = P(i×20)",
                "definition": "Values that divide data into five equal parts (20% each).",
                "simple": "Splits your data into five equal groups.",
                "example": "Used in investment risk categories, income brackets"
            }
        },
        " Correlation & Relationships": {
            "Correlation Coefficient": {
                "formula": "r = Σ[(xi-x̄)(yi-ȳ)] / √[Σ(xi-x̄)²Σ(yi-ȳ)²]",
                "definition": "Measures the strength and direction of linear relationship between two variables (-1 to +1).",
                "simple": "Shows how two things move together. +1 = perfect positive, -1 = perfect negative, 0 = no relationship.",
                "example": "Height vs Weight: r = 0.7 (strong positive). Study time vs TV time: r = -0.5 (moderate negative)"
            },
            "Coefficient of Determination (R²)": {
                "formula": "R² = r²",
                "definition": "Proportion of variance in dependent variable explained by independent variable(s).",
                "simple": "Percentage of variation in Y that's explained by X.",
                "example": "R² = 0.64 means 64% of variation in test scores is explained by study hours."
            },
            "Covariance": {
                "formula": "Cov(X,Y) = E[(X-μx)(Y-μy)]",
                "definition": "Measures how two variables vary together. Indicates direction but not strength of relationship.",
                "simple": "Shows if two variables move in same direction (positive) or opposite (negative).",
                "example": "Positive covariance: both increase together. Negative: one increases as other decreases."
            }
        },
        " Sampling & Inference": {
            "Population": {
                "formula": "N = total population size",
                "definition": "The entire group of individuals or items that we want to study or draw conclusions about.",
                "simple": "Everyone or everything you're interested in studying.",
                "example": "All students in a university, all voters in a country, all products made by a factory."
            },
            "Sample": {
                "formula": "n = sample size",
                "definition": "A subset of the population selected for study. Should be representative of the population.",
                "simple": "A smaller group chosen to represent the whole population.",
                "example": "Surveying 1,000 voters to predict election results for millions of voters."
            },
            "Sample Size": {
                "formula": "n = (Z²σ²) / E²",
                "definition": "The number of observations in a sample. Larger samples generally provide more reliable results.",
                "simple": "How many people or things you include in your study.",
                "example": "Survey of 500 people vs 50 people - the 500 is more reliable."
            },
            "Confidence Interval": {
                "formula": "CI = x̄ ± (t × SE)",
                "definition": "A range of values that likely contains the true population parameter with specified confidence.",
                "simple": "A range where we're pretty sure the true answer lies.",
                "example": "95% confident the average height is between 5'6\" and 5'10\""
            },
            "Margin of Error": {
                "formula": "ME = Z × (σ / √n)",
                "definition": "The maximum expected difference between sample statistic and true population parameter.",
                "simple": "How far off our estimate might be from the true value.",
                "example": "Poll shows 52% support ±3% margin of error = true support is 49-55%"
            },
            "Standard Error": {
                "formula": "SE = σ / √n",
                "definition": "Standard deviation of the sampling distribution. Measures precision of sample statistic.",
                "simple": "Shows how much your sample average might vary from the true average.",
                "example": "Smaller standard error = more precise estimate"
            }
        },
        " Hypothesis Testing & Inferential Statistics": {
            "Null Hypothesis (H₀)": {
                "formula": "H₀: parameter = specified value",
                "definition": "A statement of no effect or no difference. Assumes status quo until proven otherwise.",
                "simple": "The 'nothing special is happening' assumption we try to disprove.",
                "example": "H₀: New drug has no effect (same as placebo)"
            },
            "Alternative Hypothesis (H₁)": {
                "formula": "H₁: parameter ≠ specified value",
                "definition": "A statement that contradicts the null hypothesis. What we want to prove.",
                "simple": "The 'something interesting is happening' claim we're testing.",
                "example": "H₁: New drug is more effective than placebo"
            },
            "P-value": {
                "formula": "P(observing data | H₀ is true)",
                "definition": "Probability of observing the data (or more extreme) if null hypothesis is true.",
                "simple": "How surprising your results would be if nothing special was happening.",
                "example": "p = 0.03 means 3% chance of these results if null hypothesis is true"
            },
            "Significance Level (α)": {
                "formula": "α = 0.05 (common choice)",
                "definition": "Threshold for rejecting null hypothesis. Probability of Type I error.",
                "simple": "How sure you need to be before saying you found something significant.",
                "example": "α = 0.05 means you accept 5% chance of being wrong"
            },
            "Type I Error": {
                "formula": "P(reject H₀ | H₀ is true) = α",
                "definition": "Rejecting a true null hypothesis. False positive.",
                "simple": "Thinking you found something when you really didn't.",
                "example": "Concluding drug works when it actually doesn't"
            },
            "Type II Error": {
                "formula": "P(accept H₀ | H₁ is true) = β",
                "definition": "Accepting a false null hypothesis. False negative.",
                "simple": "Missing a real effect when it actually exists.",
                "example": "Concluding drug doesn't work when it actually does"
            },
            "Statistical Power": {
                "formula": "Power = 1 - β",
                "definition": "Probability of correctly rejecting a false null hypothesis.",
                "simple": "Your ability to detect a real effect when it exists.",
                "example": "80% power means 80% chance of finding effect if it's really there"
            },
            "Effect Size": {
                "formula": "d = (μ₁ - μ₂) / σ",
                "definition": "Magnitude of difference between groups, independent of sample size.",
                "simple": "How big is the difference you found, regardless of sample size.",
                "example": "Small effect: d=0.2, Medium: d=0.5, Large: d=0.8"
            },
            "Confidence Level": {
                "formula": "1 - α (e.g., 95% = 1 - 0.05)",
                "definition": "Probability that confidence interval contains true population parameter.",
                "simple": "How confident you are that your range contains the true answer.",
                "example": "95% confidence level means 95% of similar intervals would contain true value"
            }
        }
    }

    # Display terminology sections in expandable format
    for category, terms in terminology_categories.items():
        with st.expander(f"{category} ({len(terms)} terms)"):
            for term, info in terms.items():
                st.markdown(f"###  {term}")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    **Formula:** {info['formula']}

                    **Technical Definition:** {info['definition']}
                    """)

                with col2:
                    st.markdown(f"""
                    **Simple Explanation:** {info['simple']}

                    **Example:** {info['example']}
                    """)

                st.markdown("---")

    # Quick reference search functionality
    st.markdown("##  Quick Term Search")

    search_term = st.text_input("Search for a statistical term:", placeholder="e.g., mean, correlation, p-value")

    if search_term:
        found_terms = []
        search_lower = search_term.lower()

        # Search through all categories and terms
        for category, terms in terminology_categories.items():
            for term, info in terms.items():
                if (search_lower in term.lower() or
                    search_lower in info['definition'].lower() or
                    search_lower in info['simple'].lower()):
                    found_terms.append((category, term, info))

        # Display search results
        if found_terms:
            st.markdown(f"### Found {len(found_terms)} result(s):")
            for category, term, info in found_terms:
                st.markdown(f"""
                **{term}** ({category})
                - **Definition:** {info['definition']}
                - **Simple:** {info['simple']}
                - **Formula:** {info['formula']}
                """)
        else:
            st.write("No terms found. Try different keywords.")

    # Statistical symbols reference for comprehensive coverage
    st.markdown("##  Common Statistical Symbols")

    symbols = {
        "Greek Letters": {
            "μ (mu)": "Population mean",
            "σ (sigma)": "Population standard deviation",
            "σ² (sigma squared)": "Population variance",
            "α (alpha)": "Significance level, Type I error rate",
            "β (beta)": "Type II error rate",
            "ρ (rho)": "Population correlation coefficient",
            "χ² (chi-squared)": "Chi-square statistic"
        },
        "Latin Letters": {
            "x̄ (x-bar)": "Sample mean",
            "s": "Sample standard deviation",
            "s²": "Sample variance",
            "n": "Sample size",
            "N": "Population size",
            "r": "Sample correlation coefficient",
            "p": "Probability, proportion"
        },
        "Other Symbols": {
            "H₀": "Null hypothesis",
            "H₁ or Hₐ": "Alternative hypothesis",
            "≠": "Not equal to",
            "≤": "Less than or equal to",
            "≥": "Greater than or equal to",
            "Σ": "Sum of",
            "√": "Square root",
            "∞": "Infinity"
        }
    }

    # Display symbols in organized columns
    symbol_cols = st.columns(3)

    for i, (category, symbol_dict) in enumerate(symbols.items()):
        with symbol_cols[i]:
            st.markdown(f"**{category}**")
            for symbol, meaning in symbol_dict.items():
                st.write(f"**{symbol}:** {meaning}")

def show_dispersion():
    """
    Display comprehensive dispersion measures analysis with real-world applications.

    This section covers standard deviation, variance, range, IQR, and coefficient
    of variation with business context and visualizations.
    """
    st.markdown('<h1 class="main-header">� Measures of Dispersion</h1>', unsafe_allow_html=True)

    # Back button for navigation
    if st.button(" Back to Home", key="back_dispersion"):
        st.session_state.current_section = 'home'
        st.rerun()

    # Load datasets for dispersion analysis
    datasets = load_sample_datasets()

    # Dataset and column selection interface
    dataset_choice = st.selectbox("Choose a dataset:", list(datasets.keys()))
    selected_data = datasets[dataset_choice]
    numeric_cols = selected_data.select_dtypes(include=[np.number]).columns.tolist()
    selected_columns = st.multiselect("Select columns to compare:", numeric_cols, default=numeric_cols[:2])

    if selected_columns:
        st.subheader(" Dispersion Comparison")

        # Calculate dispersion measures for selected columns
        dispersion_data = []
        for col in selected_columns:
            data = selected_data[col].dropna()
            stats_dict = calculate_statistics(data)
            if stats_dict:
                dispersion_data.append({
                    'Column': col,
                    'Standard Deviation': stats_dict['std'],
                    'Variance': stats_dict['var'],
                    'Range': stats_dict['range'],
                    'IQR': stats_dict['iqr'],
                    'Coefficient of Variation (%)': stats_dict['cv']
                })

        if dispersion_data:
            # Display dispersion comparison table
            dispersion_df = pd.DataFrame(dispersion_data)
            st.dataframe(dispersion_df.round(3))

            # Enhanced visualization with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            for i, col in enumerate(selected_columns[:4]):
                if i < len(selected_columns):
                    row, column = divmod(i, 2)
                    data = selected_data[col].dropna()

                    # Box plot with statistical overlays
                    box_plot = axes[row, column].boxplot(data, patch_artist=True)
                    box_plot['boxes'][0].set_facecolor('lightblue')
                    axes[row, column].set_title(f'{col}')
                    axes[row, column].set_ylabel('Values')
                    axes[row, column].grid(True, alpha=0.3)

                    # Add mean line for reference
                    mean_val = np.mean(data)
                    axes[row, column].axhline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                    axes[row, column].legend()

            plt.tight_layout()
            st.pyplot(fig)

    # Industry applications with comprehensive explanations
    st.markdown("##  Industry Applications")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ** Manufacturing Quality Control**
        - **Standard Deviation:** Monitor consistency in production
        - **Six Sigma:** Requires products within ±3 standard deviations
        - **Range:** Set acceptable quality limits
        - **Impact:** Prevents defects, saves millions in recalls
        """)

    with col2:
        st.markdown("""
        ** Investment Risk Management**
        - **Standard Deviation:** Measure portfolio volatility
        - **Variance:** Calculate risk-adjusted returns
        - **CV:** Compare assets with different price ranges
        - **Impact:** Optimize risk-return trade-offs for billions in assets
        """)

    with col3:
        st.markdown("""
        ** Supply Chain Management**
        - **Range:** Plan for delivery time variations
        - **IQR:** Set realistic service level agreements
        - **Standard Deviation:** Buffer inventory planning
        - **Impact:** Reduce costs and improve customer satisfaction
        """)

def show_descriptive_stats():
    """
    Display comprehensive descriptive statistics overview with interactive analysis.

    This section provides a general overview of descriptive statistics with
    dataset exploration and basic statistical calculations.
    """
    st.markdown('<h1 class="main-header"> Descriptive Statistics Overview</h1>', unsafe_allow_html=True)

    # Back button for navigation
    if st.button(" Back to Home", key="back_descriptive"):
        st.session_state.current_section = 'home'
        st.rerun()

    # Load available datasets
    datasets = load_sample_datasets()

    # Dataset selection interface
    dataset_choice = st.selectbox("Choose a dataset:", list(datasets.keys()))
    selected_data = datasets[dataset_choice]

    # Display basic dataset information
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Shape:** {selected_data.shape}")
        st.write(f"**Columns:** {list(selected_data.columns)}")

    with col2:
        st.subheader("Sample Data")
        st.dataframe(selected_data.head())

    # Numeric column analysis
    numeric_cols = selected_data.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        selected_column = st.selectbox("Select a numeric column for analysis:", numeric_cols)

        if selected_column:
            data = selected_data[selected_column].dropna()
            if len(data) > 0:
                stats_dict = calculate_statistics(data)

                if stats_dict:
                    # Display comprehensive statistics
                    st.subheader(" Descriptive Statistics")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{stats_dict['mean']:.2f}")
                        st.metric("Standard Deviation", f"{stats_dict['std']:.2f}")
                    with col2:
                        st.metric("Median", f"{stats_dict['median']:.2f}")
                        st.metric("Variance", f"{stats_dict['var']:.2f}")
                    with col3:
                        st.metric("Mode", f"{stats_dict['mode']:.2f}")
                        st.metric("Range", f"{stats_dict['range']:.2f}")
                    with col4:
                        st.metric("Q1", f"{stats_dict['q1']:.2f}")
                        st.metric("Q3", f"{stats_dict['q3']:.2f}")

                    # Comprehensive visualizations
                    st.subheader(" Visualizations")

                    # Create tabs for different visualization types
                    tab1, tab2, tab3, tab4 = st.tabs(["Histogram", "Box Plot", "Distribution", "Summary"])

                    with tab1:
                        # Interactive histogram with customizable bins
                        bins = st.slider("Number of bins:", 10, 50, 30)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
                        ax.set_title(f'Histogram of {selected_column}')
                        ax.set_xlabel(selected_column)
                        ax.set_ylabel('Frequency')
                        st.pyplot(fig)

                    with tab2:
                        # Box plot for outlier detection
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.boxplot(data)
                        ax.set_title(f'Box Plot of {selected_column}')
                        ax.set_ylabel(selected_column)
                        st.pyplot(fig)

                    with tab3:
                        # Distribution plot with kernel density estimation
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(data, kde=True, ax=ax)
                        ax.set_title(f'Distribution of {selected_column}')
                        st.pyplot(fig)

                    with tab4:
                        # Comprehensive statistical summary
                        st.write(selected_data.describe())

def show_feedback_form():
    """
    Display simplified feedback form with email contact information.

    This section provides a minimal feedback interface as requested,
    with direct email contact for suggestions and improvements.
    """
    st.markdown('<h1 class="main-header"> Feedback & Suggestions</h1>', unsafe_allow_html=True)

    # Back button for navigation
    if st.button(" Back to Home", key="back_feedback"):
        st.session_state.current_section = 'home'
        st.rerun()

    # Simple feedback message as requested
    st.markdown("""
    ## 📧 Contact Information

    For any feedback or suggestions for the next versions, please mail to: **ma24m012@smail.iitm.ac.in**
    """)

# Run the main application when script is executed
if __name__ == "__main__":
    main()













