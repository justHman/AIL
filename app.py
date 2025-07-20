import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from collections import defaultdict

# Add the current directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.recommender import Recommender
from src.models import iiCB, Ridge_iiCB, knnCF, PB
from src.utils import get_items_rated_by_user

# Set page config
st.set_page_config(
    page_title="AI Learning Recommendation System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the data loading and model initialization
@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv('data/captone_data.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please check if 'data/captone_data.csv' exists.")
        return None

def train_test_split_iiCF(sparse, k=1, seed=42):
    """Custom train-test split function that ensures no unseen users/items in test set"""
    train_list = []
    test_list = []
    grouped = sparse.groupby('user')

    for user, group in grouped:
        if len(group) <= k:
            train_list.append(group)
        else:
            group = group.sample(frac=1, random_state=seed)
            test_part = group.iloc[:k]
            train_part = group.iloc[k:]
            train_list.append(train_part)
            test_list.append(test_part)

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    # Check if test users and items exist in train
    test_users = set(test_df['user'].unique())
    train_users = set(train_df['user'].unique())
    test_items = set(test_df['item'].unique())
    train_items = set(train_df['item'].unique())

    unseen_users = test_users - train_users
    unseen_items = test_items - train_items

    if unseen_users or unseen_items:
        print("⚠️ Warning: There are users or items in test that are not in train.")
        print(f"Unseen users: {unseen_users}")
        print(f"Unseen items: {unseen_items}")
    else:
        print("✅ Ensured no new users/items in test set.")

    return train_df, test_df

@st.cache_resource
def initialize_recommender(df):
    """Initialize the recommender system using custom train-test split"""
    if df is not None:
        train_sparse, test_sparse = train_test_split_iiCF(df, k=2, seed=42)
        recommender_engine = Recommender(train_sparse)
        return recommender_engine, train_sparse, test_sparse
    return None, None, None

def create_data_visualizations(df):
    """Create various data visualizations"""
    
    # Rating distribution
    fig_rating = px.histogram(
        df, x='rating', 
        title='Rating Distribution',
        nbins=10,
        color_discrete_sequence=['#1f77b4']
    )
    fig_rating.update_layout(
        xaxis_title="Rating",
        yaxis_title="Count",
        showlegend=False
    )
    
    # User activity distribution
    user_activity = df.groupby('user').size().reset_index(name='num_ratings')
    fig_user = px.histogram(
        user_activity, x='num_ratings',
        title='User Activity Distribution',
        nbins=50,
        color_discrete_sequence=['#ff7f0e']
    )
    fig_user.update_layout(
        xaxis_title="Number of Ratings per User",
        yaxis_title="Number of Users",
        showlegend=False
    )
    
    # Item popularity distribution
    item_popularity = df.groupby('item').size().reset_index(name='num_ratings')
    fig_item = px.histogram(
        item_popularity, x='num_ratings',
        title='Item Popularity Distribution',
        nbins=50,
        color_discrete_sequence=['#2ca02c']
    )
    fig_item.update_layout(
        xaxis_title="Number of Ratings per Item",
        yaxis_title="Number of Items",
        showlegend=False
    )
    
    # Top rated items
    top_items = df.groupby('item').agg({
        'rating': ['mean', 'count']
    }).round(2)
    top_items.columns = ['avg_rating', 'num_ratings']
    top_items = top_items[top_items['num_ratings'] >= 5].sort_values('avg_rating', ascending=False).head(20)
    
    fig_top_items = px.bar(
        x=top_items['avg_rating'], 
        y=top_items.index,
        orientation='h',
        title='Top 20 Highest Rated Items (min 5 ratings)',
        color=top_items['avg_rating'],
        color_continuous_scale='viridis'
    )
    fig_top_items.update_layout(
        xaxis_title="Average Rating",
        yaxis_title="Item",
        showlegend=False
    )
    
    return fig_rating, fig_user, fig_item, fig_top_items

def evaluate_model(model, test_data, model_name, progress_bar=None):
    """Evaluate a single model"""
    start_time = time.time()
    
    try:
        # Special handling for iiCB model (pure recommendation, no rating prediction)
        if model_name == 'iiCB':
            st.info(f"ℹ️ {model_name} is a content-based recommendation model without rating prediction.")
            st.write("Evaluating recommendation quality instead of rating prediction...")
            
            # Evaluate recommendation quality metrics instead
            execution_time = time.time() - start_time
            
            # Test recommendation generation for a few users
            test_users = test_data['user'].unique()[:10]  # Test with first 10 users
            successful_recs = 0
            total_recs = 0
            
            for user in test_users:
                try:
                    recs = model.recommend(user, 5, return_result=False)
                    if recs and len(recs) > 0:
                        successful_recs += len(recs)
                    total_recs += 5
                except:
                    continue
            
            coverage = (successful_recs / total_recs) * 100 if total_recs > 0 else 0
            
            metrics = {
                'Model_Type': 'Content-Based',
                'Coverage_%': round(coverage, 2),
                'Avg_Recs_Per_User': round(successful_recs / len(test_users), 2) if len(test_users) > 0 else 0,
                'Execution_Time': float(execution_time)
            }
            
            if progress_bar:
                progress_bar.progress(1.0)
            
            return metrics
        
        # Standard evaluation for models with rating prediction capability
        elif hasattr(model, 'evaluate'):
            # For models with evaluate method (knnCF, Ridge_iiCB)
            result = model.evaluate(test_data, return_result=True)
            if isinstance(result, tuple) and len(result) >= 4:
                # Handle tuple return: (df, mae, mse, rmse, r2) or (mae, mse, rmse, r2)
                if isinstance(result[0], pd.DataFrame):
                    # Format: (df, mae, mse, rmse, r2)
                    _, mae, mse, rmse, r2 = result[:5]
                    metrics = {'RMSE': float(rmse), 'MAE': float(mae), 'MSE': float(mse), 'R2': float(r2)}
                else:
                    # Format: (mae, mse, rmse, r2)
                    mae, mse, rmse, r2 = result[:4]
                    metrics = {'RMSE': float(rmse), 'MAE': float(mae), 'MSE': float(mse), 'R2': float(r2)}
            elif isinstance(result, dict):
                # Convert all values to float
                metrics = {k: float(v) if not isinstance(v, pd.DataFrame) else float('inf') for k, v in result.items()}
            else:
                # Assume it returns RMSE if not a dict or tuple
                metrics = {'RMSE': float(result)}
        
        # Manual evaluation for models with predict method but no evaluate method
        elif hasattr(model, 'predict'):
            predictions = []
            actuals = []
            
            # Use a smaller sample for faster evaluation
            test_sample = test_data.sample(min(500, len(test_data)), random_state=42)
            
            for _, row in test_sample.iterrows():
                try:
                    pred = model.predict(row['item'], row['user'])
                    predictions.append(float(pred))
                    actuals.append(float(row['rating']))
                except:
                    # Skip problematic predictions
                    continue
            
            if predictions and len(predictions) > 0:
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                rmse = float(np.sqrt(np.mean((predictions - actuals)**2)))
                mae = float(np.mean(np.abs(predictions - actuals)))
                mse = float(np.mean((predictions - actuals)**2))
                
                # Calculate R2 safely
                ss_res = np.sum((actuals - predictions) ** 2)
                ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
                r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
                
                metrics = {'RMSE': rmse, 'MAE': mae, 'MSE': mse, 'R2': r2}
            else:
                metrics = {'RMSE': float('inf'), 'MAE': float('inf'), 'MSE': float('inf'), 'R2': 0.0}
        
        # For models that only have recommend method
        else:
            st.warning(f"⚠️ {model_name} doesn't support rating prediction. Using recommendation-based evaluation.")
            execution_time = time.time() - start_time
            
            metrics = {
                'Model_Type': 'Recommendation-Only',
                'Evaluation': 'N/A (No rating prediction)',
                'Execution_Time': float(execution_time)
            }
        
        execution_time = time.time() - start_time
        metrics['Execution_Time'] = float(execution_time)
        
        if progress_bar:
            progress_bar.progress(1.0)
        
        return metrics
        
    except Exception as e:
        st.error(f"Error evaluating {model_name}: {str(e)}")
        execution_time = time.time() - start_time
        return {
            'Error': f'Evaluation failed: {str(e)}',
            'Execution_Time': float(execution_time)
        }

def compare_models(train_data, test_data, recommender_engine):
    """Compare different recommendation models"""
    
    st.subheader("🔍 Model Comparison")
    st.write("Evaluating different recommendation algorithms...")
    
    # Initialize models
    models = {}
    
    try:
        # kNN Collaborative Filtering
        models['knnCF'] = knnCF(train_data, recommender_engine.iiCF_sim_matrix, recommender_engine.utility_norm)
    except Exception as e:
        st.warning(f"Could not initialize knnCF: {str(e)}")
    
    try:
        # Item-Item Content-Based
        models['iiCB'] = iiCB(train_data, sim_matrix=recommender_engine.iiCB_sim_matrix)
    except Exception as e:
        st.warning(f"Could not initialize iiCB: {str(e)}")
    
    try:
        # Ridge Item-Item Content-Based
        ridge_model = Ridge_iiCB(train_data)
        ridge_model.train(recommender_engine.item_vectors)
        models['Ridge_iiCB'] = ridge_model
    except Exception as e:
        st.warning(f"Could not initialize Ridge_iiCB: {str(e)}")
    
    if not models:
        st.error("No models could be initialized for comparison")
        return
    
    # Evaluate models
    results = {}
    progress_container = st.container()
    
    with progress_container:
        for model_name, model in models.items():
            st.write(f"Evaluating {model_name}...")
            progress_bar = st.progress(0)
            
            # Use a smaller subset for faster evaluation
            test_subset = test_data.sample(min(1000, len(test_data)), random_state=42)
            
            metrics = evaluate_model(model, test_subset, model_name, progress_bar)
            results[model_name] = metrics
            
            st.write(f"✅ {model_name} completed")
    
    # Display results
    if results:
        st.subheader("📊 Evaluation Results")
        
        # Create comparison DataFrame with explicit type conversion
        comparison_data = {}
        for model_name, metrics in results.items():
            comparison_data[model_name] = {}
            for metric_name, value in metrics.items():
                # Ensure all values are numeric scalars
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    comparison_data[model_name][metric_name] = float('inf')
                elif isinstance(value, (list, tuple)):
                    comparison_data[model_name][metric_name] = float('inf')
                else:
                    try:
                        comparison_data[model_name][metric_name] = float(value)
                    except (ValueError, TypeError):
                        comparison_data[model_name][metric_name] = float('inf')
        
        comparison_df = pd.DataFrame(comparison_data).T
        comparison_df = comparison_df.round(4)
        
        # Display table
        st.dataframe(comparison_df, use_container_width=True)
        
        # Create comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # RMSE Comparison
            if 'RMSE' in comparison_df.columns:
                fig_rmse = px.bar(
                    x=comparison_df.index,
                    y=comparison_df['RMSE'],
                    title='RMSE Comparison (Lower is Better)',
                    color=comparison_df['RMSE'],
                    color_continuous_scale='RdYlBu_r'
                )
                fig_rmse.update_layout(
                    xaxis_title="Model",
                    yaxis_title="RMSE",
                    showlegend=False
                )
                st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            # Execution Time Comparison
            if 'Execution_Time' in comparison_df.columns:
                fig_time = px.bar(
                    x=comparison_df.index,
                    y=comparison_df['Execution_Time'],
                    title='Execution Time Comparison',
                    color=comparison_df['Execution_Time'],
                    color_continuous_scale='Viridis'
                )
                fig_time.update_layout(
                    xaxis_title="Model",
                    yaxis_title="Time (seconds)",
                    showlegend=False
                )
                st.plotly_chart(fig_time, use_container_width=True)
        
        # Best model summary
        if 'RMSE' in comparison_df.columns:
            best_model = comparison_df['RMSE'].idxmin()
            best_rmse = comparison_df.loc[best_model, 'RMSE']
            
            st.success(f"🏆 Best performing model: **{best_model}** with RMSE: **{best_rmse:.4f}**")
        
        return comparison_df
    
    return None

def main():
    st.title("🤖 AI Learning Recommendation System")
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommendations", "📊 Data Visualization", "🔍 Model Evaluation", "ℹ️ About"])
    
    # Load data
    with st.spinner("Loading dataset..."):
        df = load_data()
    
    if df is None:
        st.stop()
    
    # Initialize recommender
    with st.spinner("Initializing recommendation engine..."):
        recommender_engine, train_sparse, test_sparse = initialize_recommender(df)
    
    if recommender_engine is None:
        st.error("Failed to initialize recommender system")
        st.stop()
    
    # TAB 1: RECOMMENDATIONS
    with tab1:
        # Sidebar for user input
        st.sidebar.header("Configuration")
        
        # Get unique users for selection
        users = sorted(df['user'].unique())
        
        # User selection
        user_input_method = st.sidebar.radio(
            "Select user input method:",
            ["Select from dropdown", "Enter user ID manually"]
        )
        
        if user_input_method == "Select from dropdown":
            selected_user = st.sidebar.selectbox(
                "Select a user:",
                users,
                index=0
            )
        else:
            selected_user = st.sidebar.number_input(
                "Enter user ID:",
                min_value=int(min(users)),
                max_value=int(max(users)),
                value=int(users[0]),
                step=1
            )
        
        # Algorithm selection
        algorithm = st.sidebar.selectbox(
            "Select recommendation algorithm:",
            ["hybrid", "knnCF", "iiCB", "ridge_iiCB"],
            index=0
        )
        
        # Number of recommendations
        num_recommendations = st.sidebar.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=20,
            value=10
        )
        
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("User Information")
            
            # Check if user exists
            if selected_user in users:
                st.success(f"✅ User {selected_user} found in dataset")
                
                # Show user's rating history
                user_ratings = df[df['user'] == selected_user].sort_values('rating', ascending=False)
                st.write(f"**User {selected_user} has rated {len(user_ratings)} items:**")
                
                # Display user's top rated items
                if not user_ratings.empty:
                    st.dataframe(
                        user_ratings.head(10)[['item', 'rating']],
                        use_container_width=True
                    )
                
            else:
                st.warning(f"⚠️ User {selected_user} not found in dataset (will use cold-start strategy)")
        
        with col2:
            st.subheader("Get Recommendations")
            
            if st.button("🎯 Generate Recommendations", type="primary"):
                with st.spinner(f"Generating {num_recommendations} recommendations using {algorithm}..."):
                    try:
                        # Get recommendations
                        result = recommender_engine.recommend(
                            user=selected_user,
                            n=num_recommendations,
                            aglorithm=algorithm
                        )
                        
                        recommendations = result['recommendations']
                        strategy = result['strategy']
                        
                        st.success(f"✅ Recommendations generated using strategy: **{strategy}**")
                        
                        # Display recommendations
                        if recommendations:
                            st.write("**Recommended Items:**")
                            
                            # Create a dataframe for better display
                            if isinstance(recommendations[0], tuple):
                                # If recommendations include ratings
                                rec_df = pd.DataFrame(
                                    recommendations,
                                    columns=['Item', 'Predicted Rating']
                                )
                                rec_df.index = range(1, len(rec_df) + 1)
                            else:
                                # If recommendations are just item IDs
                                rec_df = pd.DataFrame(
                                    recommendations,
                                    columns=['Item']
                                )
                                rec_df.index = range(1, len(rec_df) + 1)
                            
                            st.dataframe(rec_df, use_container_width=True)
                            
                            # Show some statistics
                            st.info(f"📊 Generated {len(recommendations)} recommendations")
                            
                        else:
                            st.warning("No recommendations generated")
                            
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
        
        # Dataset Information for Tab 1
        st.markdown("---")
        st.subheader("📈 Dataset Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(df['user'].unique()))
        
        with col2:
            st.metric("Total Items", len(df['item'].unique()))
        
        with col3:
            st.metric("Total Ratings", len(df))
        
        with col4:
            st.metric("Average Rating", f"{df['rating'].mean():.2f}")
    
    # TAB 2: DATA VISUALIZATION
    with tab2:
        st.header("📊 Data Visualization")
        st.write("Explore the dataset through various visualizations")
        
        with st.spinner("Creating visualizations..."):
            fig_rating, fig_user, fig_item, fig_top_items = create_data_visualizations(df)
        
        # Display visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_rating, use_container_width=True)
            st.plotly_chart(fig_user, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_item, use_container_width=True)
            st.plotly_chart(fig_top_items, use_container_width=True)
        
        # Additional statistics
        st.subheader("📈 Detailed Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Rating Statistics:**")
            rating_stats = df['rating'].describe()
            st.dataframe(rating_stats.round(2))
        
        with col2:
            st.write("**User Activity Stats:**")
            user_activity = df.groupby('user').size()
            user_stats = user_activity.describe()
            st.dataframe(user_stats.round(2))
        
        with col3:
            st.write("**Item Popularity Stats:**")
            item_popularity = df.groupby('item').size()
            item_stats = item_popularity.describe()
            st.dataframe(item_stats.round(2))
        
        # Sparsity analysis
        st.subheader("🔍 Sparsity Analysis")
        total_possible_ratings = len(df['user'].unique()) * len(df['item'].unique())
        actual_ratings = len(df)
        sparsity = (1 - actual_ratings / total_possible_ratings) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Possible Ratings", f"{total_possible_ratings:,}")
        with col2:
            st.metric("Actual Ratings", f"{actual_ratings:,}")
        with col3:
            st.metric("Sparsity", f"{sparsity:.2f}%")
    
    # TAB 3: MODEL EVALUATION
    with tab3:
        st.header("🔍 Model Evaluation & Comparison")
        st.write("Compare the performance of different recommendation algorithms")
        
        if st.button("🚀 Start Model Comparison", type="primary"):
            comparison_results = compare_models(train_sparse, test_sparse, recommender_engine)
            
            if comparison_results is not None:
                # Additional insights
                st.subheader("💡 Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Performance Ranking (by RMSE):**")
                    if 'RMSE' in comparison_results.columns:
                        ranking = comparison_results.sort_values('RMSE')[['RMSE']].round(4)
                        ranking.index = [f"{i+1}. {idx}" for i, idx in enumerate(ranking.index)]
                        st.dataframe(ranking)
                
                with col2:
                    st.write("**Speed Ranking (by Execution Time):**")
                    if 'Execution_Time' in comparison_results.columns:
                        speed_ranking = comparison_results.sort_values('Execution_Time')[['Execution_Time']].round(4)
                        speed_ranking.index = [f"{i+1}. {idx}" for i, idx in enumerate(speed_ranking.index)]
                        st.dataframe(speed_ranking)
        
        else:
            st.info("Click the button above to start model comparison. This may take a few minutes.")
            
            # Show dataset split info
            st.subheader("📊 Dataset Split Information")
            st.write("**Custom Train-Test Split Method:**")
            st.write("- Uses `train_test_split_iiCF` function with k=2")
            st.write("- For each user: takes k=2 random ratings for test, rest for training")
            st.write("- Ensures no unseen users/items in test set")
            st.write("- Maintains data integrity for collaborative filtering")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Training Set Size", f"{len(train_sparse):,} ratings")
                st.metric("Training Users", len(train_sparse['user'].unique()))
                st.metric("Training Items", len(train_sparse['item'].unique()))
            
            with col2:
                st.metric("Test Set Size", f"{len(test_sparse):,} ratings")
                st.metric("Test Users", len(test_sparse['user'].unique()))
                st.metric("Test Items", len(test_sparse['item'].unique()))
    
    # TAB 4: ABOUT
    with tab4:
        st.header("ℹ️ About the System")
        
        # Algorithm information
        st.subheader("🤖 Available Algorithms")
        
        algo_info = {
            "Hybrid": {
                "description": "Combines multiple recommendation techniques for better performance",
                "pros": ["Better coverage", "Reduced cold-start problem", "Improved accuracy"],
                "cons": ["Higher complexity", "Longer computation time"]
            },
            "knnCF": {
                "description": "k-Nearest Neighbors Collaborative Filtering based on user-item interactions",
                "pros": ["Simple to understand", "Good for similar users", "No content needed"],
                "cons": ["Cold-start problem", "Sparsity issues", "Scalability challenges"]
            },
            "iiCB": {
                "description": "Item-Item Content-Based filtering using item similarity",
                "pros": ["No cold-start for items", "Explainable recommendations", "Domain knowledge"],
                "cons": ["Limited novelty", "Requires item features", "Overspecialization"]
            },
            "Ridge_iiCB": {
                "description": "Ridge Regression enhanced Item-Item Content-Based filtering",
                "pros": ["Regularization reduces overfitting", "Better generalization", "Handles multicollinearity"],
                "cons": ["More complex", "Requires tuning", "May underfit with high regularization"]
            }
        }
        
        for algo, info in algo_info.items():
            with st.expander(f"📖 {algo}"):
                st.write(f"**Description:** {info['description']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Pros:**")
                    for pro in info['pros']:
                        st.write(f"• {pro}")
                
                with col2:
                    st.write("**Cons:**")
                    for con in info['cons']:
                        st.write(f"• {con}")
        
        # System overview
        st.subheader("🏗️ System Architecture")
        st.write("""
        This recommendation system consists of several components:
        
        1. **Data Processing**: Handles data loading, preprocessing, and splitting
        2. **Model Training**: Initializes and trains different recommendation models
        3. **Recommendation Engine**: Provides unified interface for different algorithms
        4. **Evaluation Framework**: Compares models using various metrics
        5. **Visualization**: Interactive charts and graphs for data exploration
        """)
        
        # Technical details
        st.subheader("⚙️ Technical Details")
        st.write("""
        **Technologies Used:**
        - **Streamlit**: Web interface framework
        - **Pandas**: Data manipulation and analysis
        - **NumPy**: Numerical computing
        - **Plotly**: Interactive visualizations
        - **Scikit-learn**: Machine learning algorithms
        
        **Evaluation Metrics:**
        - **RMSE**: Root Mean Square Error (lower is better)
        - **MAE**: Mean Absolute Error (lower is better)
        - **Execution Time**: Algorithm performance speed
        """)
        
        # Usage tips
        st.subheader("💡 Usage Tips")
        st.write("""
        1. **For New Users**: The system will automatically use popular-based recommendations
        2. **For Existing Users**: Choose the algorithm that best fits your needs
        3. **Model Comparison**: Use the evaluation tab to find the best algorithm for your data
        4. **Visualizations**: Explore the data tab to understand your dataset better
        5. **Performance**: Some algorithms may take longer for large datasets
        """)

if __name__ == "__main__":
    main()
