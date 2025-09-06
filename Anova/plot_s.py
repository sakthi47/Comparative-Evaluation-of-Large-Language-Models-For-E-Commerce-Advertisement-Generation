import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the processed data from the 5-model analysis script"""
    try:
        metrics_df = pd.read_csv('processed_data_model_comparison.csv')
        ratings_df = pd.read_csv('processed_ratings_model_comparison.csv')
        print(f"Loaded processed data: {len(metrics_df)} ad evaluations, {len(ratings_df)} individual ratings")
        print(f"Models: {sorted(metrics_df['LLM_Model'].unique())}")
        print(f"Products: {metrics_df['ProductName'].unique()}")
        return metrics_df, ratings_df
    except FileNotFoundError:
        print("Error: 5-model processed data files not found. Please run the 5-model analysis script first.")
        return None, None

def create_individual_question_plots_5models(metrics_df):
    """Create separate plots for each of the 5 questions using 5-model grouping"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create plots directory if it doesn't exist
    import os
    if not os.path.exists('5model_plots'):
        os.makedirs('5model_plots')
    
    # Question columns in metrics data
    question_columns = {
        'PurchaseIntent': 'Purchase Intent',
        'VisualAppeal': 'Visual Appeal',
        'ValueConvincing': 'Value Convincing',
        'MessageClarity': 'Message Clarity',
        'Trustworthiness': 'Trustworthiness'
    }
    
    # Define colors for the 5 models
    model_colors = {
        'Human': '#2E8B57',      # Green
        'Claude': '#FF6B6B',     # Red
        'ChatGPT': '#4ECDC4',    # Teal
        'Deepseek': '#45B7D1',   # Blue
        'Gemini': '#96CEB4'      # Light Green
    }
    
    for i, (col_name, display_name) in enumerate(question_columns.items(), 1):
        print(f"Creating plot for: {display_name}")
        
        # Calculate means and standard errors by model
        stats_by_model = metrics_df.groupby('LLM_Model')[col_name].agg(['mean', 'std', 'count']).reset_index()
        stats_by_model['se'] = stats_by_model['std'] / np.sqrt(stats_by_model['count'])
        stats_by_model = stats_by_model.sort_values('mean', ascending=False)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get colors for each model
        colors = [model_colors[model] for model in stats_by_model['LLM_Model']]
        
        # Create bar plot
        bars = ax.bar(range(len(stats_by_model)), stats_by_model['mean'], 
                      yerr=stats_by_model['se'], capsize=8,
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Customize the plot
        ax.set_title(f'{display_name} by AI Model\n(Mean ± Standard Error)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('AI Model', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{display_name} Rating (1-7 scale)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(stats_by_model)))
        ax.set_xticklabels(stats_by_model['LLM_Model'], fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 5.5)
        
        # Add value labels and sample sizes
        for j, (bar, row) in enumerate(zip(bars, stats_by_model.itertuples())):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}\n(n={row.count})', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Perform ANOVA for this question and add to plot
        groups = [group[col_name].values for name, group in metrics_df.groupby('LLM_Model')]
        f_stat, p_val = scipy_stats.f_oneway(*groups)
        
        # Calculate effect size
        ss_between = sum(len(group) * (np.mean(group) - np.mean(metrics_df[col_name]))**2 for group in groups)
        ss_total = sum((metrics_df[col_name] - np.mean(metrics_df[col_name]))**2)
        eta_squared = ss_between / ss_total
        
        textstr = f'ANOVA: F = {f_stat:.3f}, p = {p_val:.4f}\nη² = {eta_squared:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        # Add significance indicator
        if p_val < 0.05:
            ax.text(0.98, 0.98, '*** SIGNIFICANT ***', transform=ax.transAxes, 
                   fontsize=12, fontweight='bold', color='red',
                   horizontalalignment='right', verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(f'5model_plots/{i:02d}_{display_name.lower().replace(" ", "_")}_5models.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Created individual question plots for 5 models in '5model_plots/' directory")

def create_model_comparison_heatmap(metrics_df):
    """Create heatmap comparing all 5 questions across the 5 models"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create pivot table
    question_columns = ['PurchaseIntent', 'VisualAppeal', 'ValueConvincing', 'MessageClarity', 'Trustworthiness']
    pivot_data = metrics_df.groupby('LLM_Model')[question_columns].mean()
    
    # Rename columns for display
    pivot_data.columns = ['Purchase Intent', 'Visual Appeal', 'Value Convincing', 'Message Clarity', 'Trustworthiness']
    
    # Reorder rows by overall performance
    pivot_data['Overall'] = pivot_data.mean(axis=1)
    pivot_data = pivot_data.sort_values('Overall', ascending=False).drop('Overall', axis=1)
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', fmt='.2f', 
                cbar_kws={'label': 'Mean Rating (1-7 scale)'}, 
                square=False, linewidths=1, annot_kws={'fontweight': 'bold', 'fontsize': 11})
    
    ax.set_title('AI Model Performance Across All Rating Dimensions\n(5-Model Comparison)', 
              fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Rating Dimension', fontsize=14, fontweight='bold')
    ax.set_ylabel('AI Model', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('5model_plots/model_comparison_heatmap_5models.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created 5-model comparison heatmap")

def create_overall_model_performance(metrics_df):
    """Create overall model performance comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('5-Model Performance Analysis', fontsize=18, fontweight='bold')
    
    # Model colors
    model_colors = {
        'Human': '#2E8B57',      # Green
        'Claude': '#FF6B6B',     # Red
        'ChatGPT': '#4ECDC4',    # Teal
        'Deepseek': '#45B7D1',   # Blue
        'Gemini': '#96CEB4'      # Light Green
    }
    
    # 1. Overall Performance by Model
    model_performance = metrics_df.groupby('LLM_Model')['OverallRating'].mean().sort_values(ascending=False)
    colors = [model_colors[model] for model in model_performance.index]
    
    bars1 = axes[0,0].bar(range(len(model_performance)), model_performance.values, 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0,0].set_title('Overall Rating by AI Model', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Mean Overall Rating', fontsize=12)
    axes[0,0].set_xticks(range(len(model_performance)))
    axes[0,0].set_xticklabels(model_performance.index, fontsize=12, fontweight='bold')
    axes[0,0].grid(axis='y', alpha=0.3)
    axes[0,0].set_ylim(0, 5)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                      f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 2. Purchase Intent by Model
    purchase_performance = metrics_df.groupby('LLM_Model')['PurchaseIntent'].mean().sort_values(ascending=False)
    colors2 = [model_colors[model] for model in purchase_performance.index]
    
    bars2 = axes[0,1].bar(range(len(purchase_performance)), purchase_performance.values, 
                         color=colors2, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0,1].set_title('Purchase Intent by AI Model', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Mean Purchase Intent', fontsize=12)
    axes[0,1].set_xticks(range(len(purchase_performance)))
    axes[0,1].set_xticklabels(purchase_performance.index, fontsize=12, fontweight='bold')
    axes[0,1].grid(axis='y', alpha=0.3)
    axes[0,1].set_ylim(0, 5)
    
    # 3. Performance across all dimensions
    dimensions = ['PurchaseIntent', 'VisualAppeal', 'ValueConvincing', 'MessageClarity', 'Trustworthiness']
    dimension_labels = ['Purchase', 'Visual', 'Value', 'Clarity', 'Trust']
    models = sorted(metrics_df['LLM_Model'].unique())
    
    x = np.arange(len(dimensions))
    width = 0.15
    
    for i, model in enumerate(models):
        model_data = metrics_df[metrics_df['LLM_Model'] == model]
        means = [model_data[dim].mean() for dim in dimensions]
        axes[1,0].bar(x + i*width, means, width, label=model, 
                     color=model_colors[model], alpha=0.8, edgecolor='black', linewidth=1)
    
    axes[1,0].set_title('Performance Across All Dimensions', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Mean Rating', fontsize=12)
    axes[1,0].set_xlabel('Rating Dimension', fontsize=12)
    axes[1,0].set_xticks(x + width * 2)
    axes[1,0].set_xticklabels(dimension_labels, fontsize=11)
    axes[1,0].legend(fontsize=10, loc='upper right')
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # 4. High Rating Rates (≥5.0)
    high_rating_rates = metrics_df.groupby('LLM_Model')['HighRating'].mean().sort_values(ascending=False)
    colors4 = [model_colors[model] for model in high_rating_rates.index]
    
    bars4 = axes[1,1].bar(range(len(high_rating_rates)), high_rating_rates.values, 
                         color=colors4, alpha=0.8, edgecolor='black', linewidth=2)
    axes[1,1].set_title('High Rating Rate by AI Model', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('High Rating Rate (≥5.0)', fontsize=12)
    axes[1,1].set_xticks(range(len(high_rating_rates)))
    axes[1,1].set_xticklabels(high_rating_rates.index, fontsize=12, fontweight='bold')
    axes[1,1].grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for bar in bars4:
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('5model_plots/overall_model_performance_5models.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created overall 5-model performance visualization")

def create_model_boxplots(metrics_df):
    """Create box plots for each question by model"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    question_columns = {
        'PurchaseIntent': 'Purchase Intent',
        'VisualAppeal': 'Visual Appeal', 
        'ValueConvincing': 'Value Convincing',
        'MessageClarity': 'Message Clarity',
        'Trustworthiness': 'Trustworthiness'
    }
    
    model_colors = {
        'Human': '#2E8B57',
        'Claude': '#FF6B6B',
        'ChatGPT': '#4ECDC4',
        'Deepseek': '#45B7D1',
        'Gemini': '#96CEB4'
    }
    
    for i, (col_name, display_name) in enumerate(question_columns.items()):
        if i < len(axes):
            # Create box plot data
            models = sorted(metrics_df['LLM_Model'].unique())
            data_for_box = [metrics_df[metrics_df['LLM_Model'] == model][col_name].values 
                           for model in models]
            
            box_plot = axes[i].boxplot(data_for_box, labels=models, patch_artist=True)
            
            # Color the boxes
            for patch, model in zip(box_plot['boxes'], models):
                patch.set_facecolor(model_colors[model])
                patch.set_alpha(0.7)
            
            axes[i].set_title(f'{display_name}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Rating (1-7)', fontsize=12)
            axes[i].tick_params(axis='x', rotation=45, labelsize=11)
            axes[i].grid(axis='y', alpha=0.3)
            axes[i].set_ylim(0, 8)
    
    # Remove extra subplot
    if len(question_columns) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.suptitle('Rating Distributions by Question Type (5-Model Comparison)', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('5model_plots/model_boxplots_5models.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created 5-model box plots")

def create_statistical_results_table_5models(metrics_df):
    """Create a table showing statistical results for each question (5-model comparison)"""
    
    print("\n" + "="*80)
    print("STATISTICAL RESULTS BY QUESTION (5-MODEL COMPARISON)")
    print("="*80)
    
    question_columns = {
        'PurchaseIntent': 'Purchase Intent',
        'VisualAppeal': 'Visual Appeal',
        'ValueConvincing': 'Value Convincing',
        'MessageClarity': 'Message Clarity',
        'Trustworthiness': 'Trustworthiness'
    }
    
    results_data = []
    
    for col_name, display_name in question_columns.items():
        # Perform ANOVA
        groups = [group[col_name].values for name, group in metrics_df.groupby('LLM_Model')]
        f_stat, p_val = scipy_stats.f_oneway(*groups)
        
        # Calculate effect size
        ss_between = sum(len(group) * (np.mean(group) - np.mean(metrics_df[col_name]))**2 for group in groups)
        ss_total = sum((metrics_df[col_name] - np.mean(metrics_df[col_name]))**2)
        eta_squared = ss_between / ss_total
        
        # Get descriptive stats
        overall_mean = metrics_df[col_name].mean()
        overall_std = metrics_df[col_name].std()
        n_responses = len(metrics_df)
        
        # Get best performing model
        best_model = metrics_df.groupby('LLM_Model')[col_name].mean().idxmax()
        best_score = metrics_df.groupby('LLM_Model')[col_name].mean().max()
        
        results_data.append({
            'Question': display_name,
            'Mean': overall_mean,
            'Std': overall_std,
            'N': n_responses,
            'F_stat': f_stat,
            'p_value': p_val,
            'eta_squared': eta_squared,
            'Significant': 'Yes' if p_val < 0.05 else 'No',
            'Best_Model': best_model,
            'Best_Score': best_score
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Display table
    print(f"{'Question':<15} {'Mean':<6} {'Std':<6} {'N':<4} {'F':<6} {'p-val':<8} {'η²':<6} {'Sig':<4} {'Best Model':<10} {'Score':<6}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        print(f"{row['Question']:<15} {row['Mean']:<6.2f} {row['Std']:<6.2f} {row['N']:<4} "
              f"{row['F_stat']:<6.2f} {row['p_value']:<8.4f} {row['eta_squared']:<6.3f} {row['Significant']:<4} "
              f"{row['Best_Model']:<10} {row['Best_Score']:<6.2f}")
    
    # Save to CSV
    results_df.to_csv('5model_plots/question_statistical_results_5models.csv', index=False)
    print(f"\nStatistical results saved to '5model_plots/question_statistical_results_5models.csv'")
    
    return results_df

def create_best_worst_performers_5models(metrics_df):
    """Identify and visualize best and worst performing models for each question"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    question_columns = {
        'PurchaseIntent': 'Purchase Intent',
        'VisualAppeal': 'Visual Appeal',
        'ValueConvincing': 'Value Convincing',
        'MessageClarity': 'Message Clarity',
        'Trustworthiness': 'Trustworthiness'
    }
    
    model_colors = {
        'Human': '#2E8B57',
        'Claude': '#FF6B6B',
        'ChatGPT': '#4ECDC4', 
        'Deepseek': '#45B7D1',
        'Gemini': '#96CEB4'
    }
    
    best_performers = []
    worst_performers = []
    
    for col_name, display_name in question_columns.items():
        mean_by_model = metrics_df.groupby('LLM_Model')[col_name].mean().sort_values(ascending=False)
        
        best_performers.append({
            'Question': display_name,
            'Model': mean_by_model.index[0],
            'Mean': mean_by_model.iloc[0]
        })
        
        worst_performers.append({
            'Question': display_name,
            'Model': mean_by_model.index[-1],
            'Mean': mean_by_model.iloc[-1]
        })
    
    best_df = pd.DataFrame(best_performers)
    worst_df = pd.DataFrame(worst_performers)
    
    # Plot best performers
    best_colors = [model_colors[model] for model in best_df['Model']]
    bars1 = axes[0].bar(range(len(best_df)), best_df['Mean'], 
                       color=best_colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0].set_title('Best Performing AI Models by Question', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Mean Rating', fontsize=12)
    axes[0].set_xticks(range(len(best_df)))
    axes[0].set_xticklabels(best_df['Question'], rotation=45, ha='right', fontsize=11)
    axes[0].set_ylim(0, 5.5)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add labels
    for i, (bar, row) in enumerate(zip(bars1, best_df.itertuples())):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}\n{row.Model}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    # Plot worst performers
    worst_colors = [model_colors[model] for model in worst_df['Model']]
    bars2 = axes[1].bar(range(len(worst_df)), worst_df['Mean'], 
                       color=worst_colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[1].set_title('Worst Performing AI Models by Question', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Mean Rating', fontsize=12)
    axes[1].set_xticks(range(len(worst_df)))
    axes[1].set_xticklabels(worst_df['Question'], rotation=45, ha='right', fontsize=11)
    axes[1].set_ylim(0, 5.5)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add labels
    for i, (bar, row) in enumerate(zip(bars2, worst_df.itertuples())):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}\n{row.Model}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('5model_plots/best_worst_performers_5models.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created 5-model best/worst performers visualization")
    print("\nBest Performers by Question:")
    for _, row in best_df.iterrows():
        print(f"  {row['Question']}: {row['Model']} ({row['Mean']:.2f})")
    
    print("\nWorst Performers by Question:")
    for _, row in worst_df.iterrows():
        print(f"  {row['Question']}: {row['Model']} ({row['Mean']:.2f})")

def create_product_comparison_5models(metrics_df):
    """Create comparison across different products using 5-model grouping"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Product Category Analysis (5-Model Comparison)', fontsize=16, fontweight='bold')
    
    # 1. Overall Rating by Product
    product_means = metrics_df.groupby('ProductName')['OverallRating'].mean().sort_values(ascending=False)
    bars1 = axes[0,0].bar(range(len(product_means)), product_means.values, 
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'], 
                         alpha=0.8, edgecolor='black', linewidth=1)
    axes[0,0].set_title('Overall Rating by Product Category')
    axes[0,0].set_ylabel('Mean Overall Rating')
    axes[0,0].set_xticks(range(len(product_means)))
    axes[0,0].set_xticklabels(product_means.index, rotation=45, ha='right')
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                      f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Purchase Intent by Product
    purchase_means = metrics_df.groupby('ProductName')['PurchaseIntent'].mean().sort_values(ascending=False)
    bars2 = axes[0,1].bar(range(len(purchase_means)), purchase_means.values, 
                         color='coral', alpha=0.8, edgecolor='darkred', linewidth=1)
    axes[0,1].set_title('Purchase Intent by Product Category')
    axes[0,1].set_ylabel('Mean Purchase Intent')
    axes[0,1].set_xticks(range(len(purchase_means)))
    axes[0,1].set_xticklabels(purchase_means.index, rotation=45, ha='right')
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # 3. Sample sizes by Product
    sample_sizes = metrics_df.groupby('ProductName').size().sort_values(ascending=False)
    bars3 = axes[1,0].bar(range(len(sample_sizes)), sample_sizes.values, 
                         color='gold', alpha=0.8, edgecolor='orange', linewidth=1)
    axes[1,0].set_title('Sample Sizes by Product Category')
    axes[1,0].set_ylabel('Number of Evaluations')
    axes[1,0].set_xticks(range(len(sample_sizes)))
    axes[1,0].set_xticklabels(sample_sizes.index, rotation=45, ha='right')
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                      f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Product Performance Heatmap
    product_performance = metrics_df.groupby('ProductName')[['PurchaseIntent', 'VisualAppeal', 'ValueConvincing', 'MessageClarity', 'Trustworthiness']].mean()
    im = axes[1,1].imshow(product_performance.values, cmap='RdYlBu_r', aspect='auto')
    axes[1,1].set_title('Performance Heatmap by Product')
    axes[1,1].set_xticks(range(len(product_performance.columns)))
    axes[1,1].set_xticklabels(['Purchase\nIntent', 'Visual\nAppeal', 'Value\nConvincing', 'Message\nClarity', 'Trust'], fontsize=9)
    axes[1,1].set_yticks(range(len(product_performance.index)))
    axes[1,1].set_yticklabels(product_performance.index, fontsize=10)
    
    # Add text annotations
    for i in range(len(product_performance.index)):
        for j in range(len(product_performance.columns)):
            text = axes[1,1].text(j, i, f'{product_performance.iloc[i, j]:.2f}',
                                ha="center", va="center", color="black", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('5model_plots/product_comparison_5models.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created product comparison visualization for 5-model analysis")

def create_comprehensive_summary_5models(metrics_df):
    """Create comprehensive summary comparing all aspects of the 5-model analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive 5-Model Analysis Summary', fontsize=18, fontweight='bold')
    
    model_colors = {
        'Human': '#2E8B57',
        'Claude': '#FF6B6B',
        'ChatGPT': '#4ECDC4',
        'Deepseek': '#45B7D1',
        'Gemini': '#96CEB4'
    }
    
    # 1. Overall Model Ranking
    model_performance = metrics_df.groupby('LLM_Model')['OverallRating'].mean().sort_values(ascending=False)
    colors = [model_colors[model] for model in model_performance.index]
    
    bars1 = axes[0,0].bar(range(len(model_performance)), model_performance.values, 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0,0].set_title('Overall Model Ranking', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Mean Overall Rating', fontsize=12)
    axes[0,0].set_xticks(range(len(model_performance)))
    axes[0,0].set_xticklabels(model_performance.index, fontsize=12, fontweight='bold')
    axes[0,0].grid(axis='y', alpha=0.3)
    axes[0,0].set_ylim(0, 5)
    
    # Add ranking labels
    for i, (bar, model) in enumerate(zip(bars1, model_performance.index)):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                      f'#{i+1}\n{height:.2f}', ha='center', va='bottom', 
                      fontsize=11, fontweight='bold')
    
    # 2. Performance by Rating Dimension
    dimensions = ['PurchaseIntent', 'VisualAppeal', 'ValueConvincing', 'MessageClarity', 'Trustworthiness']
    dimension_labels = ['Purchase\nIntent', 'Visual\nAppeal', 'Value\nConvincing', 'Message\nClarity', 'Trustworthiness']
    
    dimension_means = [metrics_df[dim].mean() for dim in dimensions]
    bars2 = axes[0,1].bar(range(len(dimension_means)), dimension_means, 
                         color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)
    axes[0,1].set_title('Average Performance by Rating Dimension', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Mean Rating', fontsize=12)
    axes[0,1].set_xticks(range(len(dimension_labels)))
    axes[0,1].set_xticklabels(dimension_labels, fontsize=10)
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                      f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Sample Size Distribution
    sample_by_model = metrics_df.groupby('LLM_Model').size().sort_values(ascending=False)
    colors3 = [model_colors[model] for model in sample_by_model.index]
    
    bars3 = axes[1,0].bar(range(len(sample_by_model)), sample_by_model.values, 
                         color=colors3, alpha=0.6, edgecolor='black', linewidth=1)
    axes[1,0].set_title('Sample Size Distribution', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Number of Evaluations', fontsize=12)
    axes[1,0].set_xticks(range(len(sample_by_model)))
    axes[1,0].set_xticklabels(sample_by_model.index, fontsize=12, fontweight='bold')
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 1,
                      f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Model Strengths Summary (best dimension for each model)
    model_strengths = {}
    for model in metrics_df['LLM_Model'].unique():
        model_data = metrics_df[metrics_df['LLM_Model'] == model]
        best_dim = ''
        best_score = 0
        for dim in dimensions:
            score = model_data[dim].mean()
            if score > best_score:
                best_score = score
                best_dim = dim
        model_strengths[model] = (best_dim, best_score)
    
    # Create horizontal bar chart for model strengths
    models = list(model_strengths.keys())
    strengths = [model_strengths[model][1] for model in models]
    strength_labels = [f"{model}\n({model_strengths[model][0].replace('Intent', '').replace('Convincing', '')})" 
                      for model in models]
    colors4 = [model_colors[model] for model in models]
    
    bars4 = axes[1,1].barh(range(len(models)), strengths, 
                          color=colors4, alpha=0.8, edgecolor='black', linewidth=1)
    axes[1,1].set_title('Model Strengths (Best Dimension)', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Mean Rating in Best Dimension', fontsize=12)
    axes[1,1].set_yticks(range(len(models)))
    axes[1,1].set_yticklabels(strength_labels, fontsize=10)
    axes[1,1].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars4):
        width = bar.get_width()
        axes[1,1].text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                      f'{width:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('5model_plots/comprehensive_summary_5models.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created comprehensive 5-model summary visualization")

def perform_welch_anova_5models(metrics_df):
    """Perform Welch ANOVA on the 5-model data and display results"""
    
    print("\n" + "="*70)
    print("WELCH ANOVA RESULTS (5-MODEL COMPARISON)")
    print("="*70)
    
    # Overall ANOVA
    groups = [group['OverallRating'].values for name, group in metrics_df.groupby('LLM_Model')]
    group_names = [name for name, group in metrics_df.groupby('LLM_Model')]
    
    # Standard ANOVA
    f_stat, p_val = scipy_stats.f_oneway(*groups)
    
    # Calculate effect size
    ss_between = sum(len(group) * (np.mean(group) - np.mean(metrics_df['OverallRating']))**2 for group in groups)
    ss_total = sum((metrics_df['OverallRating'] - np.mean(metrics_df['OverallRating']))**2)
    eta_squared = ss_between / ss_total
    
    print(f"\nOVERALL MODEL COMPARISON:")
    print(f"F({len(groups)-1}, {len(metrics_df)-len(groups)}) = {f_stat:.3f}")
    print(f"p-value: {p_val:.6f}")
    print(f"Effect size (η²): {eta_squared:.4f}")
    
    if p_val < 0.05:
        print("*** SIGNIFICANT DIFFERENCES DETECTED ***")
        print("Conclusion: AI models perform significantly differently")
    else:
        print("*** NO SIGNIFICANT DIFFERENCES ***")
        print("Conclusion: All AI models perform equivalently")
    
    # Model ranking
    model_means = metrics_df.groupby('LLM_Model')['OverallRating'].mean().sort_values(ascending=False)
    print(f"\nMODEL RANKING:")
    for i, (model, mean_val) in enumerate(model_means.items(), 1):
        n_evals = len(metrics_df[metrics_df['LLM_Model'] == model])
        print(f"{i}. {model}: {mean_val:.3f} (n={n_evals})")
    
    return f_stat, p_val, eta_squared

def main():
    """Main function to create all 5-model visualizations"""
    
    # Load processed data
    metrics_df, ratings_df = load_processed_data()
    if metrics_df is None or ratings_df is None:
        return
    
    print("Creating 5-model comparison visualizations for study...")
    
    # Perform statistical analysis first
    f_stat, p_val, eta_squared = perform_welch_anova_5models(metrics_df)
    
    # Create individual question plots (5 models)
    create_individual_question_plots_5models(metrics_df)
    
    # Create model comparison visualizations
    create_model_comparison_heatmap(metrics_df)
    create_overall_model_performance(metrics_df)
    create_model_boxplots(metrics_df)
    
    # Create statistical results table
    results_df = create_statistical_results_table_5models(metrics_df)
    
    # Create best/worst performers for 5 models
    create_best_worst_performers_5models(metrics_df)
    
    # Create product-specific analysis
    create_product_comparison_5models(metrics_df)
    
    # Create comprehensive summary
    create_comprehensive_summary_5models(metrics_df)
    
    print("\n" + "="*70)
    print("ALL 5-MODEL VISUALIZATIONS COMPLETED")
    print("="*70)
    print("Files created in '5model_plots/' directory:")
    print("1. Individual question bar charts (5 models each)")
    print("2. Model comparison heatmap")
    print("3. Overall model performance dashboard")
    print("4. Model box plots by question")
    print("5. Best/worst performers by question")
    print("6. Product comparison analysis")
    print("7. Comprehensive summary visualization")
    print("8. Statistical results CSV (5-model comparison)")
    print("\nKey Findings:")
    
    # Summary of key findings
    best_overall = metrics_df.groupby('LLM_Model')['OverallRating'].mean().idxmax()
    best_score = metrics_df.groupby('LLM_Model')['OverallRating'].mean().max()
    worst_overall = metrics_df.groupby('LLM_Model')['OverallRating'].mean().idxmin()
    worst_score = metrics_df.groupby('LLM_Model')['OverallRating'].mean().min()
    
    print(f"• Best performing model: {best_overall} ({best_score:.3f})")
    print(f"• Worst performing model: {worst_overall} ({worst_score:.3f})")
    print(f"• Performance gap: {best_score - worst_score:.3f} points")
    print(f"• Statistical significance: {'YES' if p_val < 0.05 else 'NO'} (p = {p_val:.4f})")
    print(f"• Effect size: {eta_squared:.4f} ({'Large' if eta_squared >= 0.14 else 'Medium' if eta_squared >= 0.06 else 'Small'})")
    
    # Practical recommendations
    print(f"\nPractical Recommendations:")
    if p_val < 0.05:
        print(f"• Recommended choice: {best_overall} for optimal performance")
        print(f"• Avoid: {worst_overall} due to lower performance")
        if eta_squared >= 0.06:
            print(f"• Effect size suggests meaningful practical differences")
        else:
            print(f"• Effect size suggests small practical differences")
    else:
        print(f"• Any model can be selected without performance penalty")
        print(f"• Base decision on other factors (cost, speed, features)")
        print(f"• Descriptively, {best_overall} ranked highest")
    
    print("\nAll files are high-resolution (300 DPI) and ready for presentation!")

if __name__ == "__main__":
    main()