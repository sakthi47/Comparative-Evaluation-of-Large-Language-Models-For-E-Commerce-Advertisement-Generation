import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway, levene, bartlett, shapiro
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(csv_file):
    """
    Load CSV data and clean it for analysis
    """
    print("Loading and cleaning data...")
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Remove the header row with ImportId information (row 2)
    df = df[df['ResponseId'].str.startswith('R_', na=False)].copy()
    
    # Convert numeric columns
    numeric_cols = ['Duration (in seconds)', 'Progress']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean product name column
    df['ProductName'] = df['ProductName'].fillna('Unknown')
    
    # Debug: Print available columns to identify the age column
    print("Available columns:", [col for col in df.columns if 'QID' in col or 'age' in col.lower() or 'Q272' in col])
    
    # Create age group mapping - let's check both possible column names
    age_mapping = {
        1: '18-25',
        2: '26-35', 
        3: '36-45',
        4: '46-55',
        5: '55+'
    }
    
    # Try different possible age column names
    age_column = None
    possible_age_cols = ['QID272', 'Q272', 'What is your age group?']
    for col in possible_age_cols:
        if col in df.columns:
            age_column = col
            break
    
    if age_column:
        df['AgeGroup'] = df[age_column].map(age_mapping)
        print(f"Using age column: {age_column}")
    else:
        df['AgeGroup'] = 'Unknown'
        print("Age column not found, setting all to 'Unknown'")
    
    # Create shopping frequency mapping (QID7)
    shopping_mapping = {
        1: 'Daily',
        2: 'Weekly',
        3: 'Monthly',
        4: 'Rarely'
    }
    
    # Try different possible shopping frequency column names
    shopping_column = None
    possible_shopping_cols = ['QID7', 'How often do you shop online?']
    for col in possible_shopping_cols:
        if col in df.columns:
            shopping_column = col
            break
    
    if shopping_column:
        df['ShoppingFreq'] = df[shopping_column].map(shopping_mapping)
        print(f"Using shopping frequency column: {shopping_column}")
    else:
        df['ShoppingFreq'] = 'Unknown'
        print("Shopping frequency column not found, setting all to 'Unknown'")
    
    print(f"Data loaded: {len(df)} responses")
    print(f"Product categories: {df['ProductName'].value_counts()}")
    
    return df

def extract_ad_ratings(df):
    """
    Extract ad ratings for each product category and ad variant
    """
    print("Extracting ad ratings...")
    
    # Dictionary to store ratings by product and ad
    ratings_data = []
    
    # Define rating questions (Q_1 to Q_5 for each ad)
    rating_questions = ['Q_1', 'Q_2', 'Q_3', 'Q_4', 'Q_5']
    question_names = [
        'Purchase Intent',
        'Visual Appeal', 
        'Value Convincing',
        'Message Clarity',
        'Trustworthiness'
    ]
    
    # Product categories and their column patterns - updated for new dataset
    products = {
        'T-Shirt': range(1, 14),  # Ad 1 to Ad 13
        'Coffee Mug': range(1, 14),
        'Umbrella': range(1, 14),
        'Backpack': range(1, 14),
        'Phonecase': range(1, 14)
    }
    
    for idx, row in df.iterrows():
        response_id = row['ResponseId']
        age_group = row.get('AgeGroup', 'Unknown')
        shopping_freq = row.get('ShoppingFreq', 'Unknown')
        duration = row.get('Duration (in seconds)', 0)
        
        # Get the assigned product for this participant
        assigned_product = row['ProductName'] if 'ProductName' in row.index and pd.notna(row['ProductName']) else None
        
        print(f"Processing ResponseId: {response_id}, Product: {assigned_product}")
        
        if assigned_product and assigned_product in products:
            for ad_num in products[assigned_product]:
                for q_idx, question in enumerate(rating_questions):
                    # Construct column name based on product and ad number
                    col_name = get_column_name(assigned_product, ad_num, q_idx + 1)
                    
                    if col_name in df.columns:
                        rating = pd.to_numeric(row[col_name], errors='coerce')
                        if not pd.isna(rating) and 1 <= rating <= 7:  # Assuming 1-7 scale
                            ratings_data.append({
                                'ResponseId': response_id,
                                'ProductName': assigned_product,
                                'AdNumber': ad_num,
                                'QuestionType': question_names[q_idx],
                                'QuestionNumber': q_idx + 1,
                                'Rating': rating,
                                'AgeGroup': age_group,
                                'ShoppingFreq': shopping_freq,
                                'Duration': duration
                            })
    
    ratings_df = pd.DataFrame(ratings_data)
    print(f"Extracted {len(ratings_df)} individual ratings")
    
    if len(ratings_df) > 0:
        print(f"Unique products: {ratings_df['ProductName'].unique()}")
        print(f"Unique questions: {ratings_df['QuestionType'].unique()}")
    
    return ratings_df

def get_column_name(product, ad_num, q_num):
    """Generate correct column name based on product and ad number"""
    
    if product == 'T-Shirt':
        if ad_num == 1:
            return f'Ad {ad_num} - T-shirt Q_{q_num}'
        elif ad_num <= 3:
            return f'Ad {ad_num} - t shirt_{q_num}'  # Different pattern for ads 2-3
        else:
            return f'Ad {ad_num} T shirt Q_{q_num}'
    
    elif product == 'Coffee Mug':
        return f'ad {ad_num} - coffee mug q_{q_num}'
    
    elif product == 'Umbrella':
        return f'u{ad_num}q_{q_num}'
    
    elif product == 'Backpack':
        return f'b{ad_num}q_{q_num}'
    
    elif product == 'Phonecase':
        return f'p{ad_num}q_{q_num}'
    
    return None

def calculate_ad_metrics(ratings_df):
    """
    Calculate metrics based on actual survey ratings and map to experimental conditions
    """
    print("Calculating ad-based metrics...")
    
    # Group by Response and Ad to get overall metrics
    ad_metrics = []
    
    # Define ad categories as per study design
    ad_categories = {
        1: 'Human Both',
        2: 'Human Image + ChatGPT Copy',
        3: 'Human Image + Claude Copy', 
        4: 'Human Image + Deepseek Copy',
        5: 'Human Image + Gemini Copy',
        6: 'ChatGPT Image + Human Copy',
        7: 'Claude Image + Human Copy',
        8: 'Deepseek Image + Human Copy',
        9: 'Gemini Image + Human Copy',
        10: 'ChatGPT Both',
        11: 'Claude Both',
        12: 'Deepseek Both',
        13: 'Gemini Both'
    }
    
    # LLM model mapping for 5-group analysis
    llm_mapping = {
        1: 'Human',
        2: 'ChatGPT', 3: 'Claude', 4: 'Deepseek', 5: 'Gemini',
        6: 'ChatGPT', 7: 'Claude', 8: 'Deepseek', 9: 'Gemini',
        10: 'ChatGPT', 11: 'Claude', 12: 'Deepseek', 13: 'Gemini'
    }
    
    for (response_id, product, ad_num), group in ratings_df.groupby(['ResponseId', 'ProductName', 'AdNumber']):
        if len(group) >= 5:  # Ensure all 5 questions answered
            purchase_intent = group[group['QuestionType'] == 'Purchase Intent']['Rating'].iloc[0]
            visual_appeal = group[group['QuestionType'] == 'Visual Appeal']['Rating'].iloc[0]
            value_convincing = group[group['QuestionType'] == 'Value Convincing']['Rating'].iloc[0]
            message_clarity = group[group['QuestionType'] == 'Message Clarity']['Rating'].iloc[0]
            trustworthiness = group[group['QuestionType'] == 'Trustworthiness']['Rating'].iloc[0]
            
            # Calculate overall rating (simple average of all 5 questions)
            overall_rating = group['Rating'].mean()
            
            # Binary outcome: high rating (>=5 on 1-7 scale)
            high_rating = 1 if overall_rating >= 5.0 else 0
            
            # Map to experimental conditions
            ad_type = ad_categories.get(ad_num, 'Unknown')
            llm_model = llm_mapping.get(ad_num, 'Unknown')
            
            ad_metrics.append({
                'ResponseId': response_id,
                'ProductName': product,
                'AdNumber': ad_num,
                'AdType': ad_type,
                'LLM_Model': llm_model,
                'PurchaseIntent': purchase_intent,
                'VisualAppeal': visual_appeal,
                'ValueConvincing': value_convincing,
                'MessageClarity': message_clarity,
                'Trustworthiness': trustworthiness,
                'OverallRating': overall_rating,
                'HighRating': high_rating,
                'AgeGroup': group['AgeGroup'].iloc[0],
                'ShoppingFreq': group['ShoppingFreq'].iloc[0]
            })
    
    metrics_df = pd.DataFrame(ad_metrics)
    print(f"Calculated metrics for {len(metrics_df)} ad evaluations")
    
    return metrics_df

def create_model_groups(df):
    """
    Map all ad types to 5 model groups: Human, Claude, ChatGPT, Gemini, Deepseek
    Each model gets credit for all conditions where it was involved
    """
    
    print("\n" + "="*60)
    print("5-MODEL GROUP MAPPING")
    print("="*60)
    
    # The LLM_Model column is already created in calculate_ad_metrics
    # Just verify the mapping
    print("Model Group Assignments:")
    for model in ['Human', 'Claude', 'ChatGPT', 'Deepseek', 'Gemini']:
        model_data = df[df['LLM_Model'] == model]
        if len(model_data) > 0:
            ad_types = model_data['AdType'].unique()
            print(f"\n{model}:")
            for ad_type in ad_types:
                count = len(model_data[model_data['AdType'] == ad_type])
                print(f"  • {ad_type}: {count} evaluations")
    
    # Show group sizes
    print(f"\nTotal Evaluations per Model Group:")
    group_counts = df.groupby('LLM_Model').size().sort_values(ascending=False)
    for model, count in group_counts.items():
        mean_rating = df[df['LLM_Model'] == model]['OverallRating'].mean()
        print(f"  {model}: {count} evaluations (M = {mean_rating:.3f})")
    
    return df

def welch_anova(groups, group_names):
    """
    Perform Welch's ANOVA (one-way ANOVA with unequal variances)
    """
    from scipy import stats
    
    # Calculate group statistics
    k = len(groups)  # number of groups
    n_total = sum(len(group) for group in groups)
    
    # Group means and sample sizes
    means = [np.mean(group) for group in groups]
    ns = [len(group) for group in groups]
    vars = [np.var(group, ddof=1) for group in groups]  # sample variance
    
    # Weights (inverse of variance divided by sample size)
    ws = [n / var if var > 0 else 0 for n, var in zip(ns, vars)]
    
    # Weighted grand mean
    grand_mean = sum(w * mean for w, mean in zip(ws, means)) / sum(ws) if sum(ws) > 0 else np.mean([np.mean(group) for group in groups])
    
    # Welch's F statistic numerator
    numerator = sum(w * (mean - grand_mean)**2 for w, mean in zip(ws, means)) / (k - 1)
    
    # Calculate lambda for denominator
    lambdas = [w / sum(ws) if sum(ws) > 0 else 1/k for w in ws]
    denominator_term = sum((1 - lam)**2 / (n - 1) if n > 1 else 0 for lam, n in zip(lambdas, ns))
    
    # Welch's F statistic
    f_welch = numerator / (1 + (2 * (k - 2) * denominator_term) / (k**2 - 1)) if denominator_term > 0 else 0
    
    # Degrees of freedom for Welch's ANOVA
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * denominator_term) if denominator_term > 0 else float('inf')
    
    # P-value
    p_value = 1 - stats.f.cdf(f_welch, df1, df2) if df2 > 0 and not np.isnan(f_welch) else 1.0
    
    return f_welch, p_value, df1, df2

def check_assumptions_5_models(df):
    """Check assumptions for ANOVA with 5 model groups"""
    
    print("\n" + "="*60)
    print("ASSUMPTION CHECKING (5 MODEL GROUPS)")
    print("="*60)
    
    # 1. Normality test by group
    print("\n1. NORMALITY TESTS (Shapiro-Wilk)")
    print("-" * 40)
    
    normality_results = {}
    for model in df['LLM_Model'].unique():
        model_data = df[df['LLM_Model'] == model]['OverallRating'].values
        if len(model_data) >= 3:  # Shapiro-Wilk needs at least 3 observations
            stat, p_val = shapiro(model_data)
            normality_results[model] = (stat, p_val)
            print(f"{model}: W = {stat:.4f}, p = {p_val:.6f}")
        else:
            print(f"{model}: Too few observations for normality test")
    
    # 2. Homogeneity of variance tests
    print("\n2. HOMOGENEITY OF VARIANCE TESTS")
    print("-" * 40)
    
    groups = [df[df['LLM_Model'] == model]['OverallRating'].values for model in df['LLM_Model'].unique()]
    groups = [group for group in groups if len(group) > 0]
    
    # Levene's test
    levene_stat, levene_p = levene(*groups)
    print(f"Levene's Test: W = {levene_stat:.4f}, p = {levene_p:.6f}")
    
    # Bartlett's test
    bartlett_stat, bartlett_p = bartlett(*groups)
    print(f"Bartlett's Test: χ² = {bartlett_stat:.4f}, p = {bartlett_p:.6f}")
    
    # Interpretation
    print("\n3. ASSUMPTION SUMMARY")
    print("-" * 40)
    
    normal_violations = sum(1 for _, (_, p) in normality_results.items() if p < 0.05)
    print(f"Normality violations: {normal_violations}/{len(normality_results)} groups")
    
    if levene_p < 0.05:
        print("Homogeneity of variance: VIOLATED (Levene's test p < 0.05)")
        print("Recommendation: Use Welch's ANOVA (unequal variances)")
    else:
        print("Homogeneity of variance: SATISFIED (Levene's test p >= 0.05)")
        print("Recommendation: Standard ANOVA acceptable, but Welch's ANOVA still robust")
    
    return normality_results, levene_p, bartlett_p

def perform_model_comparison_anova(df):
    """
    Perform Welch ANOVA analysis comparing the 5 models
    """
    
    print("\n" + "="*70)
    print("WELCH ANOVA ANALYSIS: 5 MODEL COMPARISON")
    print("Human vs Claude vs ChatGPT vs Deepseek vs Gemini")
    print("="*70)
    
    # Check assumptions first
    normality_results, levene_p, bartlett_p = check_assumptions_5_models(df)
    
    # Group by model (5 groups)
    model_order = ['Human', 'Claude', 'ChatGPT', 'Deepseek', 'Gemini']
    groups = [df[df['LLM_Model'] == model]['OverallRating'].values for model in model_order if model in df['LLM_Model'].unique()]
    group_names = [model for model in model_order if model in df['LLM_Model'].unique()]
    
    # Perform Welch's ANOVA
    f_welch, p_welch, df1, df2 = welch_anova(groups, group_names)
    
    # Also perform standard ANOVA for comparison
    f_standard, p_standard = f_oneway(*groups)
    
    # Calculate descriptive statistics
    desc_stats = df.groupby('LLM_Model')['OverallRating'].agg(['count', 'mean', 'std', 'var']).round(4)
    desc_stats = desc_stats.reindex(group_names)  # Order by our preferred order
    
    print("\nDescriptive Statistics by Model:")
    print(desc_stats.sort_values('mean', ascending=False))
    
    print(f"\nWelch's ANOVA Results (k=5 models):")
    print(f"F({df1:.0f}, {df2:.1f}) = {f_welch:.3f}")
    print(f"p-value: {p_welch:.6f}")
    
    print(f"\nStandard ANOVA (for comparison):")
    print(f"F({df1:.0f}, {len(df)-5:.0f}) = {f_standard:.3f}")
    print(f"p-value: {p_standard:.6f}")
    
    # Effect size (eta-squared)
    ss_between = sum(len(group) * (np.mean(group) - np.mean(df['OverallRating']))**2 for group in groups)
    ss_total = sum((df['OverallRating'] - np.mean(df['OverallRating']))**2)
    eta_squared = ss_between / ss_total
    
    print(f"\nEffect Size:")
    print(f"η² = {eta_squared:.4f}")
    
    if eta_squared < 0.01:
        effect_size_interp = "negligible"
    elif eta_squared < 0.06:
        effect_size_interp = "small"
    elif eta_squared < 0.14:
        effect_size_interp = "medium"
    else:
        effect_size_interp = "large"
    
    print(f"Effect size interpretation: {effect_size_interp}")
    
    # Statistical conclusion
    alpha = 0.05
    if p_welch < alpha:
        print(f"\n*** SIGNIFICANT RESULT ***")
        print(f"Reject H₀: Models show significantly different performance (p < {alpha})")
        print("Conclusion: Different AI models (and human baseline) perform differently")
        
        # Show ranking
        model_means = desc_stats.sort_values('mean', ascending=False)['mean']
        print(f"\nModel Performance Ranking:")
        for i, (model, mean_val) in enumerate(model_means.items(), 1):
            print(f"{i}. {model}: {mean_val:.3f}")
            
    else:
        print(f"\n*** NON-SIGNIFICANT RESULT ***")
        print(f"Fail to reject H₀: No significant differences between models (p ≥ {alpha})")
        print("Conclusion: All models (including human baseline) perform equivalently")
    
    return {
        'f_stat': f_welch,
        'p_value': p_welch,
        'df1': df1,
        'df2': df2,
        'eta_squared': eta_squared,
        'effect_size': effect_size_interp,
        'descriptives': desc_stats,
        'significant': p_welch < alpha,
        'model_ranking': desc_stats.sort_values('mean', ascending=False)
    }

def detailed_model_breakdown(df):
    """
    Show detailed breakdown of how each model performed across different task types
    """
    
    print("\n" + "="*70)
    print("DETAILED MODEL PERFORMANCE BREAKDOWN")
    print("="*70)
    
    # Performance by task type for each model
    task_performance = []
    
    for model in ['Human', 'Claude', 'ChatGPT', 'Deepseek', 'Gemini']:
        model_data = df[df['LLM_Model'] == model]
        
        if len(model_data) > 0:
            # Overall performance
            overall_mean = model_data['OverallRating'].mean()
            overall_std = model_data['OverallRating'].std()
            overall_n = len(model_data)
            
            # Performance by task type
            copy_only = model_data[model_data['AdType'].str.contains('Copy', na=False) & 
                                 ~model_data['AdType'].str.contains('Both', na=False)]['OverallRating'].mean()
            image_only = model_data[model_data['AdType'].str.contains('Image', na=False) & 
                                  ~model_data['AdType'].str.contains('Both', na=False)]['OverallRating'].mean()
            both_tasks = model_data[model_data['AdType'].str.contains('Both', na=False)]['OverallRating'].mean()
            
            # Performance by question type
            question_means = {
                'Purchase_Intent': model_data['PurchaseIntent'].mean(),
                'Visual_Appeal': model_data['VisualAppeal'].mean(),
                'Value_Convincing': model_data['ValueConvincing'].mean(),
                'Message_Clarity': model_data['MessageClarity'].mean(),
                'Trustworthiness': model_data['Trustworthiness'].mean()
            }
            
            task_performance.append({
                'Model': model,
                'Overall_Mean': overall_mean,
                'Overall_Std': overall_std,
                'N_Evaluations': overall_n,
                'Copy_Only': copy_only if not pd.isna(copy_only) else np.nan,
                'Image_Only': image_only if not pd.isna(image_only) else np.nan,
                'Both_Tasks': both_tasks if not pd.isna(both_tasks) else np.nan,
                'Purchase_Intent': question_means['Purchase_Intent'],
                'Visual_Appeal': question_means['Visual_Appeal'],
                'Value_Convincing': question_means['Value_Convincing'],
                'Message_Clarity': question_means['Message_Clarity'],
                'Trustworthiness': question_means['Trustworthiness']
            })
    
    task_df = pd.DataFrame(task_performance).sort_values('Overall_Mean', ascending=False)
    
    # Display overall performance table
    print("\nOverall Model Performance:")
    print(f"{'Model':<10} {'Mean':<7} {'Std':<7} {'N':<6} {'Copy':<7} {'Image':<7} {'Both':<7}")
    print("-" * 60)
    
    for _, row in task_df.iterrows():
        copy_val = f"{row['Copy_Only']:.3f}" if not pd.isna(row['Copy_Only']) else "N/A"
        image_val = f"{row['Image_Only']:.3f}" if not pd.isna(row['Image_Only']) else "N/A"
        both_val = f"{row['Both_Tasks']:.3f}" if not pd.isna(row['Both_Tasks']) else "N/A"
        
        print(f"{row['Model']:<10} {row['Overall_Mean']:<7.3f} {row['Overall_Std']:<7.3f} "
              f"{row['N_Evaluations']:<6.0f} {copy_val:<7} {image_val:<7} {both_val:<7}")
    
    # Performance by question type
    print(f"\nPerformance by Question Type:")
    print(f"{'Model':<10} {'Purchase':<9} {'Visual':<8} {'Value':<8} {'Clarity':<8} {'Trust':<8}")
    print("-" * 60)
    
    for _, row in task_df.iterrows():
        print(f"{row['Model']:<10} {row['Purchase_Intent']:<9.3f} {row['Visual_Appeal']:<8.3f} "
              f"{row['Value_Convincing']:<8.3f} {row['Message_Clarity']:<8.3f} {row['Trustworthiness']:<8.3f}")
    
    # Best performer by category
    print(f"\nBest Performer by Category:")
    categories = ['Overall_Mean', 'Copy_Only', 'Image_Only', 'Both_Tasks', 
                 'Purchase_Intent', 'Visual_Appeal', 'Value_Convincing', 'Message_Clarity', 'Trustworthiness']
    category_names = ['Overall', 'Copy Only', 'Image Only', 'Both Tasks',
                     'Purchase Intent', 'Visual Appeal', 'Value Convincing', 'Message Clarity', 'Trustworthiness']
    
    for cat, cat_name in zip(categories, category_names):
        valid_data = task_df[~pd.isna(task_df[cat])]
        if len(valid_data) > 0:
            best_model = valid_data.loc[valid_data[cat].idxmax()]
            print(f"  {cat_name}: {best_model['Model']} ({best_model[cat]:.3f})")
    
    return task_df

def games_howell_models(df):
    """
    Perform Games-Howell post-hoc test for the 5 models
    """
    print("\n" + "="*60)
    print("GAMES-HOWELL POST-HOC ANALYSIS (5 MODELS)")
    print("="*60)
    
    from scipy.stats import t
    from itertools import combinations
    
    # Get groups
    groups = {}
    for model in ['Human', 'Claude', 'ChatGPT', 'Deepseek', 'Gemini']:
        model_data = df[df['LLM_Model'] == model]['OverallRating'].values
        if len(model_data) > 0:
            groups[model] = model_data
    
    group_names = list(groups.keys())
    n_groups = len(group_names)
    
    # Calculate all pairwise comparisons
    results = []
    
    for i, j in combinations(range(n_groups), 2):
        name1, name2 = group_names[i], group_names[j]
        group1, group2 = groups[name1], groups[name2]
        
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Games-Howell test statistic
        pooled_se = np.sqrt((var1/n1) + (var2/n2))
        t_stat = (mean1 - mean2) / pooled_se if pooled_se > 0 else 0
        
        # Welch-Satterthwaite degrees of freedom
        if var1 > 0 and var2 > 0:
            df_welch = ((var1/n1) + (var2/n2))**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        else:
            df_welch = min(n1-1, n2-1)
        
        # Two-tailed p-value
        p_value = 2 * (1 - t.cdf(abs(t_stat), df_welch)) if df_welch > 0 else 1.0
        
        results.append({
            'Model1': name1,
            'Model2': name2,
            'Mean1': mean1,
            'Mean2': mean2,
            'Mean_Diff': mean1 - mean2,
            't_stat': t_stat,
            'df': df_welch,
            'p_value': p_value
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply Holm correction
    p_values = results_df['p_value'].values
    n_comparisons = len(p_values)
    
    sorted_indices = np.argsort(p_values)
    holm_corrected = np.zeros_like(p_values)
    
    for idx, original_idx in enumerate(sorted_indices):
        holm_corrected[original_idx] = min(1.0, p_values[original_idx] * (n_comparisons - idx))
    
    results_df['p_holm'] = holm_corrected
    results_df['significant'] = results_df['p_holm'] < 0.05
    
    # Display results
    print(f"Pairwise Model Comparisons (Games-Howell with Holm correction):")
    print(f"Total comparisons: {n_comparisons}")
    
    for _, row in results_df.iterrows():
        sig_marker = "***" if row['significant'] else ""
        print(f"{row['Model1']} vs {row['Model2']}: "
              f"diff = {row['Mean_Diff']:.3f}, p = {row['p_holm']:.4f} {sig_marker}")
    
    # Summary
    significant_pairs = results_df[results_df['significant']]
    if len(significant_pairs) > 0:
        print(f"\nSignificant differences found:")
        for _, row in significant_pairs.iterrows():
            direction = ">" if row['Mean_Diff'] > 0 else "<"
            print(f"• {row['Model1']} {direction} {row['Model2']} (p = {row['p_holm']:.4f})")
    else:
        print("\nNo significant pairwise differences found after correction.")
    
    return results_df

def save_model_results(df, ratings_df, results, detailed_breakdown):
    """Save results to files"""
    
    with open('welch_anova_model_comparison_results.txt', 'w') as f:
        f.write("Welch ANOVA Analysis Results - Study (5 Model Comparison)\n")
        f.write("="*70 + "\n\n")
        f.write("Study: Leveraging Commercial LLMs for E-Commerce Advertising\n")
        f.write("Research Question: Which AI model performs best for ad generation?\n\n")
        
        f.write(f"MODEL COMPARISON ANALYSIS:\n")
        f.write(f"F({results['df1']:.0f}, {results['df2']:.1f}) = {results['f_stat']:.3f}, p = {results['p_value']:.6f}\n")
        f.write(f"Effect Size (η²): {results['eta_squared']:.4f}\n")
        f.write(f"Conclusion: {'Significant' if results['significant'] else 'Non-significant'} differences found\n\n")
        
        f.write("Model Performance Ranking:\n")
        for i, (model, row) in enumerate(results['model_ranking'].iterrows(), 1):
            f.write(f"{i}. {model}: {row['mean']:.3f} (n={row['count']})\n")
        
        f.write("\nDescriptive Statistics:\n")
        f.write(str(results['descriptives']))
    
    # Save processed data with model mappings
    df.to_csv('processed_data_model_comparison.csv', index=False)
    ratings_df.to_csv('processed_ratings_model_comparison.csv', index=False)
    
    # Save detailed breakdown
    detailed_breakdown.to_csv('model_performance_breakdown.csv', index=False)
    
    print(f"\nResults saved to 'welch_anova_model_comparison_results.txt'")
    print(f"Data with model mappings saved to 'processed_data_model_comparison.csv'")
    print(f"Detailed breakdown saved to 'model_performance_breakdown.csv'")

def perform_question_specific_analysis(df):
    """
    Perform model comparison for each question type separately
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON BY QUESTION TYPE")
    print("="*70)
    
    question_columns = ['PurchaseIntent', 'VisualAppeal', 'ValueConvincing', 'MessageClarity', 'Trustworthiness']
    question_names = ['Purchase Intent', 'Visual Appeal', 'Value Convincing', 'Message Clarity', 'Trustworthiness']
    
    question_results = {}
    
    for col, name in zip(question_columns, question_names):
        print(f"\n--- {name} ---")
        
        # Group by model for this question
        model_order = ['Human', 'Claude', 'ChatGPT', 'Deepseek', 'Gemini']
        groups = [df[df['LLM_Model'] == model][col].values for model in model_order if model in df['LLM_Model'].unique()]
        group_names = [model for model in model_order if model in df['LLM_Model'].unique()]
        
        if len(groups) > 1:
            # Perform Welch ANOVA for this question
            f_welch_q, p_welch_q, df1_q, df2_q = welch_anova(groups, group_names)
            
            # Standard ANOVA for comparison
            f_standard_q, p_standard_q = f_oneway(*groups)
            
            # Descriptive statistics
            desc_stats_q = df.groupby('LLM_Model')[col].agg(['count', 'mean', 'std']).round(3)
            desc_stats_q = desc_stats_q.reindex(group_names)
            
            print(f"Welch F({df1_q:.0f}, {df2_q:.1f}) = {f_welch_q:.3f}, p = {p_welch_q:.6f}")
            
            if p_welch_q < 0.05:
                print("*** Significant differences detected ***")
            else:
                print("No significant differences")
            
            # Show ranking for this question
            ranking = desc_stats_q.sort_values('mean', ascending=False)
            print("Model ranking:")
            for i, (model, row) in enumerate(ranking.iterrows(), 1):
                print(f"  {i}. {model}: {row['mean']:.3f}")
            
            question_results[name] = {
                'f_stat': f_welch_q,
                'p_value': p_welch_q,
                'df1': df1_q,
                'df2': df2_q,
                'significant': p_welch_q < 0.05,
                'ranking': ranking
            }
    
    return question_results

def create_comprehensive_summary(df, results, detailed_breakdown, question_results):
    """
    Create a comprehensive summary of all analyses
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE STUDY SUMMARY")
    print("="*80)
    
    print(f"\nStudy Overview:")
    print(f"• Total participants: {df['ResponseId'].nunique()}")
    print(f"• Total ad evaluations: {len(df)}")
    print(f"• Products tested: {', '.join(df['ProductName'].unique())}")
    print(f"• Models compared: {', '.join(sorted(df['LLM_Model'].unique()))}")
    
    print(f"\nPrimary Research Question: Which AI model performs best for ad generation?")
    
    # Overall results
    print(f"\nOverall Model Comparison:")
    print(f"F({results['df1']:.0f}, {results['df2']:.1f}) = {results['f_stat']:.3f}, p = {results['p_value']:.6f}")
    print(f"Effect size (η²) = {results['eta_squared']:.4f} ({results['effect_size']})")
    
    if results['significant']:
        print(f"*** SIGNIFICANT RESULT: Models perform differently ***")
        best_model = results['model_ranking'].iloc[0]
        worst_model = results['model_ranking'].iloc[-1]
        print(f"Best performer: {best_model.name} (M = {best_model['mean']:.3f})")
        print(f"Worst performer: {worst_model.name} (M = {worst_model['mean']:.3f})")
        print(f"Performance gap: {best_model['mean'] - worst_model['mean']:.3f} points")
    else:
        print(f"*** NON-SIGNIFICANT RESULT: All models perform equivalently ***")
    
    # Complete ranking
    print(f"\nComplete Model Ranking:")
    for i, (model, row) in enumerate(results['model_ranking'].iterrows(), 1):
        print(f"{i}. {model}: {row['mean']:.3f} (n={row['count']}, SD={row['std']:.3f})")
    
    # Question-specific insights
    print(f"\nQuestion-Specific Analysis:")
    sig_questions = [q for q, r in question_results.items() if r['significant']]
    if sig_questions:
        print(f"Significant differences found in: {', '.join(sig_questions)}")
        for question in sig_questions:
            best_q = question_results[question]['ranking'].iloc[0]
            print(f"• {question}: {best_q.name} leads ({best_q['mean']:.3f})")
    else:
        print("No significant differences found in any individual question")
    
    # Task-specific insights
    print(f"\nTask-Specific Performance:")
    task_categories = ['Copy_Only', 'Image_Only', 'Both_Tasks']
    task_names = ['Copy Generation', 'Image Generation', 'Both Tasks']
    
    for cat, name in zip(task_categories, task_names):
        valid_data = detailed_breakdown[~pd.isna(detailed_breakdown[cat])]
        if len(valid_data) > 0:
            best_task = valid_data.loc[valid_data[cat].idxmax()]
            print(f"• {name}: {best_task['Model']} performs best ({best_task[cat]:.3f})")
    
    # Practical recommendations
    print(f"\nPractical Recommendations:")
    if results['significant']:
        best_overall = detailed_breakdown.iloc[0]
        print(f"• Recommended model: {best_overall['Model']} for overall best performance")
        
        # Check if the same model is best across categories
        best_models = set()
        for cat in ['Copy_Only', 'Image_Only', 'Both_Tasks']:
            valid_data = detailed_breakdown[~pd.isna(detailed_breakdown[cat])]
            if len(valid_data) > 0:
                best_models.add(valid_data.loc[valid_data[cat].idxmax()]['Model'])
        
        if len(best_models) == 1:
            print(f"• {list(best_models)[0]} consistently performs best across all task types")
        else:
            print(f"• Different models excel at different tasks - consider task-specific selection")
            
        # Effect size interpretation
        if results['eta_squared'] >= 0.14:
            print(f"• Large effect size suggests meaningful practical differences")
        elif results['eta_squared'] >= 0.06:
            print(f"• Medium effect size suggests moderate practical importance")
        else:
            print(f"• Small effect size - differences may not be practically meaningful")
    else:
        print(f"• Any model can be selected without performance penalty")
        print(f"• Base decision on other factors (cost, speed, integration ease)")
        
        # Show descriptive ranking anyway
        best_descriptive = results['model_ranking'].iloc[0]
        print(f"• Descriptively, {best_descriptive.name} ranked highest ({best_descriptive['mean']:.3f})")

def main():
    """Main function to run the 5-model comparison analysis"""
    
    # Load and process data
    file_path = 'LLM_August 7, 2025_09.13.csv' 
    
    try:
        df = load_and_clean_data(file_path)
        
        if len(df) == 0:
            print("No valid data found. Please check the CSV file.")
            return
        
        # Extract ratings data
        ratings_df = extract_ad_ratings(df)
        
        if len(ratings_df) == 0:
            print("No ratings data extracted. Please check column naming patterns.")
            return
        
        # Calculate ad-based metrics
        metrics_df = calculate_ad_metrics(ratings_df)
        
        if len(metrics_df) == 0:
            print("No metrics calculated. Please check data processing.")
            return
        
        # Create model groups (already done in calculate_ad_metrics)
        metrics_df = create_model_groups(metrics_df)
        
        # Perform 5-model Welch ANOVA analysis
        results = perform_model_comparison_anova(metrics_df)
        
        # Detailed breakdown by task type and question
        detailed_breakdown = detailed_model_breakdown(metrics_df)
        
        # Question-specific analysis
        question_results = perform_question_specific_analysis(metrics_df)
        
        # If significant, perform post-hoc analysis
        if results['significant']:
            post_hoc_results = games_howell_models(metrics_df)
        
        # Save results
        save_model_results(metrics_df, ratings_df, results, detailed_breakdown)
        
        # Create comprehensive summary
        create_comprehensive_summary(metrics_df, results, detailed_breakdown, question_results)
        
        print(f"\nAnalysis completed. Check output files for detailed results.")
        
        return metrics_df, ratings_df, results, detailed_breakdown, question_results
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{file_path}'")
        print("Please make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()