import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io
import re
from itertools import combinations
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar
from google.colab import files

warnings.filterwarnings('ignore')

# Global constants
VALID_EMOTIONS = ['anger', 'fear', 'happiness', 'hate', 'sadness', 'surprise']
BASE_FILES_MODELS = {
    'Claude_emotion_results': 'Claude',
    'DeepSeek_emotion_results': 'DeepSeek',
    'Gemini_emotion_results': 'Gemini',
    'GPT4o_emotion_results': 'GPT4o'
}

def clean_emotion_data(df, columns):
    """Clean emotion data and filter to valid samples"""
    df_clean = df.copy()

    # Convert to lowercase and strip whitespace
    for col in columns:
        if col in df_clean.columns: # Added check
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        else:
            print(f"⚠️  Column '{col}' not found in DataFrame during cleaning.") # Added warning

    # Filter to valid emotions
    mask = True
    if 'emotion' in df_clean.columns: # Check if 'emotion' column exists
        mask = df_clean['emotion'].isin(VALID_EMOTIONS)
        for col in [c for c in columns if c != 'emotion']: # Corrected list comprehension syntax
            if col in df_clean.columns:
                mask &= df_clean[col].isin(VALID_EMOTIONS)
        return df_clean[mask].reset_index(drop=True)
    else:
        # Handle cases where 'emotion' is not expected in the cleaning process (e.g., temp dfs in bootstrap)
        print("⚠️  'emotion' column not found in DataFrame during cleaning. Skipping emotion-based filtering.")
        # Still apply other cleaning like lowercasing and stripping
        for col in columns:
             if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        return df_clean.reset_index(drop=True)


def match_file_to_model(filename):
    """Match uploaded filename to model, handling version numbers"""
    base_name = filename.replace('.csv', '')
    base_name = re.sub(r'\s*\(\d+\)$', '', base_name)
    return BASE_FILES_MODELS.get(base_name)

def load_and_merge_results():
    """Load all 4 CSV files using file upload and merge them into a single dataframe"""
    # Removed introductory print statements
    print("Click the 'Choose Files' button below and select all 4 files:")

    uploaded = files.upload()
    if not uploaded:
        print("❌ No files were uploaded.")
        return None

    print(f"\n✅ Uploaded {len(uploaded)} file(s): {list(uploaded.keys())}")

    merged_df = None
    loaded_models = []

    for filename, file_content in uploaded.items():
        model_name = match_file_to_model(filename)

        if model_name:
            try:
                temp_df = pd.read_csv(io.BytesIO(file_content))
                print(f"✓ Processed {filename} as {model_name} with {len(temp_df)} rows")

                if merged_df is None:
                    merged_df = temp_df.copy()
                    merged_df = merged_df.rename(columns={'predicted_emotion': model_name})
                else:
                    merged_df[model_name] = temp_df['predicted_emotion']

                loaded_models.append(model_name)

            except Exception as e:
                print(f"⚠️  Error processing {filename}: {str(e)}")
        else:
            print(f"⚠️  Could not match {filename} to any expected model file")

    if merged_df is None or len(loaded_models) < 2:
        print(f"\n❌ Need at least 2 models for comparison. Only loaded: {loaded_models}")
        return None

    print(f"\n✅ Successfully loaded {len(loaded_models)} model files: {loaded_models}")
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Columns: {list(merged_df.columns)}") # Added print for columns
    print(f"First 3 rows preview:")
    display(merged_df.head(3)) # Changed to display for better formatting

    return merged_df

def bootstrap_metric(y_true, y_pred, metric_func, metric_name, n_bootstrap=1000, confidence_level=0.95, **kwargs):
    """Generic bootstrap function for any metric"""
    df_temp = pd.DataFrame({'emotion': y_true, 'pred': y_pred})
    df_clean = clean_emotion_data(df_temp, ['emotion', 'pred'])

    if len(df_clean) < 10:
        print(f"⚠️  Too few valid samples ({len(df_clean)}) for bootstrap {metric_name}")
        return None

    n_samples = len(df_clean)
    bootstrap_scores = []
    np.random.seed(42)

    for _ in range(n_bootstrap):
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_true = df_clean['emotion'].iloc[bootstrap_indices]
        bootstrap_pred = df_clean['pred'].iloc[bootstrap_indices]

        try:
            score = metric_func(bootstrap_true, bootstrap_pred, **kwargs)
            bootstrap_scores.append(score)
        except Exception as e:
            print(f"⚠️  Error during bootstrap {metric_name} calculation: {e}")
            continue

    if len(bootstrap_scores) == 0:
        return None

    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100

    return {
        f'mean_{metric_name.lower()}': np.mean(bootstrap_scores),
        f'std_{metric_name.lower()}': np.std(bootstrap_scores),
        'ci_lower': np.percentile(bootstrap_scores, lower_percentile),
        'ci_upper': np.percentile(bootstrap_scores, upper_percentile),
        'bootstrap_samples': bootstrap_scores,
        'valid_samples': len(df_clean)
    }

def bootstrap_accuracy(y_true, y_pred, **kwargs):
    """Calculate bootstrap confidence interval for accuracy"""
    return bootstrap_metric(y_true, y_pred, accuracy_score, 'accuracy', **kwargs)

def bootstrap_f1_score(y_true, y_pred, **kwargs):
    """Calculate bootstrap confidence interval for F1-score"""
    return bootstrap_metric(y_true, y_pred, f1_score, 'f1', average='macro', zero_division=0, **kwargs)

def mcnemar_test(y_true, y_pred1, y_pred2, model1_name, model2_name):
    """Perform McNemar's test to compare two models"""
    df_temp = pd.DataFrame({'emotion': y_true, 'pred1': y_pred1, 'pred2': y_pred2})
    df_clean = clean_emotion_data(df_temp, ['emotion', 'pred1', 'pred2'])

    if len(df_clean) == 0:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'model1_better': 0,
            'model2_better': 0,
            'interpretation': f"No valid samples for comparison between {model1_name} and {model2_name}"
        }

    # Create contingency table based on agreement/disagreement with true labels
    correct1 = (df_clean['emotion'] == df_clean['pred1']).astype(int)
    correct2 = (df_clean['emotion'] == df_clean['pred2']).astype(int)

    both_correct = np.sum((correct1 == 1) & (correct2 == 1))
    model1_only = np.sum((correct1 == 1) & (correct2 == 0))
    model2_only = np.sum((correct1 == 0) & (correct2 == 1))
    both_wrong = np.sum((correct1 == 0) & (correct2 == 0))

    contingency_table = np.array([[both_correct, model1_only],
                                  [model2_only, both_wrong]])

    if model1_only + model2_only == 0:
        return {
            'statistic': 0,
            'p_value': 1.0,
            'model1_better': model1_only,
            'model2_better': model2_only,
            'contingency_table': contingency_table,
            'interpretation': f"No disagreement between {model1_name} and {model2_name} on {len(df_clean)} valid samples"
        }

    try:
        result = mcnemar(contingency_table, exact=True)
        test_stat = result.statistic
        p_value = result.pvalue
    except ValueError:
        # Added a check to prevent division by zero if model1_only + model2_only is 0
        if (model1_only + model2_only) > 0:
            test_stat = ((abs(model1_only - model2_only) - 1) ** 2) / (model1_only + model2_only)
            p_value = 1 - stats.chi2.cdf(test_stat, df=1)
            print(f"⚠️  Exact McNemar test failed for {model1_name} vs {model2_name}, using chi-squared approximation.")
        else:
             test_stat = np.nan
             p_value = np.nan
             print(f"⚠️  Exact McNemar test failed and denominator is zero for {model1_name} vs {model2_name}.")


    return {
        'statistic': test_stat,
        'p_value': p_value,
        'model1_better': model1_only,
        'model2_better': model2_only,
        'contingency_table': contingency_table,
        'interpretation': f"{model1_name} vs {model2_name}: {'Significant difference' if not np.isnan(p_value) and p_value < 0.05 else ('No significant difference' if not np.isnan(p_value) else 'Test not applicable/failed')} (p={p_value:.4f})" if not np.isnan(p_value) else f"{model1_name} vs {model2_name}: Test not applicable/failed on {len(df_clean)} valid samples" # Improved interpretation message
    }

def calculate_model_metrics(df, model_columns):
    """Calculate basic metrics for all models"""
    results = {}
    print("\nCalculating basic metrics...")

    # Clean the dataframe based on the 'emotion' column
    df_emotion_clean = clean_emotion_data(df, ['emotion'])

    if len(df_emotion_clean) == 0:
        print("⚠️  No valid samples found for basic metrics calculation.")
        return {}

    for model in model_columns:
        # For each model, filter for valid emotion and valid model predictions
        df_clean_model = clean_emotion_data(df_emotion_clean, [model])

        if len(df_clean_model) == 0:
            results[model] = {'accuracy': np.nan, 'f1_macro': np.nan, 'f1_weighted': np.nan, 'valid_samples': 0}
            continue

        accuracy = accuracy_score(df_clean_model['emotion'], df_clean_model[model])
        f1_macro = f1_score(df_clean_model['emotion'], df_clean_model[model], average='macro', zero_division=0)
        f1_weighted = f1_score(df_clean_model['emotion'], df_clean_model[model], average='weighted', zero_division=0)

        results[model] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'valid_samples': len(df_clean_model) # Use the count after filtering for this specific model
        }
        print(f"  Calculated metrics for {model} on {len(df_clean_model)} valid samples.")

    return results

def perform_statistical_analysis(df):
    """Perform comprehensive statistical analysis"""
    expected_models = ['Claude', 'DeepSeek', 'Gemini', 'GPT4o']
    model_columns = [col for col in expected_models if col in df.columns]

    if len(model_columns) < 2:
        print("❌ Need at least 2 models for statistical comparison")
        return None

    print("=" * 60)
    print("STATISTICAL ANALYSIS RESULTS")
    print("=" * 60)

    # 1. Basic metrics
    print("\n1. BASIC METRICS")
    print("-" * 40)
    basic_metrics = calculate_model_metrics(df, model_columns)
    if not basic_metrics:
        return None

    metrics_df = pd.DataFrame(basic_metrics).T
    display(metrics_df[['accuracy', 'f1_macro', 'f1_weighted', 'valid_samples']].round(4)) # Use display for DataFrame

    # 2. Bootstrap analysis
    print("\n\n2. BOOTSTRAP CONFIDENCE INTERVALS")
    print("-" * 40)
    bootstrap_results = {}

    for model in model_columns:
        print(f"\nBootstrap Analysis for {model}:")
        # Ensure 'emotion' column exists before calling bootstrap functions
        if 'emotion' in df.columns:
            acc_bootstrap = bootstrap_accuracy(df['emotion'], df[model])
            f1_bootstrap = bootstrap_f1_score(df['emotion'], df[model])

            bootstrap_results[model] = {'accuracy': acc_bootstrap, 'f1_macro': f1_bootstrap}

            if acc_bootstrap:
                print(f"  Accuracy: {acc_bootstrap['mean_accuracy']:.4f} ± {acc_bootstrap['std_accuracy']:.4f}")
                print(f"  95% CI: [{acc_bootstrap['ci_lower']:.4f}, {acc_bootstrap['ci_upper']:.4f}] (Valid samples: {acc_bootstrap['valid_samples']})")
            else:
                print(f"  Accuracy: Not enough valid samples for bootstrap.")

            if f1_bootstrap:
                print(f"  F1-Macro: {f1_bootstrap['mean_f1']:.4f} ± {f1_bootstrap['std_f1']:.4f}") # Corrected variable name
                print(f"  95% CI: [{f1_bootstrap['ci_lower']:.4f}, {f1_bootstrap['ci_upper']:.4f}] (Valid samples: {f1_bootstrap['valid_samples']})") # Corrected variable name
            else:
                print(f"  F1-Macro: Not enough valid samples for bootstrap.")
        else:
            print(f"⚠️  'emotion' column not found in main DataFrame. Skipping bootstrap for {model}.")
            bootstrap_results[model] = {'accuracy': None, 'f1_macro': None} # Add None results


    # 3. McNemar's test
    print("\n\n3. MCNEMAR'S TEST (Pairwise Model Comparisons)")
    print("-" * 40)
    mcnemar_results = {}

    # Ensure 'emotion' column exists before performing McNemar's test
    if 'emotion' in df.columns:
        for model1, model2 in combinations(model_columns, 2):
            print(f"\n{model1} vs {model2}:")
            mcnemar_result = mcnemar_test(df['emotion'], df[model1], df[model2], model1, model2)
            mcnemar_results[f"{model1}_vs_{model2}"] = mcnemar_result

            if not np.isnan(mcnemar_result['p_value']):
                print(f"  Chi-square statistic: {mcnemar_result['statistic']:.4f}")
                print(f"  p-value: {mcnemar_result['p_value']:.4f}")
                print(f"  {model1} better in {mcnemar_result['model1_better']} cases")
                print(f"  {model2} better in {mcnemar_result['model2_better']} cases")
                print(f"  Result: {mcnemar_result['interpretation']}")
            else:
                print(f"  Result: {mcnemar_result['interpretation']}")
    else:
         print("\n⚠️  'emotion' column not found in main DataFrame. Skipping McNemar's test.")


    return {
        'basic_metrics': basic_metrics,
        'bootstrap_results': bootstrap_results,
        'mcnemar_results': mcnemar_results
    }

def create_visualizations(df, results):
    """Create visualization plots for the results"""
    print("\nCreating visualizations...")

    expected_models = ['Claude', 'DeepSeek', 'Gemini', 'GPT4o']
    model_columns = [col for col in expected_models if col in df.columns]

    if len(model_columns) < 1:
        print("❌ Not enough models available to create visualizations.")
        return

    df_viz = clean_emotion_data(df, ['emotion'] + model_columns)
    if len(df_viz) == 0:
        print("❌ No valid samples found for visualizations.")
        return

    plt.style.use('default')
    sns.set_palette("husl")

    # Determine number of plots
    n_plots = 0
    if results and 'bootstrap_results' in results and any(results['bootstrap_results'][m] and results['bootstrap_results'][m]['accuracy'] is not None for m in model_columns): # Check for valid bootstrap results
        n_plots += 1  # Accuracy plot
    if results and 'bootstrap_results' in results and any(results['bootstrap_results'][m] and results['bootstrap_results'][m]['f1_macro'] is not None for m in model_columns): # Check for valid bootstrap results
         n_plots += 1 # F1 plot

    if results and 'mcnemar_results' in results and any(not np.isnan(v['p_value']) for v in results.get('mcnemar_results', {}).values()): # Check for valid McNemar results
        n_plots += 1  # McNemar heatmap

    if 'emotion' in df_viz.columns: # Check if 'emotion' column exists for per-class accuracy
        n_plots += 1  # Per-class accuracy


    if n_plots == 0:
        print("❌ No valid results or data available to create visualizations.")
        return

    # Set up subplot layout
    if n_plots <= 2:
        nrows, ncols = 1, n_plots
    elif n_plots == 3:
        nrows, ncols = 1, 3
    else:
        nrows, ncols = 2, 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.5, nrows * 6))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    plot_idx = 0

    # 1. Accuracy comparison with confidence intervals
    if results and 'bootstrap_results' in results:
        valid_models_acc = [m for m in model_columns if results['bootstrap_results'].get(m) and results['bootstrap_results'][m]['accuracy'] is not None]
        if valid_models_acc:
            ax = axes[plot_idx]
            accuracies = [results['bootstrap_results'][model]['accuracy']['mean_accuracy'] for model in valid_models_acc]
            ci_lower = [results['bootstrap_results'][model]['accuracy']['ci_lower'] for model in valid_models_acc]
            ci_upper = [results['bootstrap_results'][model]['accuracy']['ci_upper'] for model in valid_models_acc]

            x_pos = np.arange(len(valid_models_acc))
            bars = ax.bar(x_pos, accuracies, yerr=[np.array(accuracies) - np.array(ci_lower),
                                                   np.array(ci_upper) - np.array(accuracies)],
                         capsize=5, alpha=0.7)

            ax.set_xlabel('Models')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy with 95% Confidence Intervals')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(valid_models_acc, rotation=45)
            ax.grid(True, alpha=0.3)

            for bar, acc in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
            plot_idx += 1

    # 2. F1-Score comparison
    if results and 'bootstrap_results' in results:
        valid_models_f1 = [m for m in model_columns if results['bootstrap_results'].get(m) and results['bootstrap_results'][m]['f1_macro'] is not None]
        if valid_models_f1:
            ax = axes[plot_idx]
            f1_scores = [results['bootstrap_results'][model]['f1_macro']['mean_f1'] for model in valid_models_f1]
            f1_ci_lower = [results['bootstrap_results'][model]['f1_macro']['ci_lower'] for model in valid_models_f1]
            f1_ci_upper = [results['bootstrap_results'][model]['f1_macro']['ci_upper'] for model in valid_models_f1]

            x_pos = np.arange(len(valid_models_f1))
            bars = ax.bar(x_pos, f1_scores, yerr=[np.array(f1_scores) - np.array(f1_ci_lower),
                                                  np.array(f1_ci_upper) - np.array(f1_scores)],
                         capsize=5, alpha=0.7, color='orange')

            ax.set_xlabel('Models')
            ax.set_ylabel('F1-Score (Macro)')
            ax.set_title('F1-Score with 95% Confidence Intervals')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(valid_models_f1, rotation=45)
            ax.grid(True, alpha=0.3)

            for bar, f1 in zip(bars, f1_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{f1:.3f}', ha='center', va='bottom')
            plot_idx += 1

    # 3. McNemar's test p-values heatmap
    if results and 'mcnemar_results' in results and any(not np.isnan(v['p_value']) for v in results.get('mcnemar_results', {}).values()):
        ax = axes[plot_idx]
        models = [m for m in model_columns if m in results['basic_metrics']] # Use models that had basic metrics calculated
        p_value_matrix = np.ones((len(models), len(models))) * np.nan # Initialize with NaN

        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:
                    key1 = f"{model1}_vs_{model2}"
                    key2 = f"{model2}_vs_{model1}"
                    if key1 in results['mcnemar_results']:
                        p_val = results['mcnemar_results'][key1]['p_value']
                        if not np.isnan(p_val):
                            p_value_matrix[i, j] = p_val
                            p_value_matrix[j, i] = p_val
                    elif key2 in results['mcnemar_results']: # Check the reversed key as well
                         p_val = results['mcnemar_results'][key2]['p_value']
                         if not np.isnan(p_val):
                            p_value_matrix[i, j] = p_val
                            p_value_matrix[j, i] = p_val


        sns.heatmap(p_value_matrix, annot=True, fmt=".4f", xticklabels=models, yticklabels=models,
                   cmap='RdYlBu_r', center=0.05, ax=ax, cbar_kws={'label': 'p-value'})
        ax.set_title("McNemar's Test p-values\n(Red: p<0.05, Blue: p>0.05)")
        plot_idx += 1

    # 4. Per-class accuracy
    if 'emotion' in df_viz.columns:
        ax = axes[plot_idx]
        emotion_classes = sorted(df_viz['emotion'].unique())
        class_accuracies = {}

        for model in model_columns:
            class_acc = []
            for emotion in emotion_classes:
                mask = df_viz['emotion'] == emotion
                if mask.sum() > 0:
                    correct = (df_viz.loc[mask, 'emotion'] == df_viz.loc[mask, model]).sum()
                    total = mask.sum()
                    accuracy = correct / total
                else:
                    accuracy = 0
                class_acc.append(accuracy)
            class_accuracies[model] = class_acc

        x = np.arange(len(emotion_classes))
        width = 0.8 / len(model_columns)

        for i, model in enumerate(model_columns):
            ax.bar(x + i * width, class_accuracies[model], width, label=model, alpha=0.8)

        ax.set_xlabel('Emotion Classes')
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Class Accuracy by Model')
        ax.set_xticks(x + width * (len(model_columns)-1) / 2)
        ax.set_xticklabels(emotion_classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Remove unused subplots
    for i in range(plot_idx, len(axes)): # Start from plot_idx
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('emotion_analysis_statistical_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_results_to_file(results, df):
    """Save all results to a comprehensive report file"""
    print("\nSaving results to file...")

    with open('statistical_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("EMOTION DETECTION STATISTICAL TESTING REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("DATASET INFORMATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total samples loaded: {len(df)}\n")

        valid_df = clean_emotion_data(df, ['emotion'])
        if 'emotion' in valid_df.columns:
             f.write(f"Total valid samples for analysis: {len(valid_df)}\n")

             if len(valid_df) > 0:
                f.write(f"Emotion classes found: {valid_df['emotion'].unique()}\n")
                f.write(f"Class distribution:\n{valid_df['emotion'].value_counts().to_string()}\n\n") # Use to_string for better formatting in file
             else:
                f.write("No valid emotion classes found.\n\n")
        else:
            f.write("Could not determine total valid samples or class distribution as 'emotion' column is missing.\n\n")


        f.write("BASIC METRICS:\n")
        f.write("-" * 20 + "\n")
        if results and 'basic_metrics' in results:
            metrics_df = pd.DataFrame(results['basic_metrics']).T
            f.write(metrics_df[['accuracy', 'f1_macro', 'f1_weighted', 'valid_samples']].round(4).to_string()) # Use to_string
        else:
            f.write("Basic metrics could not be calculated.\n")
        f.write("\n\n")

        f.write("BOOTSTRAP CONFIDENCE INTERVALS (95%):\n")
        f.write("-" * 40 + "\n")
        if results and 'bootstrap_results' in results:
            for model, bootstrap_data in results['bootstrap_results'].items():
                 if bootstrap_data: # Check if bootstrap_data is not None
                    f.write(f"\n{model}:\n")
                    acc_data = bootstrap_data.get('accuracy')
                    f1_data = bootstrap_data.get('f1_macro')

                    if acc_data:
                        f.write(f"  Accuracy: {acc_data['mean_accuracy']:.4f} ± {acc_data['std_accuracy']:.4f} ")
                        f.write(f"[{acc_data['ci_lower']:.4f}, {acc_data['ci_upper']:.4f}] (Valid samples: {acc_data['valid_samples']})\n")
                    else:
                        f.write("  Accuracy: Not enough valid samples or calculation failed.\n")

                    if f1_data:
                        f.write(f"  F1-Macro: {f1_data['mean_f1']:.4f} ± {f1_data['std_f1']:.4f} ")
                        f.write(f"[{f1_data['ci_lower']:.4f}, {f1_data['ci_upper']:.4f}] (Valid samples: {f1_data['valid_samples']})\n")
                    else:
                        f.write("  F1-Macro: Not enough valid samples or calculation failed.\n")
                 else:
                    f.write(f"\n{model}:\n  Bootstrap results not available.\n") # Indicate if no bootstrap data was generated


        f.write("\n\nMCNEMAR'S TEST RESULTS:\n")
        f.write("-" * 30 + "\n")
        if results and 'mcnemar_results' in results:
            for comparison, mcnemar_data in results['mcnemar_results'].items():
                f.write(f"\n{comparison.replace('_vs_', ' vs ')}:\n")
                if not np.isnan(mcnemar_data['p_value']):
                    f.write(f"  Chi-square statistic: {mcnemar_data['statistic']:.4f}\n")
                    f.write(f"  p-value: {mcnemar_data['p_value']:.4f}\n")
                    f.write(f"  {comparison.split('_vs_')[0]} better in {mcnemar_data['model1_better']} cases\n")
                    f.write(f"  {comparison.split('_vs_')[1]} better in {mcnemar_data['model2_better']} cases\n")
                    f.write(f"  Interpretation: {mcnemar_data['interpretation']}\n")
                else:
                    f.write(f"  Interpretation: {mcnemar_data['interpretation']}\n")
        else:
            f.write("McNemar's test results could not be calculated or are not available.\n")

        f.write("\n\nVISUALIZATIONS:\n")
        f.write("-" * 20 + "\n")
        f.write("Plots are saved as 'emotion_analysis_statistical_comparison.png'.\n")

        f.write("\n\nEND OF REPORT\n")
        f.write("=" * 60 + "\n")

    print("✅ Report saved to statistical_analysis_report.txt")

def main():
    """Main function to execute the complete analysis"""
    print("EMOTION DETECTION STATISTICAL TESTING")
    print("=" * 50)
    print("Click the 'Choose Files' button below and select all 4 files:")
    print("=" * 50)

    # Load and merge data
    print("\nStep 1: Loading and merging data...")
    merged_df = load_and_merge_results()
    if merged_df is None:
        print("❌ Failed to load files. Please try again.")
        return

    # Save merged dataset
    try:
        merged_df.to_csv('merged_emotion_results.csv', index=False, encoding='utf-8-sig')
        print("✅ Merged dataset saved as 'merged_emotion_results.csv'")
    except Exception as e:
        print(f"⚠️  Error saving merged dataset: {e}")

    # Perform statistical analysis
    print("\nStep 2: Performing statistical analysis...")
    analysis_results = perform_statistical_analysis(merged_df)
    if analysis_results is None:
        print("❌ Statistical analysis could not be completed.")
        # Initialize an empty dictionary to avoid errors in subsequent steps
        analysis_results = {}


    # Create visualizations
    print("\nStep 3: Creating visualizations...")
    create_visualizations(merged_df, analysis_results)

    # Save results to file
    print("\nStep 4: Saving comprehensive report...")
    save_results_to_file(analysis_results, merged_df)

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETED!")
    print("Files generated:")
    print("  - merged_emotion_results.csv")
    print("  - emotion_analysis_statistical_comparison.png")
    print("  - statistical_analysis_report.txt")
    print("=" * 50)


# Execute the main function
if __name__ == "__main__":
    main()