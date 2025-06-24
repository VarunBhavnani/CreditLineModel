import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Modeling & preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Model evaluation
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)

# Statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def generate_monthly_features(df,months_info):
    
    dfs = []
    
    for month,pay_col,bill_col,pay_amt_col in months_info:
        temp_df = df[['ID','LIMIT_BAL',pay_col,bill_col,pay_amt_col]].copy()
        
        temp_df[f'utilization_ratio {month}'] = temp_df[bill_col]/temp_df['LIMIT_BAL']

        temp_df[f'Paid {month}'] = np.where(temp_df[pay_amt_col] > 0, 1,0)

        temp_df[f'Delinquent {month}'] = np.where(temp_df[pay_col] >= 1, 1,0)
        
        keep_cols = ['ID',f'utilization_ratio {month}',f'Paid {month}',f'Delinquent {month}']
        
        dfs.append(temp_df[keep_cols])
    
    for temp_df in dfs:
        df = pd.merge(df,temp_df,on='ID',how='inner')
            
    return df

def reorder_columns_monthwise(df):
    base_cols = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
    target_col = ['default payment next month']
    
    months = ['Sep', 'Aug', 'Jul', 'Jun', 'May', 'Apr']
    ordered_month_cols = []

    for month in months:
        for prefix in ['PAY', 'BILL_AMT', 'PAY_AMT', 'utilization_ratio', 'Paid', 'Delinquent']:
            col_name = f'{prefix} {month}' if prefix in ['utilization_ratio', 'Paid', 'Delinquent'] else f'{prefix}_{month}'
            if col_name in df.columns:
                ordered_month_cols.append(col_name)

    new_order = base_cols + ordered_month_cols + target_col
    return df[new_order]

#--------------------------- WOE, IV ---------------------------#
def iv_woe(df_1, target):
    all_bins = []
    epsilon = 1e-10
    
    for col in df_1.columns:
        if col == target:
            continue
        
        df = df_1[[col, target]].copy()
        
        if df[col].dtype != 'object' and pd.api.types.is_numeric_dtype(df[col]):
            try:
                df['bin'] = pd.qcut(df[col], 10, duplicates='drop')
                group_col = 'bin'
            except Exception as e:
                print(f"Could not bin numeric column {col}: {e}")
                continue
        else:
            df['bin'] = df[col].astype(str)
            group_col = 'bin'

        summary = df.groupby(group_col)[target].agg(['count','sum']).reset_index()
        summary.rename(columns={'count': 'Total Observations', 'sum': 'Events'}, inplace=True)
        summary['Non events'] = summary['Total Observations'] - summary['Events']
        summary['% of Non events'] = summary['Non events'] / summary['Non events'].sum()
        summary['% of Events'] = summary['Events'] / summary['Events'].sum()
        summary['% of Non events'] = summary['% of Non events'].replace(0, epsilon)
        summary['% of Events'] = summary['% of Events'].replace(0, epsilon)
        summary['WoE'] = np.log(summary['% of Non events'] / summary['% of Events'])
        summary['IV'] = summary['WoE'] * (summary['% of Non events'] - summary['% of Events'])
        summary['Feature'] = col
        summary = summary[['Feature', 'bin', 'Total Observations', 'Non events', 'Events',
                           '% of Non events', '% of Events', 'WoE', 'IV']]
        all_bins.append(summary)
    
    woe_df = pd.concat(all_bins, ignore_index=True)
    iv_df = woe_df.groupby('Feature')['IV'].sum().reset_index(name='IV')
    return woe_df, iv_df

def parse_bounds(bin_label):
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", bin_label)
    if len(matches) >= 2:
        return float(matches[0]), float(matches[1])
    else:
        return None, None

def merge_small_bins(woe_df, feature, min_bin_pct=0.02):
    woe_df_temp = woe_df[woe_df['Feature'] == feature].copy()

    # Skip merging for categorical bins
    try:
        pd.Interval(0, 1)  # test if Interval exists
        test_bin = woe_df_temp['bin'].iloc[0]
        if not isinstance(test_bin, pd.Interval):
            return woe_df_temp
    except:
        return woe_df_temp

    total_obs = woe_df_temp['Total Observations'].sum()
    min_bin_cnt = total_obs * min_bin_pct

    while (woe_df_temp['Total Observations'] < min_bin_cnt).any():
        small_bin_indx = woe_df_temp[woe_df_temp['Total Observations'] < min_bin_cnt].index
        if len(small_bin_indx) == 0:
            break

        idx = small_bin_indx[0]
        if idx == 0:
            merge_with = idx + 1
        else:
            merge_with = idx - 1

        # Merge label
        left_bound, _ = parse_bounds(str(woe_df_temp.at[merge_with, 'bin']))
        _, right_bound = parse_bounds(str(woe_df_temp.at[idx, 'bin']))
        if left_bound is not None and right_bound is not None:
            merged_bin_label = f"({left_bound}, {right_bound}]"
        else:
            merged_bin_label = f"{woe_df_temp.at[merge_with, 'bin']} + {woe_df_temp.at[idx, 'bin']}"

        woe_df_temp.at[merge_with, 'bin'] = merged_bin_label

        for col in ['Total Observations', 'Events', 'Non events']:
            woe_df_temp.at[merge_with, col] += woe_df_temp.at[idx, col]

        woe_df_temp = woe_df_temp.drop(index=idx).reset_index(drop=True)

        # Recalculate WoE/IV
        epsilon = 1e-10
        woe_df_temp['% of Non events'] = woe_df_temp['Non events'] / woe_df_temp['Non events'].sum()
        woe_df_temp['% of Events'] = woe_df_temp['Events'] / woe_df_temp['Events'].sum()
        woe_df_temp['% of Non events'] = woe_df_temp['% of Non events'].replace(0, epsilon)
        woe_df_temp['% of Events'] = woe_df_temp['% of Events'].replace(0, epsilon)
        woe_df_temp['WoE'] = np.log(woe_df_temp['% of Non events'] / woe_df_temp['% of Events'])
        woe_df_temp['IV'] = (woe_df_temp['% of Non events'] - woe_df_temp['% of Events']) * woe_df_temp['WoE']

    return woe_df_temp

def is_monotonic(col_series):
    return col_series.is_monotonic_increasing or col_series.is_monotonic_decreasing

def bin_merge(woe_df, feature):
    woe_df_temp = woe_df[woe_df['Feature'] == feature].copy()

    # Skip if not numeric bins
    try:
        test_bin = woe_df_temp['bin'].iloc[0]
        if not isinstance(test_bin, pd.Interval):
            return woe_df_temp
    except:
        return woe_df_temp

    while not is_monotonic(woe_df_temp['Total Observations']):
        diffs = woe_df_temp['WoE'].diff().abs()
        min_diff_index = diffs[1:].idxmin()
        previous_index = min_diff_index - 1

        left_bound, _ = parse_bounds(str(woe_df_temp.at[previous_index, 'bin']))
        _, right_bound = parse_bounds(str(woe_df_temp.at[min_diff_index, 'bin']))
        if left_bound is not None and right_bound is not None:
            merged_bin_label = f"({left_bound}, {right_bound}]"
        else:
            merged_bin_label = f"{woe_df_temp.at[previous_index, 'bin']} + {woe_df_temp.at[min_diff_index, 'bin']}"

        woe_df_temp.at[previous_index, 'bin'] = merged_bin_label

        for col in ['Total Observations', 'Events', 'Non events']:
            woe_df_temp.at[previous_index, col] += woe_df_temp.at[min_diff_index, col]

        woe_df_temp = woe_df_temp.drop(index=min_diff_index).reset_index(drop=True)

        # Recalculate WoE/IV
        epsilon = 1e-10
        woe_df_temp['% of Non events'] = woe_df_temp['Non events'] / woe_df_temp['Non events'].sum()
        woe_df_temp['% of Events'] = woe_df_temp['Events'] / woe_df_temp['Events'].sum()
        woe_df_temp['% of Non events'] = woe_df_temp['% of Non events'].replace(0, epsilon)
        woe_df_temp['% of Events'] = woe_df_temp['% of Events'].replace(0, epsilon)
        woe_df_temp['WoE'] = np.log(woe_df_temp['% of Non events'] / woe_df_temp['% of Events'])
        woe_df_temp['IV'] = (
            (woe_df_temp['% of Non events'] - woe_df_temp['% of Events']) * woe_df_temp['WoE']
        )

    return woe_df_temp

def plot_woe_distribution(df, feature):
    df_plot = df[df['Feature'] == feature].copy()
    df_plot['bin'] = df_plot['bin'].astype(str)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(df_plot['bin'], df_plot['% of Non events'], label='% of Non events', color='blue')
    ax1.bar(df_plot['bin'], df_plot['% of Events'], bottom=df_plot['% of Non events'],
            label='% of Events', color='red')
    ax1.set_ylabel('Event/Non-event %')
    ax1.set_xlabel('Bins')
    ax1.set_xticklabels(df_plot['bin'], rotation=45, ha='right')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(df_plot['bin'], df_plot['WoE'], color='black', marker='o', label='WoE')
    ax2.set_ylabel('Weight of Evidence')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.title(f'WoE & Event Distribution - {feature}')
    fig.tight_layout()
    ax2.legend(loc='upper right')
    plt.show()

def compute_iv_optimize(df_1, target, min_bin_pct=0.02, plot=False):
    woe_df, iv_df = iv_woe(df_1, target)
    features = woe_df['Feature'].unique()

    final_df_list = []
    for col in features:
        merged = merge_small_bins(woe_df, col, min_bin_pct=min_bin_pct)
        merged = bin_merge(merged, col)
        final_df_list.append(merged)
        if plot:
            plot_woe_distribution(merged, col)

    final_woe_df = pd.concat(final_df_list, ignore_index=True)
    iv_df = final_woe_df.groupby('Feature')['IV'].sum().reset_index(name='IV')
    return final_woe_df, iv_df


def parse_interval(interval_str):
    """Parse a string like '(30000.0, 140000.0]' into a pd.Interval."""
    if not isinstance(interval_str, str):
        return np.nan
    match = re.match(r'[\[\(]([-\d\.]+),\s*([-\d\.]+)[\]\)]', interval_str)
    if match:
        left = float(match.group(1))
        right = float(match.group(2))
        closed = 'right' if interval_str.endswith(']') else 'left'
        return pd.Interval(left, right, closed=closed)
    return np.nan

def apply_woe_binning(df, woe_temp):
    df_transformed = df.copy()

    for feature in woe_temp['Feature'].unique():
        if feature not in df_transformed.columns:
            continue

        feature_woe = woe_temp[woe_temp['Feature'] == feature].copy()
        feature_woe['bin'] = feature_woe['bin'].astype(str)

        is_interval = feature_woe['bin'].str.contains(r'[\[\(].*,.*[\]\)]').any()

        if is_interval:
            intervals = feature_woe['bin'].apply(parse_interval)
            valid = intervals.notna()
            woe_map = dict(zip(intervals[valid], feature_woe.loc[valid, 'WoE']))
            cut_intervals = pd.IntervalIndex(intervals[valid])

            # Get WoE of first and last bin
            first_bin_woe = feature_woe.loc[valid, 'WoE'].iloc[0]
            last_bin_woe = feature_woe.loc[valid, 'WoE'].iloc[-1]
            min_edge = cut_intervals.left.min()
            max_edge = cut_intervals.right.max()

            def assign_woe(val):
                if pd.isna(val):
                    return np.nan
                elif val < min_edge:
                    return first_bin_woe
                elif val > max_edge:
                    return last_bin_woe
                else:
                    for interval, woe in woe_map.items():
                        if val in interval:
                            return woe
                    return np.nan

            df_transformed[feature] = df_transformed[feature].apply(assign_woe)

        else:
            woe_map = dict(zip(feature_woe['bin'], feature_woe['WoE']))
            df_transformed[feature] = df_transformed[feature].map(woe_map)

    return df_transformed

#--------------------------- WOE, IV: END ---------------------------#



def correlation_summary(df, method='pearson', figsize=(12, 10), annot=False, cmap='coolwarm'):
    """
    Displays a correlation heatmap and returns a table of correlation pairs.

    Parameters:
    - df: DataFrame with numeric features.
    - method: Correlation method - 'pearson', 'kendall', or 'spearman'.
    - figsize: Tuple for figure size.
    - annot: Whether to annotate heatmap.
    - cmap: Colormap for heatmap.

    Returns:
    - correlation_table: DataFrame with Feature 1, Feature 2, Correlation, Absolute Correlation
    """

    # Compute correlation matrix
    corr_matrix = df.corr(method=method)

    # Display heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, square=True)
    plt.title(f'{method.capitalize()} Correlation Heatmap')
    plt.show()

    # Create correlation pairs table
    corr_pairs = (
        corr_matrix.where(~pd.np.tril(pd.np.ones(corr_matrix.shape)).astype(bool))  # keep upper triangle
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
    corr_pairs['Absolute Correlation'] = corr_pairs['Correlation'].abs()
    corr_pairs = corr_pairs.sort_values(by='Absolute Correlation', ascending=False).reset_index(drop=True)

    return corr_pairs




def backward_elimination(X, y, significance_level=0.05, verbose=True):
    """
    Perform backward feature elimination based on p-values.

    Parameters:
    - X: DataFrame of features
    - y: Series or array-like target
    - significance_level: p-value threshold for retaining features
    - verbose: if True, prints elimination steps

    Returns:
    - selected_features: list of features retained
    - final_model: fitted statsmodels regression model
    """
    X_ = X.copy()
    X_ = sm.add_constant(X_)  # add intercept
    features = list(X_.columns)

    while True:
        model = sm.OLS(y, X_).fit()  # Use sm.Logit for logistic regression
        pvalues = model.pvalues
        max_pval = pvalues.drop('const').max()
        if max_pval > significance_level:
            excluded_feature = pvalues.drop('const').idxmax()
            if verbose:
                print(f"Dropping '{excluded_feature}' with p-value {max_pval:.4f}")
            X_.drop(columns=excluded_feature, inplace=True)
            features.remove(excluded_feature)
        else:
            break

    final_model = sm.OLS(y, X_).fit()
    return features, final_model

def Bad_rate_infp(df_train,df_test,target):
    obs_train = len(df_train)
    obs_test = len(df_test)
    bad_rate_train = round(df_train[target].value_counts(dropna = False, normalize = True)[1]*100,2)
    bad_rate_test = round(df_test[target].value_counts(dropna = False, normalize = True)[1]*100,2)
    
    result = {
        'Sample': ['Train','Test'],
        '#Observations': [obs_train,obs_test],
        '%Bad': [bad_rate_train,bad_rate_test]
    }
    
    summary = pd.DataFrame(result)
    
    return summary
    
    

def logit_model_report(X_train_selected, y_train, X_test_selected, y_test):
    # Add constant
    X_train_const = sm.add_constant(X_train_selected)
    X_test_const = sm.add_constant(X_test_selected)

    # Fit logistic regression
    model = sm.Logit(y_train, X_train_const)
    result = model.fit()

    # VIF calculation
    vif = pd.Series(
        [variance_inflation_factor(X_train_const.values, i) for i in range(X_train_const.shape[1])],
        index=X_train_const.columns,
        name='VIF'
    ).round(2)

    # Extract model summary with VIF
    summary_df = result.summary2().tables[1].copy()
    summary_df['VIF'] = vif[summary_df.index]

    # Print header and results
    print(result.summary().tables[0])
    print("=" * 100)
    print(summary_df.to_string())

    # Predict on test data
    y_pred_prob = result.predict(X_test_const)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Classification metrics
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")

    return result,summary_df



def plot_lorenz_curve_train_test(y_train, y_train_scores, y_test, y_test_scores):
    """
    Plot Lorenz curves for both Train and Test datasets on the same plot.
    
    Parameters:
        y_train: true labels (train)
        y_train_scores: predicted probs/scores (train)
        y_test: true labels (test)
        y_test_scores: predicted probs/scores (test)
    """

    def prepare_lorenz_data(y_true, y_scores):
        df = pd.DataFrame({'y_true': y_true, 'y_score': y_scores})
        df.sort_values(by='y_score', ascending=False, inplace=True)
        df['population_pct'] = np.arange(1, len(df) + 1) / len(df)
        total_bads = df['y_true'].sum()
        df['cum_bads'] = df['y_true'].cumsum()
        df['cum_bads_pct'] = df['cum_bads'] / total_bads
        return df['population_pct'], df['cum_bads_pct']

    train_x, train_y = prepare_lorenz_data(y_train, y_train_scores)
    test_x, test_y = prepare_lorenz_data(y_test, y_test_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(train_x, train_y, label='Train Lorenz Curve', color='lightblue', linewidth=2)
    plt.plot(test_x, test_y, label='Test Lorenz Curve', color='#FFB347', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Model')

    plt.xlabel('Cumulative % of Population')
    plt.ylabel('Cumulative % of Defaults')
    plt.title('Lorenz Curve: Train vs Test')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def evaluate_model_performance(y_train, y_train_pred_prob, y_train_pred_label,
                                y_test, y_test_pred_prob, y_test_pred_label):
    """
    Compute AUC, Gini, and Accuracy for Train and Test sets.
    
    Returns:
        pandas DataFrame with columns: Sample | AUC | Gini | Accuracy
    """

    def compute_metrics(y_true, y_prob, y_pred):
        auc = roc_auc_score(y_true, y_prob)
        gini = 2 * auc - 1
        acc = accuracy_score(y_true, y_pred)
        return auc, gini, acc

    train_auc, train_gini, train_acc = compute_metrics(y_train, y_train_pred_prob, y_train_pred_label)
    test_auc, test_gini, test_acc = compute_metrics(y_test, y_test_pred_prob, y_test_pred_label)

    result_df = pd.DataFrame({
        'Sample': ['Train', 'Test'],
        'AUC': [train_auc, test_auc],
        'Gini': [train_gini, test_gini],
        'Accuracy': [train_acc, test_acc]
    })

    return result_df


import pandas as pd
import numpy as np

def calculate_ks_table(y_true, y_pred, n_groups=10):
    """
    Calculate KS statistic table from actual outcomes and predicted probabilities.
    
    Args:
        y_true: Array of actual binary outcomes (0 or 1), shape (n_samples,)
        y_pred: Array of predicted probabilities (0 to 1), shape (n_samples,)
        n_groups: Number of groups to split the data into (default 10)
        
    Returns:
        DataFrame with KS statistics and performance metrics by group
    """
    # Convert to numpy arrays and validate
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Input validation
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} samples, y_pred has {len(y_pred)}")
    if not ((y_pred >= 0) & (y_pred <= 1)).all():
        raise ValueError("Predicted probabilities must be between 0 and 1")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Actual values must be binary (0 or 1)")
    
    # Create dataframe
    df = pd.DataFrame({
        'predicted': y_pred,
        'actual': y_true
    })
    
    # Rank by predicted probability (descending - higher risk first)
    df['rank'] = df['predicted'].rank(ascending=False, method='first')
    
    # Create groups (handle duplicates if they exist)
    df['group'] = pd.qcut(df['rank'], n_groups, duplicates='drop')
    
    # Calculate group statistics
    grouped = df.groupby('group')
    
    # Initialize results dataframe
    results = pd.DataFrame({
        'Predicted Group': range(n_groups),
        'Total Apps': grouped.size(),
        'Bad Counts': grouped['actual'].sum(),  # Assuming 1=bad
        'Good Counts': grouped.size() - grouped['actual'].sum()
    })
    
    # Add prediction statistics
    results['Minimum Predicted'] = grouped['predicted'].min()
    results['Maximum Predicted'] = grouped['predicted'].max()
    results['Average Predicted'] = grouped['predicted'].mean()
    results['Error Actual'] = grouped['actual'].mean()
    
    # Calculate percentages
    total_bad = df['actual'].sum()
    total_good = len(df) - total_bad
    
    results['Bad %'] = results['Bad Counts'] / total_bad * 100
    results['Good %'] = results['Good Counts'] / total_good * 100
    
    # Cumulative percentages
    results['Cumulative Bad %'] = results['Bad %'].cumsum()
    results['Cumulative Good %'] = results['Good %'].cumsum()
    
    # KS statistic
    results['KS'] = abs(results['Cumulative Good %'] - results['Cumulative Bad %'])
    
    # Rankings
    results['Predicted Rank'] = range(1, n_groups+1)
    results['Actual Rank'] = range(1, n_groups+1)
    
    # Format percentages
    pct_cols = ['Bad %', 'Good %', 'Cumulative Bad %', 'Cumulative Good %', 'KS']
    results[pct_cols] = results[pct_cols].round(2)
    
    results.reset_index(inplace= True)
    results.drop(columns='Predicted Group', inplace= True)
    
    results.rename(columns={'group':'Predicted Group'}, inplace= True)
    results['Rank Order Break'] = np.where(results['Predicted Rank'] != results['Actual Rank'],1,0)
    
    return results

def ks_info(y_train,y_pred_train,y_test,y_pred_test):
    ks_train = calculate_ks_table(y_train, y_pred_train)
    ks_test = calculate_ks_table(y_test, y_pred_test)
    
    ks_30_train = ks_train['KS'].iloc[2]
    ks_max_train = ks_train['KS'].max()
    
    ks_30_test = ks_test['KS'].iloc[2]
    ks_max_test = ks_test['KS'].max()
    
    result = {
        'Sample': ['Train','Test'],
        'Max KS': [ks_max_train,ks_max_test],
        'KS 30th Percentile': [ks_30_train,ks_30_test]
    }
    
    summary = pd.DataFrame(result)
    
    return summary

def adjust_credit_limit(limit, pd_prob, max_cap=200000, min_cap=10000):
    if pd_prob < 0.01:
        return min(limit * 1.5, max_cap)
    elif pd_prob < 0.03:
        return limit * 1.2
    elif pd_prob < 0.05:
        return limit
    elif pd_prob < 0.10:
        return limit * 0.8
    else:
        return max(limit * 0.5, min_cap)
    
def get_limit_action(pd_prob):
    if pd_prob < 0.01:
        return "Increase Aggressively"
    elif pd_prob < 0.03:
        return "Increase Moderately"
    elif pd_prob < 0.05:
        return "Hold"
    elif pd_prob < 0.10:
        return "Decrease Slightly"
    else:
        return "Review/Decrease Significantly"

