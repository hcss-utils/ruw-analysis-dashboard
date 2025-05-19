#!/usr/bin/env python
# coding: utf-8

"""
Enhanced burst detection module implementing Kleinberg's algorithm with advanced features.
This module provides multiple implementations of burst detection algorithms:
1. Basic Kleinberg burst detection (2-state model)
2. Multi-state Kleinberg burst detection (n-state model)
3. Co-occurrence burst detection
4. Statistical validation and significance testing for bursts
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional, Any, Set
import logging
from datetime import datetime, timedelta
from itertools import combinations
from collections import defaultdict

# Try to import scipy components, but handle case if not available
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

###########################################
# Basic Kleinberg Burst Detection (2-state)
###########################################

def kleinberg_burst_detection(
    time_series: pd.DataFrame,
    column: str = 'count',
    date_column: str = 'date',
    s: float = 2.0,
    gamma: float = 1.0,
    smoothing: float = 0.1
) -> pd.DataFrame:
    """
    Basic implementation of Kleinberg's burst detection algorithm with 2 states.
    
    Args:
        time_series: DataFrame with time series data
        column: Column containing count data
        date_column: Column containing date data
        s: Parameter controlling burst intensity (higher = more stringent)
        gamma: Parameter controlling burst state transition cost
        smoothing: Smoothing parameter for counts
    
    Returns:
        pd.DataFrame: DataFrame with original data plus burst scores
    """
    if time_series.empty:
        logging.warning("Empty time series provided for burst detection")
        return pd.DataFrame()
    
    # Ensure data is sorted by date
    if date_column in time_series.columns:
        time_series = time_series.sort_values(date_column)
    
    # Add smoothing to avoid zeros (which cause problems with log calculations)
    time_series['smoothed_count'] = time_series[column] + smoothing
    
    # Calculate base rate (average rate across the entire series)
    total_count = time_series['smoothed_count'].sum()
    n = len(time_series)
    base_rate = total_count / n
    
    # Calculate expected probabilities
    time_series['prob'] = time_series['smoothed_count'] / total_count
    
    # Calculate bursting threshold (s times base rate)
    burst_threshold = s * base_rate
    
    # Identify points that exceed the threshold
    time_series['burst'] = (time_series['smoothed_count'] > burst_threshold).astype(int)
    
    # Calculate burst score (how many times above threshold)
    time_series['burst_score'] = time_series['smoothed_count'] / burst_threshold
    time_series.loc[time_series['burst_score'] < 1, 'burst_score'] = 0  # Only keep scores >= 1

    # Apply a rolling window to identify sustained bursts
    if n > 1:
        window_size = max(2, int(n * 0.1))  # Use 10% of the series length or at least 2 points
        time_series['burst_intensity'] = time_series['burst_score'].rolling(window=window_size, min_periods=1).mean()
    else:
        time_series['burst_intensity'] = time_series['burst_score']
    
    # Scale the burst intensity to 0-100 range if there are any bursts
    max_intensity = time_series['burst_intensity'].max()
    if max_intensity > 0:
        time_series['burst_intensity'] = (time_series['burst_intensity'] / max_intensity) * 100
    
    return time_series

###########################################
# Multi-state Kleinberg Burst Detection
###########################################

def kleinberg_multi_state_burst_detection(
    time_series: pd.DataFrame,
    column: str = 'count',
    date_column: str = 'date',
    num_states: int = 3,
    s_values: Optional[List[float]] = None,
    gamma: float = 1.0,
    smoothing: float = 0.1
) -> pd.DataFrame:
    """
    Multi-state implementation of Kleinberg's burst detection algorithm.
    Supports n states with different levels of burst intensity.
    
    Args:
        time_series: DataFrame with time series data
        column: Column containing count data
        date_column: Column containing date data
        num_states: Number of burst states (including non-burst state)
        s_values: List of state transition parameters for each state (default=None, 
                  will be calculated as [1.0, 2.0, 4.0, ...] if not provided)
        gamma: Parameter controlling burst state transition cost
        smoothing: Smoothing parameter for counts
    
    Returns:
        pd.DataFrame: DataFrame with original data plus multi-state burst scores
    """
    if time_series.empty:
        logging.warning("Empty time series provided for multi-state burst detection")
        return pd.DataFrame()
    
    # Ensure data is sorted by date
    if date_column in time_series.columns:
        time_series = time_series.sort_values(date_column)
    
    # Create s_values if not provided
    if s_values is None:
        s_values = [1.0] + [2.0 * (i+1) for i in range(num_states-1)]
    elif len(s_values) != num_states:
        logging.warning(f"s_values length ({len(s_values)}) does not match num_states ({num_states})")
        # Ensure correct length by truncating or extending
        if len(s_values) < num_states:
            s_values = s_values + [s_values[-1] * 1.5] * (num_states - len(s_values))
        else:
            s_values = s_values[:num_states]
    
    # Add smoothing to avoid zeros
    time_series['smoothed_count'] = time_series[column] + smoothing
    
    # Calculate base rate (average rate across the entire series)
    total_count = time_series['smoothed_count'].sum()
    n = len(time_series)
    base_rate = total_count / n
    
    # Initialize HMM parameters
    states = list(range(num_states))  # 0 is non-burst state, 1..n are burst states
    
    # Calculate cost matrix for Viterbi algorithm
    # Cost of being in state k at time t
    cost = np.zeros((n, num_states))
    for t in range(n):
        for k in range(num_states):
            # Cost is -log(p(x_t|q_t=k))
            # For state 0, p is from Poisson with rate base_rate
            # For state k, p is from Poisson with rate s_values[k] * base_rate
            rate = base_rate if k == 0 else s_values[k] * base_rate
            count = time_series.iloc[t]['smoothed_count']
            # Using log probability for numerical stability
            if SCIPY_AVAILABLE:
                cost[t, k] = -stats.poisson.logpmf(count, rate)
            else:
                # Simple approximation of Poisson log PMF without scipy
                # log(e^(-λ) * λ^k / k!) = -λ + k*log(λ) - log(k!)
                # For large counts, use Stirling's approximation for log(k!)
                count_int = int(count)
                if count_int > 20:
                    # Stirling's approximation: log(n!) ≈ n*log(n) - n
                    log_factorial = count_int * np.log(count_int) - count_int
                else:
                    try:
                        log_factorial = np.log(np.math.factorial(count_int))
                    except OverflowError:
                        # Fallback to Stirling's approximation if factorial overflows
                        log_factorial = count_int * np.log(count_int) - count_int
                        
                logpmf = -rate + count * np.log(rate) - log_factorial
                cost[t, k] = -logpmf
    
    # State transition cost
    # Higher gamma = higher cost to transition between states
    trans_cost = np.zeros((num_states, num_states))
    for i in range(num_states):
        for j in range(num_states):
            if i != j:
                trans_cost[i, j] = gamma * abs(i - j)
    
    # Viterbi algorithm to find optimal state sequence
    # Initialize
    V = np.zeros((n, num_states))
    backpointer = np.zeros((n, num_states), dtype=int)
    
    # Base case (t=0)
    for k in range(num_states):
        V[0, k] = cost[0, k]
        backpointer[0, k] = 0
    
    # Recursive case
    for t in range(1, n):
        for k in range(num_states):
            # Find the minimum cost path to state k at time t
            min_cost = float('inf')
            min_state = 0
            for prev_k in range(num_states):
                temp_cost = V[t-1, prev_k] + trans_cost[prev_k, k] + cost[t, k]
                if temp_cost < min_cost:
                    min_cost = temp_cost
                    min_state = prev_k
            V[t, k] = min_cost
            backpointer[t, k] = min_state
    
    # Find optimal end state
    optimal_end_state = np.argmin(V[n-1, :])
    
    # Backtrace to find optimal state sequence
    optimal_states = np.zeros(n, dtype=int)
    optimal_states[n-1] = optimal_end_state
    for t in range(n-2, -1, -1):
        optimal_states[t] = backpointer[t+1, optimal_states[t+1]]
    
    # Add state and burst intensity to original data
    time_series['burst_state'] = optimal_states
    time_series['burst_intensity'] = np.zeros(n)
    
    for t in range(n):
        state = optimal_states[t]
        if state > 0:  # If in a burst state
            # Intensity is proportional to the burst state
            # Scale from 0-100 based on the state
            time_series.loc[time_series.index[t], 'burst_intensity'] = (state / (num_states-1)) * 100
    
    # Apply a rolling window to smooth burst intensity
    if n > 1:
        window_size = max(2, int(n * 0.1))  # Use 10% of the series length or at least 2 points
        time_series['burst_intensity'] = time_series['burst_intensity'].rolling(window=window_size, min_periods=1).mean()
    
    return time_series

###########################################
# Burst Co-occurrence Detection
###########################################

def detect_burst_co_occurrences(
    burst_data: Dict[str, Dict[str, pd.DataFrame]],
    min_burst_intensity: float = 20.0,
    min_periods: int = 2
) -> Dict[str, Dict[str, float]]:
    """
    Detect co-occurring bursts across different elements and data types.
    
    Args:
        burst_data: Dictionary mapping data types to dictionaries of element burst DataFrames
        min_burst_intensity: Minimum burst intensity to consider as a significant burst
        min_periods: Minimum number of periods where bursts must co-occur
        
    Returns:
        Dict mapping element pairs to co-occurrence strength
    """
    # Extract all elements with bursts above threshold
    element_bursts = {}
    
    for data_type, elements in burst_data.items():
        for element_name, df in elements.items():
            if df.empty or 'burst_intensity' not in df.columns or 'period' not in df.columns:
                continue
                
            # Create a qualified name with data type prefix
            qualified_name = f"{data_type}:{element_name}"
            
            # Record periods with significant bursts
            burst_periods = set()
            for _, row in df.iterrows():
                if row['burst_intensity'] >= min_burst_intensity:
                    burst_periods.add(row['period'])
                    
            if burst_periods:
                element_bursts[qualified_name] = burst_periods
    
    # Find co-occurrences
    co_occurrences = {}
    
    for elem1, elem2 in combinations(element_bursts.keys(), 2):
        # Find common burst periods
        common_periods = element_bursts[elem1].intersection(element_bursts[elem2])
        
        if len(common_periods) >= min_periods:
            # Calculate Jaccard similarity as co-occurrence strength
            union_periods = element_bursts[elem1].union(element_bursts[elem2])
            strength = len(common_periods) / len(union_periods)
            
            # Store co-occurrence with its strength
            pair_key = (elem1, elem2)
            co_occurrences[pair_key] = {
                'strength': strength,
                'common_periods': list(common_periods),
                'num_common_periods': len(common_periods)
            }
    
    return co_occurrences

def generate_co_occurrence_network(
    co_occurrences: Dict[Tuple[str, str], Dict],
    min_strength: float = 0.3
) -> Dict[str, Any]:
    """
    Generate a network representation of co-occurring burst elements.
    
    Args:
        co_occurrences: Dictionary mapping element pairs to co-occurrence data
        min_strength: Minimum co-occurrence strength to include in the network
        
    Returns:
        Dict with nodes and edges for network visualization
    """
    # Initialize network components
    nodes = set()
    edges = []
    
    # Process co-occurrences
    for (elem1, elem2), data in co_occurrences.items():
        if data['strength'] >= min_strength:
            nodes.add(elem1)
            nodes.add(elem2)
            
            # Create edge with attributes
            edges.append({
                'source': elem1,
                'target': elem2,
                'weight': data['strength'],
                'periods': data['common_periods'],
                'count': data['num_common_periods']
            })
    
    # Convert nodes to list with data type information
    node_list = []
    for node in nodes:
        data_type, element = node.split(':', 1)
        node_list.append({
            'id': node,
            'data_type': data_type,
            'label': element
        })
    
    return {
        'nodes': node_list,
        'edges': edges
    }

###########################################
# Statistical Validation of Bursts
###########################################

def validate_bursts_statistically(
    time_series: pd.DataFrame,
    column: str = 'count',
    date_column: str = 'date',
    confidence_level: float = 0.95,
    window_size: int = 5
) -> pd.DataFrame:
    """
    Statistically validate bursts using confidence intervals and Z-scores.
    
    Args:
        time_series: DataFrame with time series data
        column: Column containing count data
        date_column: Column containing date data
        confidence_level: Confidence level for statistical validation (default=0.95)
        window_size: Size of the moving window for local baseline calculation
        
    Returns:
        pd.DataFrame: DataFrame with burst validation metrics
    """
    if time_series.empty:
        logging.warning("Empty time series provided for statistical validation")
        return pd.DataFrame()
    
    # Ensure data is sorted by date
    if date_column in time_series.columns:
        time_series = time_series.sort_values(date_column)
    
    # Create a copy to avoid modifying the original
    result_df = time_series.copy()
    n = len(result_df)
    
    # Calculate global statistics
    global_mean = result_df[column].mean()
    global_std = result_df[column].std()
    
    # Add global Z-scores
    if global_std > 0:
        result_df['global_z_score'] = (result_df[column] - global_mean) / global_std
    else:
        result_df['global_z_score'] = 0
    
    # Calculate confidence interval for global baseline
    if SCIPY_AVAILABLE:
        z_critical = stats.norm.ppf((1 + confidence_level) / 2)
    else:
        # Simple approximation of z-values without scipy
        if confidence_level >= 0.99:
            z_critical = 2.58  # ~99% confidence
        elif confidence_level >= 0.95:
            z_critical = 1.96  # ~95% confidence
        elif confidence_level >= 0.90:
            z_critical = 1.65  # ~90% confidence
        else:
            z_critical = 1.28  # ~80% confidence
    
    margin_of_error = z_critical * (global_std / np.sqrt(n))
    
    result_df['global_ci_lower'] = global_mean - margin_of_error
    result_df['global_ci_upper'] = global_mean + margin_of_error
    result_df['global_exceeds_ci'] = result_df[column] > result_df['global_ci_upper']
    
    # Calculate local statistics using rolling window
    if n > window_size:
        # Use centered window when possible
        half_window = window_size // 2
        
        local_means = []
        local_stds = []
        local_z_scores = []
        local_ci_lowers = []
        local_ci_uppers = []
        local_exceeds_ci = []
        
        for i in range(n):
            # Define window boundaries
            start_idx = max(0, i - half_window)
            end_idx = min(n, i + half_window + 1)
            
            # Exclude the current point from local stats
            window_values = list(result_df[column].iloc[start_idx:i])
            window_values.extend(list(result_df[column].iloc[i+1:end_idx]))
            
            if not window_values:
                # Edge case: if we can't get any window values, use global stats
                local_means.append(global_mean)
                local_stds.append(global_std)
                local_z_scores.append(result_df['global_z_score'].iloc[i])
                local_ci_lowers.append(result_df['global_ci_lower'].iloc[i])
                local_ci_uppers.append(result_df['global_ci_upper'].iloc[i])
                local_exceeds_ci.append(result_df['global_exceeds_ci'].iloc[i])
                continue
            
            local_mean = np.mean(window_values)
            local_std = np.std(window_values)
            
            # Calculate Z-score
            if local_std > 0:
                local_z = (result_df[column].iloc[i] - local_mean) / local_std
            else:
                local_z = 0
                
            # Calculate confidence interval
            local_margin = z_critical * (local_std / np.sqrt(len(window_values)))
            local_ci_lower = local_mean - local_margin
            local_ci_upper = local_mean + local_margin
            
            # Store values
            local_means.append(local_mean)
            local_stds.append(local_std)
            local_z_scores.append(local_z)
            local_ci_lowers.append(local_ci_lower)
            local_ci_uppers.append(local_ci_upper)
            local_exceeds_ci.append(result_df[column].iloc[i] > local_ci_upper)
        
        # Add local statistics to result
        result_df['local_mean'] = local_means
        result_df['local_std'] = local_stds
        result_df['local_z_score'] = local_z_scores
        result_df['local_ci_lower'] = local_ci_lowers
        result_df['local_ci_upper'] = local_ci_uppers
        result_df['local_exceeds_ci'] = local_exceeds_ci
        
        # Define statistical burst intensity
        # Scale the local Z-score to a 0-100 score
        result_df['stat_burst_intensity'] = np.clip(result_df['local_z_score'], 0, None)
        max_z = result_df['stat_burst_intensity'].max()
        if max_z > 0:
            result_df['stat_burst_intensity'] = (result_df['stat_burst_intensity'] / max_z) * 100
    else:
        # For very short series, just use global statistics
        result_df['local_mean'] = global_mean
        result_df['local_std'] = global_std
        result_df['local_z_score'] = result_df['global_z_score']
        result_df['local_ci_lower'] = result_df['global_ci_lower']
        result_df['local_ci_upper'] = result_df['global_ci_upper']
        result_df['local_exceeds_ci'] = result_df['global_exceeds_ci']
        result_df['stat_burst_intensity'] = np.clip(result_df['global_z_score'], 0, None)
        max_z = result_df['stat_burst_intensity'].max()
        if max_z > 0:
            result_df['stat_burst_intensity'] = (result_df['stat_burst_intensity'] / max_z) * 100
    
    # Final burst probability based on confidence level and Z-score
    # Probability of observing a value at least this extreme
    if SCIPY_AVAILABLE:
        result_df['burst_probability'] = 1 - stats.norm.cdf(result_df['local_z_score'])
    else:
        # Simple approximation of normal CDF without scipy
        # Using an approximation of the error function
        def approx_norm_cdf(z):
            neg = z < 0
            z = abs(z)
            # Approximation of error function
            a1 = 0.254829592
            a2 = -0.284496736
            a3 = 1.421413741
            a4 = -1.453152027
            a5 = 1.061405429
            p = 0.3275911
            t = 1.0 / (1.0 + p * z)
            erf = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * np.exp(-z * z)
            cdf = 0.5 * (1 + erf)
            return 1 - cdf if neg else cdf
        
        result_df['burst_probability'] = result_df['local_z_score'].apply(lambda z: 1 - approx_norm_cdf(z))
    
    # Calculate statistical significance (p-value < (1 - confidence_level))
    alpha = 1 - confidence_level
    result_df['burst_significant'] = result_df['burst_probability'] < alpha
    
    return result_df

###########################################
# Helper Functions
###########################################

def prepare_time_periods(
    period: str,
    n_periods: int = 10
) -> Tuple[List[datetime], List[str]]:
    """
    Prepare time period boundaries for burst detection.
    
    Args:
        period: Period type ('week', 'month', or 'quarter')
        n_periods: Number of periods to include
    
    Returns:
        Tuple containing list of period boundaries and list of period labels
    """
    today = datetime.now()
    period_boundaries = []
    period_labels = []
    
    if period == 'week':
        for i in range(n_periods):
            end_date = today - timedelta(days=i * 7)
            start_date = end_date - timedelta(days=7)
            period_boundaries.append((start_date, end_date))
            period_labels.append(f"Week {i+1}")
            
    elif period == 'month':
        for i in range(n_periods):
            # Simple approximation using 30 days for a month
            end_date = today - timedelta(days=i * 30)
            start_date = end_date - timedelta(days=30)
            period_boundaries.append((start_date, end_date))
            period_labels.append(end_date.strftime('%b %Y'))
            
    elif period == 'quarter':
        for i in range(n_periods):
            # Simple approximation using 90 days for a quarter
            end_date = today - timedelta(days=i * 90)
            start_date = end_date - timedelta(days=90)
            period_boundaries.append((start_date, end_date))
            
            # Determine quarter label
            quarter = ((end_date.month - 1) // 3) + 1
            period_labels.append(f"Q{quarter} {end_date.year}")
    
    # Reverse the lists so they're in chronological order
    return period_boundaries[::-1], period_labels[::-1]


def normalize_burst_scores(burst_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Normalize burst scores across all items to make them comparable.
    
    Args:
        burst_data: Dictionary mapping item names to burst DataFrames
    
    Returns:
        Dict with normalized burst data
    """
    # Collect all burst intensities to find global max
    all_intensities = []
    for item_df in burst_data.values():
        if not item_df.empty and 'burst_intensity' in item_df.columns:
            all_intensities.extend(item_df['burst_intensity'].tolist())
    
    if not all_intensities:
        return burst_data  # No data to normalize
    
    # Find global max
    global_max = max(all_intensities) if all_intensities else 1.0
    global_max = max(global_max, 1.0)  # Avoid division by zero
    
    # Normalize each item's burst intensity
    normalized_data = {}
    for item, item_df in burst_data.items():
        if not item_df.empty and 'burst_intensity' in item_df.columns:
            item_df_copy = item_df.copy()
            item_df_copy['normalized_intensity'] = (item_df['burst_intensity'] / global_max) * 100
            normalized_data[item] = item_df_copy
        else:
            normalized_data[item] = item_df
    
    return normalized_data


def find_cascade_patterns(
    burst_data: Dict[str, Dict[str, pd.DataFrame]],
    window_size: int = 2
) -> List[Dict[str, Any]]:
    """
    Identify causal cascade patterns where bursts in one element lead to bursts in others.
    
    Args:
        burst_data: Dictionary mapping data types to dictionaries of element burst DataFrames
        window_size: Number of periods to look ahead for potential cascade effects
        
    Returns:
        List of cascade patterns with leader and follower elements
    """
    cascade_patterns = []
    
    # Collect all unique periods
    all_periods = set()
    for data_type, elements in burst_data.items():
        for element, df in elements.items():
            if not df.empty and 'period' in df.columns:
                all_periods.update(df['period'].unique())
    
    # Sort periods chronologically
    all_periods = sorted(all_periods)
    if len(all_periods) < 2:
        return cascade_patterns  # Not enough periods for cascade detection
    
    # Create a map of elements with significant bursts in each period
    period_bursts = defaultdict(list)
    
    for data_type, elements in burst_data.items():
        for element, df in elements.items():
            if df.empty or 'burst_intensity' not in df.columns or 'period' not in df.columns:
                continue
                
            # Create a qualified name with data type prefix
            qualified_name = f"{data_type}:{element}"
            
            # Record periods with significant bursts
            for _, row in df.iterrows():
                if row['burst_intensity'] >= 50:  # Consider bursts with intensity >= 50 as significant
                    period = row['period']
                    period_bursts[period].append({
                        'element': qualified_name,
                        'intensity': row['burst_intensity']
                    })
    
    # Look for cascade patterns across periods
    for i, period in enumerate(all_periods[:-1]):  # Skip the last period as it can't initiate a cascade
        # Get elements with significant bursts in this period
        leader_elements = period_bursts[period]
        
        # Look ahead up to window_size periods
        for j in range(1, min(window_size + 1, len(all_periods) - i)):
            follower_period = all_periods[i + j]
            follower_elements = period_bursts[follower_period]
            
            # Skip if no bursts in either period
            if not leader_elements or not follower_elements:
                continue
            
            # Check for potential cascade relationships
            for leader in leader_elements:
                for follower in follower_elements:
                    # Skip self-cascades
                    if leader['element'] == follower['element']:
                        continue
                    
                    # Record potential cascade pattern
                    cascade_patterns.append({
                        'leader': leader['element'],
                        'leader_intensity': leader['intensity'],
                        'leader_period': period,
                        'follower': follower['element'],
                        'follower_intensity': follower['intensity'],
                        'follower_period': follower_period,
                        'lag': j,  # Number of periods between leader and follower
                        'combined_strength': (leader['intensity'] + follower['intensity']) / 2
                    })
    
    # Sort by combined strength
    cascade_patterns.sort(key=lambda x: x['combined_strength'], reverse=True)
    
    return cascade_patterns


def analyze_concurrent_bursts(
    burst_data: Dict[str, Dict[str, pd.DataFrame]],
    threshold: float = 50.0
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze bursts occurring concurrently across multiple data types.
    
    Args:
        burst_data: Dictionary mapping data types to dictionaries of element burst DataFrames
        threshold: Minimum burst intensity to consider as significant
        
    Returns:
        Dict mapping periods to lists of significant concurrent bursts
    """
    # Initialize results by period
    concurrent_bursts = defaultdict(list)
    
    # Process all data types and elements
    for data_type, elements in burst_data.items():
        for element, df in elements.items():
            if df.empty or 'burst_intensity' not in df.columns or 'period' not in df.columns:
                continue
            
            # Find significant bursts
            for _, row in df.iterrows():
                if row['burst_intensity'] >= threshold:
                    period = row['period']
                    concurrent_bursts[period].append({
                        'data_type': data_type,
                        'element': element,
                        'intensity': row['burst_intensity'],
                        'count': row['count'] if 'count' in row else None
                    })
    
    # Sort bursts within each period by intensity
    for period in concurrent_bursts:
        concurrent_bursts[period].sort(key=lambda x: x['intensity'], reverse=True)
    
    return dict(concurrent_bursts)