#!/usr/bin/env python
# coding: utf-8

"""
Burst detection implementation using Kleinberg's algorithm.
This is used to detect bursts in time series data for keywords, named entities, and taxonomy elements.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional
import logging
from datetime import datetime, timedelta

def kleinberg_burst_detection(
    time_series: pd.DataFrame,
    column: str = 'count',
    date_column: str = 'date',
    s: float = 2.0,
    gamma: float = 1.0,
    smoothing: float = 0.1
) -> pd.DataFrame:
    """
    Implement Kleinberg's burst detection algorithm to find bursts in time series data.
    
    Args:
        time_series: DataFrame with time series data
        column: Column containing count data
        date_column: Column containing date data
        s: Parameter controlling burst intensity
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