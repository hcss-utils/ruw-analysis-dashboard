# NetworkX Dependency Fix

## Overview

This fix addresses the robust handling of optional dependencies, specifically for NetworkX, scikit-learn, and SciPy. These libraries are used for advanced visualizations and calculations but might not be available in all deployment environments. The application now gracefully degrades functionality rather than crashing when these dependencies are unavailable.

## Implemented Changes

### String vs Numeric Period Handling

1. Fixed type errors in the predictive visualizations when dealing with string periods:
   ```python
   # Check if periods are strings or numbers before arithmetic operations
   if isinstance(all_periods[-1], str) or isinstance(next_periods[-1], str):
       # Handle string periods appropriately
       # ...
   else:
       # Handle numeric periods
       # ...
   ```

2. Added special handling for confidence interval visualization with string periods:
   ```python
   # Handle string periods differently for concatenation
   if isinstance(next_periods[0], str):
       # For string periods, we need to create a full list for x values
       x_values_for_fill = []
       x_values_for_fill.extend(next_periods)  # Add periods in order
       x_values_for_fill.extend(next_periods[::-1])  # Add periods in reverse order
       
       # Use the properly constructed x values
       # ...
   ```

### NetworkX Handling

1. Added try/except blocks for NetworkX imports with a `NETWORKX_AVAILABLE` flag:
   ```python
   try:
       import networkx as nx
       NETWORKX_AVAILABLE = True
   except ImportError:
       NETWORKX_AVAILABLE = False
   ```

2. Created fallback visualization functions for network graphs:
   - `create_matrix_visualization_fallback` - Creates a matrix heatmap visualization instead of a network graph

3. Added conditional checks before using NetworkX functionality:
   ```python
   if not NETWORKX_AVAILABLE:
       # Use fallback visualization
       return create_matrix_visualization_fallback(...)
   ```

4. Removed duplicate implementations:
   - Consolidated network visualization code to `visualizations/co_occurrence.py`
   - Updated imports in `visualizations/bursts.py` and other modules to use the single implementation

### scikit-learn Handling

1. Added try/except blocks for scikit-learn imports with a `SKLEARN_AVAILABLE` flag:
   ```python
   try:
       from sklearn.linear_model import LinearRegression
       SKLEARN_AVAILABLE = True
   except ImportError:
       SKLEARN_AVAILABLE = False
   ```

2. Created fallback implementations using NumPy:
   ```python
   if SKLEARN_AVAILABLE:
       # Use sklearn LinearRegression
       model = LinearRegression()
       model.fit(X, y)
       # Make predictions
       predictions = model.predict(future_X)
   else:
       # Fallback to manual implementation with NumPy
       # Calculate slope and intercept
       n = len(X)
       x_mean = np.mean(X)
       y_mean = np.mean(y)
       # Calculate slope
       numerator = np.sum((X.flatten() - x_mean) * (y - y_mean))
       denominator = np.sum((X.flatten() - x_mean) ** 2)
       slope = numerator / denominator if denominator != 0 else 0
       # Calculate intercept
       intercept = y_mean - slope * x_mean
       # Predict
       predictions = slope * future_X.flatten() + intercept
   ```

### SciPy Handling

1. Added try/except blocks for SciPy imports with a `SCIPY_AVAILABLE` flag:
   ```python
   try:
       from scipy.interpolate import interp1d
       from scipy.stats import pearsonr
       import scipy.stats as stats
       SCIPY_AVAILABLE = True
   except ImportError:
       SCIPY_AVAILABLE = False
   ```

2. Created fallback calculations for statistical functions:
   ```python
   if SCIPY_AVAILABLE:
       t_value = stats.t.ppf((1 + confidence_level) / 2, len(x_values) - 2)
   else:
       # Simple approximation without scipy
       if confidence_level >= 0.99:
           t_value = 2.58  # ~99% confidence
       elif confidence_level >= 0.95:
           t_value = 1.96  # ~95% confidence
       elif confidence_level >= 0.90:
           t_value = 1.65  # ~90% confidence
       else:
           t_value = 1.28  # ~80% confidence
   ```

## Files Modified

1. `visualizations/bursts.py`
   - Added import flags and fallbacks for NetworkX, scikit-learn, and SciPy
   - Removed duplicate implementation of `create_co_occurrence_network`
   - Added imports from `visualizations.co_occurrence`

2. `visualizations/co_occurrence.py`
   - Added NetworkX fallback mechanism using matrix visualization
   - Ensured consistent fallback function naming

3. `tabs/burstiness.py`
   - Updated imports to use the correct `create_co_occurrence_network` implementation

4. `utils/burst_detection.py`
   - Added proper handling of SciPy as an optional dependency
   - Implemented fallbacks for Poisson distribution calculations
   - Added approximations for normal distribution functions
   - Used Stirling's approximation for factorial calculations when needed

## Testing

The application has been tested to ensure it functions properly with and without the optional dependencies. The fallback mechanisms provide simplified but still informative visualizations when the advanced libraries are unavailable.