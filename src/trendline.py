import polars as pl
import numpy as np

### Trendline checking algorithm using gradiendt descent ###

def check_trendline(support: bool, pivot: int, slope: float, y: np.array):

    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept

    diffs = line_vals - y

    # Check to see if the line is valid, return -1 if it's not valid
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < 1e-5:
        return -1.0
    
    # Squared sum of diffs between data and line
    err = (diffs ** 2).sum()
    return err

def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.array):

    # Amount to change slope by, multiplied by opt_step
    slope_unit = (y.max() - y.min()) / len(y)

    # Optimization variables
    opt_step = 1.0
    min_step = 0.0001
    current_step = opt_step

    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_error = check_trendline(support, pivot, init_slope, y)
    assert(best_error >= 0.0) # Shouldn't ever fail with initial slope

    get_derivative = True
    derivative = None
    while current_step > min_step:
        if get_derivative:
            #Numerical differentiation, increase slope by very small amount
            # To see if error increases or decreases
            # That gives us the direction to change the slope
            slope_change = best_slope + slope_unit * min_step
            test_error = check_trendline(support, pivot, slope_change, y)
            derivative = best_error - test_error

            # If increasing by a small amount fails, try decreasing by a small amount
            if test_error < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_error = check_trendline(support, pivot, slope_change, y)
                derivative = best_error - test_error

            if test_error < 0.0: # Derivative failed, pass
                raise Exception("Derivative failed - check the data. ")
            
            get_derivative = False
        
        if get_derivative > 0.0: # Increasing slope increased error
            test_slope = best_slope - slope_unit * current_step
        else: # Increasing slope decreased error
            test_slope = best_slope + slope_unit * current_step

        test_error = check_trendline(support, pivot, test_slope, y)
        if test_error < 0 or test_error >= best_error:
            # Slope failed, didn't reduce error
            current_step *= 0.5
        else: # Test slope reduced error
            best_error = test_error
            best_slope = test_slope
            get_derivative = True

    # Optimization done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y[pivot])

def fit_trendlines_single(data: np.array):
    # Find the lione of best fit(least squared)
    # coefs[0] = slope, coefs[1] = intercept
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax()
    lower_pivot = (data - line_points).argmin()

    # Optimize the slope for both trendlines
    supports_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return(supports_coefs, resist_coefs)

def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array): 
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax()
    lower_pivot = (low - line_points).argmin()

    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)




