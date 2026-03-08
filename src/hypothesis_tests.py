"""
hypothesis_tests.py

Functions for performing hypothesis tests on datasets.
"""

from scipy import stats
import pandas as pd


def t_test_one_sample(data: pd.Series, popmean: float):
    """Perform a one-sample t-test."""
    stat, p_value = stats.ttest_1samp(data.dropna(), popmean)
    return {"statistic": stat, "p_value": p_value}


def t_test_two_sample(group1: pd.Series, group2: pd.Series):
    """Perform an independent two-sample t-test."""
    stat, p_value = stats.ttest_ind(group1.dropna(), group2.dropna())
    return {"statistic": stat, "p_value": p_value}


def chi_square_test(observed, expected=None):
    """Perform a chi-square goodness-of-fit or independence test."""
    if expected is not None:
        stat, p_value = stats.chisquare(observed, f_exp=expected)
    else:
        stat, p_value = stats.chisquare(observed)
    return {"statistic": stat, "p_value": p_value}


def anova_one_way(*groups):
    """Perform a one-way ANOVA test across multiple groups."""
    stat, p_value = stats.f_oneway(*groups)
    return {"statistic": stat, "p_value": p_value}
