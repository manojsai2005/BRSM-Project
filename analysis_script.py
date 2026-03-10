import os
import warnings

warnings.filterwarnings('ignore')

from data_cleaning import load_and_clean
from statistical_tests import run_statistical_tests
from visualizations import run_visualizations


def run_analysis(data_dir):
    # Step 1: Data Cleaning
    clean_df, col_names = load_and_clean(data_dir)
    if clean_df is None:
        return

    results_dir = os.path.join(data_dir, "analysis_results")
    os.makedirs(results_dir, exist_ok=True)

    # Step 2: Statistical Tests
    run_statistical_tests(clean_df, col_names, results_dir)

    # Step 3: Visualizations
    run_visualizations(clean_df, col_names, results_dir)


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    run_analysis(data_dir)
