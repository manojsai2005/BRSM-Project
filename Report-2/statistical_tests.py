import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats



def run_mannwhitney_safe(data1, data2, tail=1):
    """Non-parametric alternative to t-test for outliers/non-normal data."""
    if len(data1.dropna()) > 1 and len(data2.dropna()) > 1:
        alt = 'greater' if tail == 1 else 'two-sided'
        u, p = stats.mannwhitneyu(data1, data2, alternative=alt, nan_policy='omit')
        return u, p
    return np.nan, np.nan





def run_multiple_regression(clean_df, corr_col, conf_col, sf):
    required_cols = ['Gender', conf_col, corr_col]
    if all(col in clean_df.columns for col in required_cols):
        reg_df = clean_df.dropna(subset=required_cols).copy()
        
        reg_sub = reg_df.groupby(['participant', 'Group', 'Gender']).agg({
            corr_col: 'mean',
            conf_col: 'mean'
        }).reset_index()
        
        reg_sub = reg_sub.dropna()
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from patsy import dmatrices
        
        reg_formula = f"Q('{corr_col}') ~ C(Group) + C(Gender) + Q('{conf_col}')"
        try:
            y, X = dmatrices(reg_formula, reg_sub, return_type='dataframe')
            reg_model = sm.OLS(y, X).fit()
            
            sf.write("\n--- MULTIPLE LINEAR REGRESSION (Predicting Accuracy) ---\n")
            sf.write(reg_model.summary().as_text())
            sf.write("\n\n")
            
            sf.write("--- REGRESSION DIAGNOSTICS (Assumptions Check) ---\n")
            
            sf.write("1. Variance Inflation Factor (VIF):\n")
            vif_data = pd.DataFrame()
            vif_data["feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
            sf.write(vif_data.to_string(index=False) + "\n\n")
            
            shapiro_test = stats.shapiro(reg_model.resid)
            sf.write(f"2. Normality of Residuals (Shapiro-Wilk Test):\n")
            sf.write(f"   W-statistic = {shapiro_test.statistic:.4f}, p-value = {shapiro_test.pvalue:.4f}\n")
            sf.write("   (If p > 0.05, residuals are normally distributed)\n\n")
            
            influence = reg_model.get_influence()
            cooks_d, pvals = influence.cooks_distance
            max_cooks = np.max(cooks_d)
            sf.write(f"3. Cook's Distance (Outlier Influence):\n")
            sf.write(f"   Max Cook's Distance = {max_cooks:.4f}\n")
            sf.write("   (Values > 1 indicate problematic influential outliers)\n\n")
            
            reg_sub_std = reg_sub.copy()
            for col in [corr_col, conf_col]:
                reg_sub_std[col] = (reg_sub_std[col] - reg_sub_std[col].mean()) / reg_sub_std[col].std()
            
            std_model = ols(reg_formula, data=reg_sub_std).fit()
            sf.write("4. Standardized Coefficients (Betas):\n")
            sf.write(std_model.params.to_string() + "\n")
            sf.write("\n\n")
            
            print("Regression & Diagnostics results written to statistical_results.txt")
        except Exception as e:
            print(f"Regression could not be calculated: {e}")
    else:
        print("Skipping Regression: Required demographic/confidence data missing.")

def run_ordinal_logistic_regression(clean_df, conf_col, sf):
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        if conf_col in clean_df.columns:
            ord_df = clean_df.dropna(subset=['Group', conf_col]).copy()
            ord_df[conf_col] = pd.to_numeric(ord_df[conf_col], errors='coerce')
            ord_df = ord_df.dropna(subset=[conf_col])
            ord_df[conf_col] = ord_df[conf_col].astype(int)
            
            ord_df['Is_Natural'] = (ord_df['Group'] == 'NB').astype(float)
            
            cat_type = pd.CategoricalDtype(categories=sorted(ord_df[conf_col].unique()), ordered=True)
            ord_df['Conf_Cat'] = ord_df[conf_col].astype(cat_type)
            
            ord_model = OrderedModel(ord_df['Conf_Cat'], ord_df[['Is_Natural']], distr='logit')
            ord_res = ord_model.fit(method='bfgs', disp=False)
            
            sf.write("\n--- ORDINAL LOGISTIC REGRESSION (Predicting Confidence/H4) ---\n")
            sf.write(ord_res.summary().as_text())
            sf.write("\n\n")
            print("Ordinal Regression results written to statistical_results.txt")
        else:
            print("Skipping Ordinal Regression: Confidence column not found.")
    except ImportError:
            print("Skipping Ordinal Regression: statsmodels.miscmodels.ordinal_model could not be imported.")
    except Exception as e:
            print(f"Ordinal Regression could not be calculated: {e}")
