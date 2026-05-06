import os
import numpy as np
import pandas as pd
from scipy import stats


def run_ttest_safe(data1, data2, tail=1):
    if len(data1.dropna()) > 0 and len(data2.dropna()) > 0:
        t, p = stats.ttest_ind(data1, data2, nan_policy='omit')
        return t, (p / 2 if tail == 1 else p)
    return np.nan, np.nan


def run_paired_ttest_safe(data1, data2, tail=1):
    if len(data1.dropna()) > 0 and len(data2.dropna()) > 0:
        t, p = stats.ttest_rel(data1, data2, nan_policy='omit')
        return t, (p / 2 if tail == 1 else p)
    return np.nan, np.nan


def run_statistical_tests(clean_df, col_names, results_dir):
    corr_col = col_names['corr_col']
    rt_col = col_names['rt_col']
    conf_col = col_names['conf_col']

    def get_group(df, group_name, col): return df[df['Group'] == group_name][col]
    def get_int(df, group_name, frame_name, col): return df[(df['Group'] == group_name) & (df['Frame_Type'] == frame_name)][col]

    acc_sub = clean_df.groupby(['participant', 'Group'])[corr_col].mean().reset_index()
    rt_sub = clean_df[clean_df[corr_col] == 1].groupby(['participant', 'Group'])[rt_col].mean().reset_index()
    conf_sub = clean_df.groupby(['participant', 'Group'])[conf_col].mean().reset_index()
    interaction_acc = clean_df.groupby(['participant', 'Group', 'Frame_Type'])[corr_col].mean().reset_index()
    interaction_rt = clean_df[clean_df[corr_col] == 1].groupby(['participant', 'Group', 'Frame_Type'])[rt_col].mean().reset_index()
    interaction_conf = clean_df.groupby(['participant', 'Group', 'Frame_Type'])[conf_col].mean().reset_index()

    stats_file = os.path.join(results_dir, "statistical_results.txt")
    sf = open(stats_file, 'w', encoding='utf-8')
    sf.write("--- STATISTICAL TEST RESULTS ---\n\n")

    print("\n" + "="*50)
    print("      ADVANCED HYPOTHESES & VISUALIZATIONS")
    print("="*50)

    # H1 — Overall Accuracy & Response Time (NB vs AB)
    nb_acc = get_group(acc_sub, 'NB', corr_col)
    ab_acc = get_group(acc_sub, 'AB', corr_col)
    t_h1_acc, p_h1_acc = run_ttest_safe(nb_acc, ab_acc, tail=1)
    sf.write(f"H1 - Overall Accuracy (NB > AB):\n  t-statistic = {t_h1_acc:.3f}, p-value = {p_h1_acc:.4f}\n")

    nb_rt = get_group(rt_sub, 'NB', rt_col)
    ab_rt = get_group(rt_sub, 'AB', rt_col)
    t_h1_rt, p_h1_rt = run_ttest_safe(ab_rt, nb_rt, tail=1)  # NB < AB means AB > NB
    sf.write(f"H1 - Response Time (NB < AB):\n  t-statistic = {t_h1_rt:.3f}, p-value = {p_h1_rt:.4f}\n\n")

    # H2 — Pre-Boundary (BB) Accuracy
    nb_bb = get_int(interaction_acc, 'NB', 'BB', corr_col)
    ab_bb = get_int(interaction_acc, 'AB', 'BB', corr_col)
    t_h2, p_h2 = run_ttest_safe(nb_bb, ab_bb, tail=1)
    sf.write(f"H2 - Pre-Boundary Accuracy (NB > AB):\n  t-statistic = {t_h2:.3f}, p-value = {p_h2:.4f}\n\n")

    # H3 — Event-Middle (EM) Accuracy
    nb_em = get_int(interaction_acc, 'NB', 'EM', corr_col)
    ab_em = get_int(interaction_acc, 'AB', 'EM', corr_col)
    t_h3, p_h3 = run_ttest_safe(nb_em, ab_em, tail=2)  # 2-tailed as predicting no difference
    sf.write(f"H3 - Event-Middle Accuracy (NB ≈ AB):\n  t-statistic = {t_h3:.3f}, p-value = {p_h3:.4f}\n\n")

    # H4 — Confidence Ratings
    nb_conf = get_group(conf_sub, 'NB', conf_col)
    ab_conf = get_group(conf_sub, 'AB', conf_col)
    t_h4, p_h4 = run_ttest_safe(nb_conf, ab_conf, tail=1)
    sf.write(f"H4 - Confidence Ratings (NB > AB):\n  t-statistic = {t_h4:.3f}, p-value = {p_h4:.4f}\n\n")

    # H5 — Trial Order / Fatigue Effect
    trial_col = next((col for col in clean_df.columns if 'thisn' in col.lower() or 'trialn' in col.lower()), None)

    if trial_col:
        nb_clean = clean_df[clean_df['Group'] == 'NB'].dropna(subset=[trial_col, corr_col])
        ab_clean = clean_df[clean_df['Group'] == 'AB'].dropna(subset=[trial_col, corr_col])
        r_nb, p_nb = stats.pearsonr(nb_clean[trial_col], nb_clean[corr_col]) if len(nb_clean) > 0 else (np.nan, np.nan)
        r_ab, p_ab = stats.pearsonr(ab_clean[trial_col], ab_clean[corr_col]) if len(ab_clean) > 0 else (np.nan, np.nan)

        sf.write(f"H5 - Fatigue (Accuracy vs Trial Order):\n  NB: r = {r_nb:.3f}, p = {p_nb:.4f}\n  AB: r = {r_ab:.3f}, p = {p_ab:.4f}\n\n")

    # H6a/b — Vision & Gender
    if 'Vision' in clean_df.columns and 'Gender' in clean_df.columns:
        clean_df['Vision_Clean'] = clean_df['Vision'].astype(str).apply(lambda x: 'Normal' if 'normal' in x.lower() and 'corrected' not in x.lower() else ('Corrected' if 'corrected' in x.lower() else np.nan))

        bb_only = clean_df[clean_df['Frame_Type'] == 'BB']
        sub_bb_vision = bb_only.groupby(['participant', 'Group', 'Vision_Clean'])[corr_col].mean().reset_index()
        norm_bb = sub_bb_vision[sub_bb_vision['Vision_Clean'] == 'Normal'][corr_col]
        corr_bb = sub_bb_vision[sub_bb_vision['Vision_Clean'] == 'Corrected'][corr_col]
        t_h6a, p_h6a = run_ttest_safe(norm_bb, corr_bb, tail=1)
        sf.write(f"H6a - Vision (Normal > Corrected on BB):\n  t-statistic = {t_h6a:.3f}, p-value = {p_h6a:.4f}\n")

        sub_gender = clean_df.groupby(['participant', 'Gender'])[corr_col].mean().reset_index()
        fem_acc = sub_gender[sub_gender['Gender'] == 'Female'][corr_col]
        mal_acc = sub_gender[sub_gender['Gender'] == 'Male'][corr_col]
        t_h6b, p_h6b = run_ttest_safe(fem_acc, mal_acc, tail=1)
        sf.write(f"H6b - Gender (Female > Male overall):\n  t-statistic = {t_h6b:.3f}, p-value = {p_h6b:.4f}\n\n")

    # H7a/b — Handedness & Age
    if 'Handedness' in clean_df.columns and 'Age' in clean_df.columns:
        key_cols = [c for c in clean_df.columns if 'keys' in c.lower() and 'resp' in c.lower()]
        key_col = key_cols[0] if key_cols else None
        correct_trials = clean_df[clean_df[corr_col] == 1].copy()

        if key_col:
            top_keys = correct_trials[key_col].value_counts().nlargest(2).index.tolist()
            if len(top_keys) == 2:
                left_handed = correct_trials[correct_trials['Handedness'].str.lower() == 'left'][rt_col]
                right_handed = correct_trials[correct_trials['Handedness'].str.lower() == 'right'][rt_col]
                t_h7a, p_h7a = run_ttest_safe(right_handed, left_handed, tail=2)
                sf.write(f"H7a - Handedness RT diff:\n  t-statistic = {t_h7a:.3f}, p-value = {p_h7a:.4f}\n")

        clean_df['Age'] = pd.to_numeric(clean_df['Age'], errors='coerce')
        sub_age = clean_df.groupby(['participant', 'Group', 'Age'])[corr_col].mean().reset_index()
        r_age, p_age = stats.pearsonr(sub_age['Age'].dropna(), sub_age.dropna(subset=['Age'])[corr_col]) if len(sub_age.dropna(subset=['Age'])) > 0 else (np.nan, np.nan)
        sf.write(f"H7b - Age correlation with Accuracy:\n  r = {r_age:.3f}, p = {p_age:.4f}\n\n")

    sf.close()
