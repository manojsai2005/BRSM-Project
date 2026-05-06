import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_visualizations(clean_df, col_names, results_dir):
    corr_col = col_names['corr_col']
    rt_col = col_names['rt_col']
    conf_col = col_names['conf_col']

    sns.set_theme(style="whitegrid")

    acc_sub = clean_df.groupby(['participant', 'Group'])[corr_col].mean().reset_index()
    rt_sub = clean_df[clean_df[corr_col] == 1].groupby(['participant', 'Group'])[rt_col].mean().reset_index()
    conf_sub = clean_df.groupby(['participant', 'Group'])[conf_col].mean().reset_index()
    interaction_acc = clean_df.groupby(['participant', 'Group', 'Frame_Type'])[corr_col].mean().reset_index()
    interaction_rt = clean_df[clean_df[corr_col] == 1].groupby(['participant', 'Group', 'Frame_Type'])[rt_col].mean().reset_index()
    interaction_conf = clean_df.groupby(['participant', 'Group', 'Frame_Type'])[conf_col].mean().reset_index()

    # H1 — Overall Accuracy & Response Time (NB vs AB)
    print("\n--- H1: Overall Acc & RT ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=acc_sub, x='Group', y=corr_col, ax=axes[0])
    axes[0].set_title('Group Mean Accuracy')

    sns.violinplot(data=clean_df[clean_df[corr_col]==1], x='Group', y=rt_col, inner='quartile', ax=axes[1])
    axes[1].set_title('RT Distribution (Correct Trials)')

    plt.tight_layout()
    plt.savefig(f"{results_dir}/H1_Acc_RT.png")
    plt.close()

    # H2 — Pre-Boundary (BB) Accuracy
    print("\n--- H2: Pre-Boundary (BB) Acc ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bb_data = interaction_acc[interaction_acc['Frame_Type'] == 'BB']

    sns.barplot(data=bb_data, x='Group', y=corr_col, ax=axes[0])
    axes[0].set_title('BB Accuracy Across Groups')

    sns.stripplot(data=bb_data, x='Group', y=corr_col, jitter=True, alpha=0.6, ax=axes[1])
    sns.pointplot(data=bb_data, x='Group', y=corr_col, color="black", markers="D", linestyle='none', errorbar=None, ax=axes[1])
    axes[1].set_title('Individual BB Accuracy Distribution')

    plt.tight_layout()
    plt.savefig(f"{results_dir}/H2_BB_Accuracy.png")
    plt.close()

    # H3 — Event-Middle (EM) Accuracy
    print("\n--- H3: Event-Middle vs Boundary ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=interaction_acc, x='Group', y=corr_col, hue='Frame_Type', ax=axes[0])
    axes[0].set_title('Accuracy: BB vs EM')

    heatmap_data = interaction_acc.pivot_table(index='Group', columns='Frame_Type', values=corr_col, aggfunc='mean')
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=0.5, vmax=1.0, ax=axes[1])
    axes[1].set_title('Mean Accuracy Matrix')

    plt.tight_layout()
    plt.savefig(f"{results_dir}/H3_EM_Accuracy.png")
    plt.close()

    # H4 — Confidence Ratings
    print("\n--- H4: Confidence Ratings ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    conf_counts = clean_df.groupby(['Group', conf_col]).size().unstack(fill_value=0)
    conf_props = conf_counts.div(conf_counts.sum(axis=1), axis=0)
    conf_props.plot(kind='bar', stacked=True, ax=axes[0], cmap='viridis')
    axes[0].set_title('Confidence Level Distribution (1-5)')
    axes[0].set_ylabel('Proportion')

    sns.boxplot(data=clean_df, x='Group', y=conf_col, ax=axes[1])
    axes[1].set_title('Confidence Score Spread')

    plt.tight_layout()
    plt.savefig(f"{results_dir}/H4_Confidence.png")
    plt.close()

    # H5 — Trial Order / Fatigue Effect
    print("\n--- H5: Trial Order Fatigue ---")
    trial_col = next((col for col in clean_df.columns if 'thisn' in col.lower() or 'trialn' in col.lower()), None)

    if trial_col:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        sns.lineplot(data=clean_df, x=trial_col, y=corr_col, hue='Group', ax=axes[0])
        axes[0].set_title('Mean Accuracy over Trial Order')

        clean_df['Trial_Bin'] = pd.qcut(clean_df[trial_col], q=5, labels=['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4', 'Bin 5'], duplicates='drop')
        heatmap_trial = clean_df.pivot_table(index='Group', columns='Trial_Bin', values=corr_col)
        sns.heatmap(heatmap_trial, annot=True, cmap='mako', ax=axes[1])
        axes[1].set_title('Binned Trial Accuracy Heatmap')

        plt.tight_layout()
        plt.savefig(f"{results_dir}/H5_Fatigue.png")
        plt.close()
    else:
        print("Could not find a Trial Number column for H5.")

    # H6a/b — Vision & Gender
    print("\n--- H6: Vision and Gender ---")
    if 'Vision' in clean_df.columns and 'Gender' in clean_df.columns:
        clean_df['Vision_Clean'] = clean_df['Vision'].astype(str).apply(lambda x: 'Normal' if 'normal' in x.lower() and 'corrected' not in x.lower() else ('Corrected' if 'corrected' in x.lower() else np.nan))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        bb_only = clean_df[clean_df['Frame_Type'] == 'BB']

        sns.barplot(data=bb_only, x='Vision_Clean', y=corr_col, hue='Group', ax=axes[0, 0])
        axes[0, 0].set_title('H6a: BB Acc by Vision x Cond')

        sub_vision = clean_df.groupby(['participant', 'Vision_Clean'])[corr_col].mean().reset_index()
        sns.boxplot(data=sub_vision, x='Vision_Clean', y=corr_col, ax=axes[0, 1])
        axes[0, 1].set_title('H6a: Baseline Acc by Vision')

        sns.barplot(data=clean_df, x='Gender', y=corr_col, hue='Group', ax=axes[1, 0])
        axes[1, 0].set_title('H6b: Acc by Gender x Cond')

        sub_gender = clean_df.groupby(['participant', 'Gender'])[corr_col].mean().reset_index()
        sns.violinplot(data=sub_gender, x='Gender', y=corr_col, inner='quartile', ax=axes[1, 1])
        axes[1, 1].set_title('H6b: Acc Distribution by Gender')

        plt.tight_layout()
        plt.savefig(f"{results_dir}/H6_Vision_Gender.png")
        plt.close()
    else:
        print("Demographic Vision/Gender data not found for H6.")

    # H7a/b — Handedness & Age
    print("\n--- H7: Handedness and Age ---")
    if 'Handedness' in clean_df.columns and 'Age' in clean_df.columns:
        key_cols = [c for c in clean_df.columns if 'keys' in c.lower() and 'resp' in c.lower()]
        key_col = key_cols[0] if key_cols else None
        correct_trials = clean_df[clean_df[corr_col] == 1].copy()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        if key_col:
            top_keys = correct_trials[key_col].value_counts().nlargest(2).index.tolist()
            if len(top_keys) == 2:
                hand_key_df = correct_trials[correct_trials[key_col].isin(top_keys)]
                sns.barplot(data=hand_key_df, x='Handedness', y=rt_col, hue=key_col, ax=axes[0, 0])
                axes[0, 0].set_title('H7a: RT by Handedness x Key')

        sub_hand = correct_trials.groupby(['participant', 'Handedness'])[rt_col].mean().reset_index()
        sns.boxplot(data=sub_hand, x='Handedness', y=rt_col, ax=axes[0, 1])
        axes[0, 1].set_title('H7a: Baseline RT by Handedness')

        clean_df['Age'] = pd.to_numeric(clean_df['Age'], errors='coerce')
        sub_age = clean_df.groupby(['participant', 'Group', 'Age'])[corr_col].mean().reset_index()
        sub_age['Age_Group'] = pd.cut(sub_age['Age'], bins=[17, 21, 25, 40], labels=['18-21', '22-25', '26+'])
        sns.barplot(data=sub_age, x='Age_Group', y=corr_col, hue='Group', ax=axes[1, 0])
        axes[1, 0].set_title('H7b: Acc by Age Group x Cond')

        sns.scatterplot(data=sub_age, x='Age', y=corr_col, hue='Group', ax=axes[1, 1])
        sns.regplot(data=sub_age, x='Age', y=corr_col, scatter=False, ax=axes[1, 1], color='black')
        axes[1, 1].set_title('H7b: Continuous Age vs Acc')

        plt.tight_layout()
        plt.savefig(f"{results_dir}/H7_Handedness_Age.png")
        plt.close()

    # Exploratory: Correlations & Pairplot
    print("\n--- Exploratory: Correlations & Pairplot ---")
    exp_df = clean_df.groupby(['participant', 'Group']).agg({
        corr_col: 'mean',
        rt_col: 'mean',
        conf_col: 'mean',
        'Age': 'first'
    }).reset_index()

    corr_matrix = exp_df[[corr_col, rt_col, conf_col, 'Age']].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/Exploratory_Corr_Heatmap.png")
    plt.close()

    sns.pairplot(exp_df, hue='Group', vars=[corr_col, rt_col, conf_col, 'Age'], corner=True)
    plt.savefig(f"{results_dir}/Exploratory_Pairplot.png")
    plt.close()

    print("\n" + "="*50)
    print("ALL ADVANCED GRAPHS SAVED TO analysis_results/ ")
    print("="*50)
