import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def set_plot_style():
    sns.set_theme(style="whitegrid")

def plot_h1_overall_acc_rt(acc_sub, clean_df, corr_col, rt_col, results_dir):
    plt.figure(figsize=(6, 5))
    sns.barplot(data=acc_sub, x='Group', y=corr_col)
    plt.title('Group Mean Accuracy')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H1_Accuracy.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.violinplot(data=clean_df[clean_df[corr_col]==1], x='Group', y=rt_col, inner='quartile')
    plt.title('RT Distribution (Correct Trials)')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H1_RT.png")
    plt.close()

def plot_h2_pre_boundary(bb_data, corr_col, results_dir):
    plt.figure(figsize=(6, 5))
    sns.barplot(data=bb_data, x='Group', y=corr_col)
    plt.title('BB Accuracy Across Groups')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H2_BB_Accuracy.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.stripplot(data=bb_data, x='Group', y=corr_col, jitter=True, alpha=0.6)
    sns.pointplot(data=bb_data, x='Group', y=corr_col, color="black", markers="D", linestyle='none', errorbar=None)
    plt.title('Individual BB Accuracy Distribution')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H2_BB_Individual.png")
    plt.close()

def plot_h3_event_middle(interaction_acc, corr_col, results_dir):
    plt.figure(figsize=(6, 5))
    sns.barplot(data=interaction_acc, x='Group', y=corr_col, hue='Frame_Type')
    plt.title('Accuracy: BB vs EM')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H3_EM_Accuracy.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    heatmap_data = interaction_acc.pivot_table(index='Group', columns='Frame_Type', values=corr_col, aggfunc='mean')
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=0.5, vmax=1.0)
    plt.title('Mean Accuracy Matrix')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H3_EM_Heatmap.png")
    plt.close()

def plot_h4_confidence(clean_df, conf_col, results_dir):
    conf_props = clean_df.groupby(['Group', conf_col]).size().reset_index(name='count')
    conf_props['percentage'] = conf_props.groupby('Group')['count'].transform(lambda x: 100 * x / x.sum())
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=conf_props, x=conf_col, y='percentage', hue='Group', palette='Set2')
    plt.title('Distribution of Confidence Ratings')
    plt.xlabel('Confidence Rating (1 = Guessing, 5 = Certain)')
    plt.ylabel('Percentage of Answers (%)')
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.1f}%", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 7), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='#333333')
                        
    plt.legend(title='Condition')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H4_Confidence_Bar.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.violinplot(data=clean_df, x='Group', y=conf_col, inner='box')
    plt.title('Confidence Score Distribution Shape')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H4_Confidence_Spread.png")
    plt.close()

def plot_h5_fatigue(clean_df, trial_col, corr_col, results_dir):
    plt.figure(figsize=(7, 5))
    sns.lineplot(data=clean_df, x=trial_col, y=corr_col, hue='Group')
    plt.title('Mean Accuracy over Trial Order')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H5_Fatigue_Line.png")
    plt.close()
    
    plt.figure(figsize=(7, 5))
    clean_df['Trial_Bin'] = pd.qcut(clean_df[trial_col], q=5, labels=['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4', 'Bin 5'], duplicates='drop')
    heatmap_trial = clean_df.pivot_table(index='Group', columns='Trial_Bin', values=corr_col)
    sns.heatmap(heatmap_trial, annot=True, cmap='mako')
    plt.title('Binned Trial Accuracy Heatmap')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H5_Fatigue_Heatmap.png")
    plt.close()

def plot_h6_vision_gender(clean_df, bb_only, sub_gender, corr_col, results_dir):
    plt.figure(figsize=(6, 5))
    sns.barplot(data=bb_only, x='Vision_Clean', y=corr_col, hue='Group')
    plt.title('H6a: BB Acc by Vision x Cond')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H6a_Vision_Bar.png")
    plt.close()
    
    plt.figure(figsize=(6, 5))
    sub_vision = clean_df.groupby(['participant', 'Vision_Clean'])[corr_col].mean().reset_index()
    sns.violinplot(data=sub_vision, x='Vision_Clean', y=corr_col, inner='box')
    plt.title('H6a: Baseline Acc Distribution by Vision')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H6a_Vision_Base.png")
    plt.close()
    
    plt.figure(figsize=(6, 5))
    sns.barplot(data=clean_df, x='Gender', y=corr_col, hue='Group')
    plt.title('H6b: Acc by Gender x Cond')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H6b_Gender_Bar.png")
    plt.close()
    
    plt.figure(figsize=(6, 5))
    sns.violinplot(data=sub_gender, x='Gender', y=corr_col, inner='quartile')
    plt.title('H6b: Acc Distribution by Gender')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/H6b_Gender_Base.png")
    plt.close()



def plot_exploratory(clean_df, corr_col, rt_col, conf_col, results_dir):
    exp_df = clean_df.groupby(['participant', 'Group']).agg({
        corr_col: 'mean',
        rt_col: 'mean',
        conf_col: 'mean'
    }).reset_index()
    
    corr_matrix = exp_df[[corr_col, rt_col, conf_col]].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/Exploratory_Corr_Heatmap.png")
    plt.close()

    sns.pairplot(exp_df, hue='Group', vars=[corr_col, rt_col, conf_col], corner=True)
    plt.savefig(f"{results_dir}/Exploratory_Pairplot.png")
    plt.close()
