import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

def min_max_score_hard_soft():
    max_score, min_score = 0,0
    for i in range(1, 11):
        max_score+= 1/i

    for i in range(1, 11):
        min_score+= 0/i
    return max_score, min_score

def precision_calculation(scores):
    
    scores = np.array(scores)
    
    accomplishemnt_freq = sum(scores == 3)
    precision = accomplishemnt_freq / 5
    return precision

def rank_mean(scores):
    rank_mean = []
    for idx, s in enumerate(scores, start =1):
        rank_mean.append(s / idx)
    rank_mean = np.array(rank_mean).sum()
    return rank_mean
        
        
def hard_rank_mean(scores):
    
    scores = np.array(scores)
    accomplishemnt_freq = np.where(scores == 3, scores, 0)
    rank_mean = []
    for idx, s in enumerate(accomplishemnt_freq, start =1):
        if s == 0:
            rank_mean.append(0)
        else:
            rank_mean.append(1/ idx)
    rank_mean = np.array(rank_mean).sum()
    return rank_mean

def soft_rank_mean(scores):
    scores = np.array(scores)
    accomplishemnt_freq = np.where((scores == 3)| (scores == 1), scores, 0)    
    rank_mean = []
    for idx, s in enumerate(accomplishemnt_freq, start =1):
        if s == 0:
            rank_mean.append(0)
        else:
            rank_mean.append(1/ idx)
    rank_mean = np.array(rank_mean).sum()
    return rank_mean


def min_max_score():
    max_score, min_score = min_max_score_hard_soft()
    return  max_score, min_score


def highlight_max(s):
    is_max = s == s.max()  # Identify the maximum value
    return ['color: red; font-weight: bold' if v else '' for v in is_max]

def line_plotting_line(col_1, col_2, max_score, type_, baseline_):
    # plt.figure(figsize=(12, 6))
    # plt.plot(col_1, col_2, label=f"Average Mean {type_} Score", marker='o', color='blue')
    # plt.axhline(y=max_score, color='green', linestyle='--', label=f"Max Score ({max_score})")
    # plt.axhline(y=baseline_, color='red', linestyle='--', label=f"Baseline")
    # plt.xticks(rotation=90)
    # plt.xlabel("Models")
    # plt.ylabel("Scores")
    # plt.title(f"Average Mean Score {type_} (max score: {max_score})")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(col_1, col_2, label=f"Average Mean {type_} Score", marker='o', color='blue')
    ax.axhline(y=max_score, color='green', linestyle='--', label=f"Max Score ({max_score})")
    ax.axhline(y=baseline_, color='red', linestyle='--', label=f"Baseline")
    ax.set_xticklabels(col_1, rotation=90)
    ax.set_xlabel("Models")
    ax.set_ylabel("Scores")
    ax.set_title(f"Average Mean Score {type_} (max score: {max_score})")
    ax.legend()
    ax.grid()
    
    # Display in Streamlit
    st.pyplot(fig)
    
    
def heatmap_diff_ranking(df_, col_name, labels):
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(col_name.set_index(x_1_no_context["model"]), annot=True, cmap='coolwarm', cbar_kws={'label': labels})
    # # plt.title("Heatmap of Distance (Max Score - Average RCS)")
    # # plt.xlabel("Metrics")
    # # plt.ylabel("Models")
    # # plt.xticks(rotation=90)
    # # plt.tight_layout()
    # plt.show()

    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.heatmap(col_name.set_index(df_["model"]), annot=True, cmap='coolwarm', cbar_kws={'label': labels})
    st.pyplot(fig)


def calc_one_shot_hard(scores):
    if scores == 'accomplishment': #SHOULD I INCLUDEALSO ACHIEVMENT?
        return 1
    else:
        return 0

    
def difference_context_vs_no_context(df_, pattern):
    x = df_.copy()
    x = x.filter(regex=pattern)
    x["DIFF"] = x.iloc[:,-1] - x.iloc[:,-2]
    return x
    
    
def plotting_resutls_comparison(df, cond_1, cond_2, value_name): #'avg_precision_no_context', 'avg_precision_with_context' 'Avg Precision'
    # df = df[df["model"]!="NCC"]
    df_melted = df.melt(
        id_vars=['model'], 
        value_vars=[cond_1, cond_2],
        var_name='Condition', 
        value_name= value_name
    )
    df_melted['Condition'] = df_melted['Condition'].replace({
        'cond_1': 'No Context',
        'cond_2': 'With Context'
    })
    fig, ax = plt.subplots(figsize=(12, 6))
    # plt.figure(figsize=(12,6))
    ax = sns.barplot(
        data=df_melted, 
        x='model', 
        y=value_name, 
        hue='Condition',  # Overlapping bars for conditions
        alpha=0.8  # Adjust transparency for overlap visibility
    )

    # Improve readability
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.ylabel(value_name)
    plt.xlabel("Model Name")
    plt.title(f"Comparison of {value_name} With and Without Context")
    plt.legend(title="Condition")  # Add legend

    # Show plot
    st.pyplot(fig)

def grouped_bar_chart(df, col_name, col_name_str):
# Grouped Bar Chart
    # Plot grouped horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="barh", figsize=(8, 5), width=0.7, ax=ax)
    ax.set_title("Performance Comparison by PLL Categories (High Value = High Surprise)")
    ax.set_xlabel("Values (Higher = Worse)")
    ax.set_ylabel("PLL Categories")
    ax.legend(title="Verbs")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Show in Streamlit
    st.pyplot(fig)
    
