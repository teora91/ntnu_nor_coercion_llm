import streamlit as st
import pandas as pd
import numpy as np
import ast
import emoji
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns


# Function to load multiple DataFrames
max_score, min_score = min_max_score()

calculations = {'avg_precision': 'mean', 'hard_rank_mean': 'mean', 'DIFF_hard_rank_mean': 'mean', 'soft_rank_mean':'mean', 'DIFF_soft_rank_mean':'mean'}
style_apply_subset = [ 'avg_precision', 'hard_rank_mean', 'DIFF_hard_rank_mean', 'soft_rank_mean', 'DIFF_soft_rank_mean']



@st.cache_data
def load_data():
    results_no_context = pd.read_csv("results_with_acc_NEW_with_ALLMODELS_updated_02-02-25_with_contex_FALSE.csv")
    results_no_context = results_no_context.fillna("0")
    results_no_context["predicted_events"] = results_no_context["predicted_events"].apply(ast.literal_eval)
    results_no_context["total_score_list"] = results_no_context["total_score_list"].apply(ast.literal_eval)
    # results_no_context["pred softmax"] = results_no_context["pred softmax"].apply(ast.literal_eval)
    results_no_context = results_no_context[~(results_no_context["entity"]== "bolle")]
    results_no_context = results_no_context[~(results_no_context["entity"]== "fane")]
    results_no_context['model'] = results_no_context['model'].str.replace(r".+/", "", regex=True)
    

    df1 = pd.read_csv("dataset_part1.csv")
    df2 = pd.read_csv("dataset_part2.csv")

    # Reunite them
    results_with_context = pd.concat([df1, df2], ignore_index=True)

    results_with_context = results_with_context.fillna("0")
    results_with_context["predicted_events"] = results_with_context["predicted_events"].apply(ast.literal_eval)
    results_with_context["total_score_list"] = results_with_context["total_score_list"].apply(ast.literal_eval)
    # results_with_context["pred softmax"] = results_with_context["pred softmax"].apply(ast.literal_eval)
    results_with_context = results_with_context[~(results_with_context["entity"]== "bolle")]
    results_with_context = results_with_context[~(results_with_context["entity"]== "fane")]
    results_with_context['model'] = results_with_context['model'].str.replace(r".+/", "", regex=True)

    return {"Results NO context": results_no_context, "Results WITH context":results_with_context}



# Load data
dataframes = load_data()

# **Title**


def top_five(selected_df_type):

    # **Select which DataFrame to view**
    selected_df_name = st.sidebar.selectbox("Select a DataFrame:", list(dataframes.keys()))
    dict_results_dataframe = {'Results NO context': "No Context", "Results NO context": "With Context"}
    
    st.title(f"Results Complement Coercion Norwegian {dict_results_dataframe[selected_df_name]}, {selected_df_type}")
    
    st.header("Results Updated 02/02/2025")


    # Get the selected DataFrame
    df = dataframes[selected_df_name].drop(columns=(["Unnamed: 0", "qualias"]))

    df['avg_precision'] = df['total_score_list'].apply(precision_calculation)

    df['hard_rank_mean'] = df['total_score_list'].apply(hard_rank_mean)
    df["DIFF_hard_rank_mean"] = df['hard_rank_mean'] - max_score #RCS discance

    df['soft_rank_mean'] = df['total_score_list'].apply(soft_rank_mean)
    df["DIFF_soft_rank_mean"] = df['soft_rank_mean'] - max_score #RCS discance

    st.text(f"Max Score: {max_score}")
    
    st.header("Results ")

    x_1 = df.groupby('model', as_index=False).agg(calculations)
    # x_1.to_csv("official_results/results_model_no_context.csv", index=True)

    x_1_ = x_1.style.apply(highlight_max, subset=style_apply_subset)
    st.dataframe(x_1_)


    #Line plotting 
    st.header("Line Plotting Hard Rank Mean")

    line_plotting_line(x_1['model'], x_1['hard_rank_mean'], max_score, "Hard", x_1[x_1['model'] == "NCC"]['hard_rank_mean'].values)

    st.header("Line Plotting Soft Rank Mean")

    line_plotting_line(x_1['model'], x_1['soft_rank_mean'], max_score, "Soft", x_1[x_1['model'] == "NCC"]['soft_rank_mean'].values)

    st.header("Heatmap Hard Rank Mean")

    heatmap_diff_ranking(x_1, x_1[['DIFF_hard_rank_mean']], 'DIFF_hard_rank_mean')
    print("___"*100)

    st.header("Heatmap Soft Rank Mean")

    heatmap_diff_ranking(x_1, x_1[['DIFF_soft_rank_mean']], 'DIFF_soft_rank_mean')


    # **Select which DataFrame to view**
    selected_element = st.sidebar.selectbox("Select:", ["Entity cateogires", "Verbs", "Post-Verbal Syntactic Structure"])

    dict_selction = {"Entity cateogires":"entity_cat", "Verbs":"verb", "Post-Verbal Syntactic Structure":"prep"}

    df_2 = df[df["model"]!= 'NCC']
    df_2 = df_2[df_2["model"]!= 'nb-gpt-j-6B']
    df_2 = df_2[df_2["model"]!= 'nb-gpt-j-6B-v2']


    st.header("Selection Best Models")
    st.text("Best Models: norbert3-base, NorLlama-3B")

    best1 = df_2[(df_2['model'].isin(['norbert3-base']))]
    best2 = df_2[(df_2['model'].isin(['NorLlama-3B']))]


    st.header(f"Filtered Analysis Based on {selected_element}")

    st.text("Remember that 'NCC', 'nb-gpt-j-6B', and 'nb-gpt-j-6B-v2' are discarded in this analysis")
    filtered_df_2 = df_2.groupby(dict_selction[selected_element], as_index=False).agg(calculations).style.apply(highlight_max, subset=style_apply_subset)

    st.dataframe(filtered_df_2)

    st.header(f"Filtered Analysis of the Best Models Based on {selected_element}")

    st.subheader(f"norbert3-base")
    filtered_best1 = best1.groupby(['model', dict_selction[selected_element]], as_index=False).agg(calculations).style.apply(highlight_max, subset=style_apply_subset)
    st.dataframe(filtered_best1)


    st.subheader(f"NorLlama-3B")
    filtered_best2 = best2.groupby(['model', dict_selction[selected_element]], as_index=False).agg(calculations).style.apply(highlight_max, subset=style_apply_subset)
    st.dataframe(filtered_best2)
    
    
    
def one_shot(selected_df_type):
    selected_df_name = st.sidebar.selectbox("Select a DataFrame:", list(dataframes.keys()))
    
    dict_results_dataframe = {'Results NO context': "No Context", "Results NO context": "With Context"}
    
    st.title(f"Results Complement Coercion Norwegian {dict_results_dataframe[selected_df_name]}, {selected_df_type}")
    st.header("Results Updated 02/02/2025")

    
    

    # Get the selected DataFrame
    df = dataframes[selected_df_name].drop(columns=(["Unnamed: 0", "qualias"]))

    df['avg_precision'] = df['total_score_list'].apply(precision_calculation)

    df['hard_rank_mean'] = df['total_score_list'].apply(hard_rank_mean)
    df["DIFF_hard_rank_mean"] = df['hard_rank_mean'] - max_score #RCS discance

    df['soft_rank_mean'] = df['total_score_list'].apply(soft_rank_mean)
    df["DIFF_soft_rank_mean"] = df['soft_rank_mean'] - max_score #RCS discance
    
    df_one_shot = df.copy()

    df_one_shot["lexical aspect"] = df_one_shot["lexical aspect"].apply(ast.literal_eval)
    # 
    df_one_shot['lexical aspect'] = df_one_shot['lexical aspect'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    df_one_shot['results_one_shot_accomplishment'] = df_one_shot['lexical aspect'].apply(calc_one_shot_hard)
    
    
    results1 = df_one_shot.groupby('model', as_index=False)['results_one_shot_accomplishment'].mean().style.apply(highlight_max, subset="results_one_shot_accomplishment")
    
    df_one_shot
    
    st.dataframe(results1)
    
    df_2_one_shot = df_one_shot[df_one_shot["model"]!= 'NCC']
    df_2_one_shot = df_2_one_shot[df_2_one_shot["model"]!= 'nb-gpt-j-6B']
    df_2_one_shot = df_2_one_shot[df_2_one_shot["model"]!= 'nb-gpt-j-6B-v2']


    
    dict_selction = {"Entity cateogires":"entity_cat", "Verbs":"verb", "Post-Verbal Syntactic Structure":"prep"}

    selected_element = st.sidebar.selectbox("Select Filtered Results:", ["Entity cateogires", "Verbs", "Post-Verbal Syntactic Structure"])
    
    
    st.text("Remember that 'NCC', 'nb-gpt-j-6B', and 'nb-gpt-j-6B-v2' are discarded in this analysis")
    filtered_df_2 = df_2_one_shot.groupby(dict_selction[selected_element], as_index=False)['results_one_shot_accomplishment'].mean().style.apply(highlight_max, subset="results_one_shot_accomplishment")

    st.header(f"Filtered Analysis Based on {selected_element}")

    st.dataframe(filtered_df_2)
    
    st.header("Selection Best Models")
    st.text("norbert3-small")

    best1_one_shot = df_2_one_shot[(df_2_one_shot['model'].isin(['norbert3-small']))]


    st.header(f"Filtered Analysis of the Best Models Based on {selected_element}")

    st.subheader(f"norbert3-small")
    filtered_best1 = best1_one_shot.groupby(['model', dict_selction[selected_element]], as_index=False)['results_one_shot_accomplishment'].mean().style.apply(highlight_max, subset="results_one_shot_accomplishment")
    
    st.dataframe(filtered_best1)

    





    
    
    
    
    
selected_df_type = st.sidebar.selectbox("Top-5 vs One-Hot:",["Top-5", "One-Shot"])
if selected_df_type == 'Top-5':
        top_five(selected_df_type)
elif selected_df_type == 'One-Shot':
        one_shot(selected_df_type)
        
        




# # **Sidebar filters**
# st.sidebar.header(f"🔍 Filters for {selected_df_name}")

# # **Dynamically create filters based on the selected DataFrame**
# filtered_df = df.copy()

# for col in df.columns:
#     if df[col].dtype == 'object':  # Categorical columns (Strings)
#         st.sidebar.subheader(f"Filter by {col}:")
#         unique_values = df[col].unique().tolist()
        
#         # Create a checkbox for each unique value
#         selected_values = [value for value in unique_values if st.sidebar.checkbox(f"Include {value}", value=True)]
        
#         # Apply filter if at least one value is selected
#         if selected_values:
#             filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
    
#     elif df[col].dtype in ['int64', 'float64']:  # Numeric columns
#         min_val, max_val = df[col].min(), df[col].max()
#         range_values = st.sidebar.slider(f"Filter by {col}:", min_value=int(min_val), max_value=int(max_val), value=(int(min_val), int(max_val)))
#         filtered_df = filtered_df[(filtered_df[col] >= range_values[0]) & (filtered_df[col] <= range_values[1])]

# # **Display the filtered DataFrame**
# st.dataframe(filtered_df)

# # **Download button for filtered data**
# st.download_button(label="Download Data", data=filtered_df.to_csv(index=False), file_name=f"{selected_df_name.replace(' ', '_')}.csv", mime="text/csv")


# for i in ["cacca", "culo"]:
# st.sidebar.checkbox()
