import streamlit as st
import pandas as pd
import numpy as np
import ast
import emoji
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import re


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
    
    pll_simple_results = pd.read_csv("../pll_analysis/results_pll_simple_six_models.csv")
    
    

    

    return {"Results NO context": results_no_context, "Results WITH context":results_with_context, "Model Comparison": [results_no_context, results_with_context], "PLL_Syntax": pll_simple_results}



# Load data
dataframes = load_data()

# **Title**


def top_five(selected_df_type):

    # **Select which DataFrame to view**
    selected_df_name = st.sidebar.selectbox("Select a DataFrame:", list(dataframes.keys())[:-1])
    dict_results_dataframe = {'Results NO context': "No Context", "Results WITH context": "With Context"}
    
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
    selected_df_name = st.sidebar.selectbox("Select a DataFrame:", list(dataframes.keys())[:-1])
    
    dict_results_dataframe = {'Results NO context': "No Context", "Results WITH context": "With Context"}
    
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
    
    
    results_1 = df_one_shot.groupby('model', as_index=False)['results_one_shot_accomplishment'].mean().style.apply(highlight_max, subset="results_one_shot_accomplishment")
    
    df_one_shot
    
    st.dataframe(results_1)
    
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

    
def model_comparison():
    st.title(f"Results Model Comparison between LLMs in Complement Coercion in Norwegian")
    
    st.header("Results Updated 02/02/2025")
    df_no_context = dataframes['Model Comparison'][0]
    df_with_context = dataframes['Model Comparison'][1]
    

    df_no_context['avg_precision'] = df_no_context['total_score_list'].apply(precision_calculation)

    df_no_context['hard_rank_mean'] = df_no_context['total_score_list'].apply(hard_rank_mean)
    df_no_context["DIFF_hard_rank_mean"] = df_no_context['hard_rank_mean'] - max_score #RCS discance

    df_no_context['soft_rank_mean'] = df_no_context['total_score_list'].apply(soft_rank_mean)
    df_no_context["DIFF_soft_rank_mean"] = df_no_context['soft_rank_mean'] - max_score #RCS discance



    df_with_context['avg_precision'] = df_with_context['total_score_list'].apply(precision_calculation)

    df_with_context['hard_rank_mean'] = df_with_context['total_score_list'].apply(hard_rank_mean)
    df_with_context["DIFF_hard_rank_mean"] = df_with_context['hard_rank_mean'] - max_score #RCS discance

    df_with_context['soft_rank_mean'] = df_with_context['total_score_list'].apply(soft_rank_mean)
    df_with_context["DIFF_soft_rank_mean"] = df_with_context['soft_rank_mean'] - max_score #RCS discance
    
    
    x_1_no_context = df_no_context.groupby('model', as_index=False).agg(calculations)
    # x_1_no_context_ = x_1_no_context.style.apply(highlight_max, subset=style_apply_subset)
    # st.dataframe(x_1_no_context_)

    x_1_with_context = df_with_context.groupby('model', as_index=False).agg(calculations)
    # x_1_with_context_ = x_1_with_context.style.apply(highlight_max, subset=style_apply_subset)
    # st.dataframe(x_1_with_context_)
    
    x_1_no_context_ = x_1_no_context.rename(columns={'avg_precision':'avg_precision_no_context' ,'hard_rank_mean':'hard_rank_mean_no_context' ,'DIFF_hard_rank_mean':'DIFF_hard_rank_mean_no_context' ,'soft_rank_mean':'soft_rank_mean_no_context' ,'DIFF_soft_rank_mean':'DIFF_soft_rank_mean_no_context'})
    x_1_with_context_ = x_1_with_context.rename(columns={'avg_precision':'avg_precision_with_context' ,'hard_rank_mean':'hard_rank_mean_with_context' ,'DIFF_hard_rank_mean':'DIFF_hard_rank_mean_with_context' ,'soft_rank_mean':'soft_rank_mean_with_context' ,'DIFF_soft_rank_mean':'DIFF_soft_rank_mean_with_context'})
    x_1_with_context_ = x_1_with_context_.drop(columns=(["model"]))
    
    concat_performance_models = pd.concat([x_1_no_context_, x_1_with_context_], axis=1)
    concat_performance_models
    pattern1 = r"model|avg_precision_"
    pattern2 = r"model|hard_rank_mean_"
    pattern3 = r"model|soft_rank_mean_"

    avg_precision = difference_context_vs_no_context(concat_performance_models, pattern1)
    st.dataframe(avg_precision)
    
    plotting_resutls_comparison(avg_precision, 'avg_precision_no_context', 'avg_precision_with_context', 'Avg Precision')


    hard_rank_mean_ = difference_context_vs_no_context(concat_performance_models, pattern2)
    st.dataframe(hard_rank_mean_)
    
    plotting_resutls_comparison(hard_rank_mean_, 'hard_rank_mean_no_context', 'hard_rank_mean_with_context', 'Hard rank mean')
    
    soft_rank_mean_ = difference_context_vs_no_context(concat_performance_models, pattern3)
    soft_rank_mean_
    
    plotting_resutls_comparison(soft_rank_mean_, 'soft_rank_mean_no_context', 'soft_rank_mean_with_context', 'Soft rank mean')


def pll_syntax():
    st.title(f"Results PLL Simple Complement Coercion Norwegian")
    
    st.header("Results Updated 04/02/2025")
    
    
    st.text("Pseudo-Log likelihood (PLL) of the whole simple coercion sentence SUB+VERB+[PP|NP]")

    st.latex(r'''
    PLL_{\text{orig}}(S) := \sum_{t=1}^{n} \log P_{\text{MLM}}(s_t | S_{\setminus t})
    ''')
    
    st.markdown("Adaptation of PLL bassed on [Kauf and Ivanova 2023](https://arxiv.org/pdf/2305.10588) for MLM models")

    
    st.latex(r'''
        PLL_{\text{ww}}(S) := \sum_{w=1}^{|S|} \sum_{t=1}^{|w|} \log P_{\text{MLM}}(s_{w,t} | S_{\setminus s_w}) ''')

    
    df_pll_simple = dataframes['PLL_Syntax']
    df_pll_simple
    
    list_models = list(df_pll_simple['model'].unique()) + ['ALL']
    list_cat = ['verb', 'entity_cat', 'entity']
    selected_model = st.sidebar.selectbox("Select a Model:",list_models )
    selected_cat = st.sidebar.selectbox("Select a Category:",list_cat )
    # dict_results_dataframe = {'Results NO context': "No Context", "Results WITH context": "With Context"}

    if selected_model == "ALL":
        for v in df_pll_simple['model'].unique():
            st.subheader(f"Resutls from {v}")
            filter_ = df_pll_simple[df_pll_simple['model']==v]
            filter_ = filter_.groupby(selected_cat, as_index=False).agg({'pll_på':'mean', 'pll_med':'mean', 'pll_null':'mean' })
            filter_  =filter_.set_index(filter_[selected_cat]).drop(columns=([selected_cat]))

            grouped_bar_chart(filter_, filter_.index, f'PLL Results of all Models based on {selected_cat}')

            print("\n\n\n\n")
    else:
        st.subheader(f"Resutls from {selected_model} based on {selected_cat}")

        results_pll_verbs = df_pll_simple[df_pll_simple['model']==selected_model].groupby(selected_cat, as_index=False).agg({'pll_på':'mean', 'pll_med':'mean', 'pll_null':'mean' })
        results_pll_verbs  =results_pll_verbs.set_index(results_pll_verbs[selected_cat]).drop(columns=([selected_cat]))

        grouped_bar_chart(results_pll_verbs, results_pll_verbs.index, selected_cat)
        results_pll_verbs


    
    
    
    
    
selected_df_type = st.sidebar.selectbox("What analyze?:",["Top-5", "One-Shot", "Model Comparison", "PLL_Syntax"])
if selected_df_type == 'Top-5':
        top_five(selected_df_type)
elif selected_df_type == 'One-Shot':
        one_shot(selected_df_type)
elif selected_df_type == 'Model Comparison':
        model_comparison()
elif selected_df_type == 'PLL_Syntax':
        pll_syntax()

        




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