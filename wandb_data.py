import wandb
import pandas as pd
import os
from tqdm.autonotebook import tqdm
import numpy as np


sample = False  # For testing purposes

get_wandb_data = False

def smooth(scalars, weight = 0.9):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

if get_wandb_data:

    api = wandb.Api()
    project_entity = "aicrowd/flatland-paper"
    runs = api.runs(project_entity)
    reports = api.reports(project_entity)
    eval_metrics = ["evaluation/custom_metrics/percentage_complete_mean","evaluation/custom_metrics/episode_score_normalized_mean","timesteps_total"]
    train_metrics = ["custom_metrics/percentage_complete_mean","custom_metrics/episode_score_normalized_mean","timesteps_total"]
    expert_metrics = ["expert_episode_completion_mean","expert_episode_reward_mean"]
    all_metrics =  train_metrics + eval_metrics
    exclude_runs = ["eval_recording_ppo_tree_obs"]

    eval_metrics_names = [eval_metric.replace("/", "_") for eval_metric in eval_metrics]

    summary_list = [] 
    config_list = [] 
    name_list = []
    df_all_eval = pd.DataFrame(columns=eval_metrics+['run','group']) 
    df_all_train = pd.DataFrame(columns=train_metrics+['run','group'])
    for run in tqdm(runs): 

        if run.name not in exclude_runs:
            # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files 
            summary_list.append(run.summary._json_dict) 

            # run.config is the input metrics.  We remove special values that start with _.
            config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')}) 

            # run.name is the name of the run.
            name_list.append(run.name)
            df_eval = pd.DataFrame(columns=eval_metrics)
            df_train = pd.DataFrame(columns=train_metrics)
            history_eval = run.scan_history(eval_metrics)
            history_train = run.scan_history(train_metrics)
            i=0
            for row in history_eval:
                df_eval = df_eval.append({eval_metrics[0]:row.get(eval_metrics[0]),
                        eval_metrics[1]:row.get(eval_metrics[1]),eval_metrics[2]:row.get(eval_metrics[2])},  
                        ignore_index = True)
                if sample:
                    i+=1
                    if i >2:
                        break
            
            i=0        
            for row in history_train:        
                df_train = df_train.append({train_metrics[0]:row.get(train_metrics[0]),
                        train_metrics[1]:row.get(train_metrics[1]),train_metrics[2]:row.get(train_metrics[2])},  
                        ignore_index = True)
                if sample:
                    i+=1
                    if i >2:
                        break
            df_eval['run'] = run.name
            df_train['run'] = run.name

            group = run.config.get('env_config',{}).get('yaml_config')

            if group:
                group = group.split(os.sep)[-1].split('.')[0]

            df_eval['group'] = group
            df_train['group'] = group

            df_eval.sort_values(by="timesteps_total",inplace=True)
            df_train.sort_values(by="timesteps_total",inplace=True)
            df_all_eval = pd.concat([df_all_eval,df_eval])
            df_all_train = pd.concat([df_all_train,df_train])         

    summary_df = pd.DataFrame.from_records(summary_list) 
    config_df = pd.DataFrame.from_records(config_list) 
    name_df = pd.DataFrame({'name': name_list}) 
    all_df = pd.concat([name_df, config_df,summary_df], axis=1)

    all_df.to_csv("project.csv")
    df_all_eval.to_csv("all_eval_runs.csv",index=False)
    df_all_train.to_csv("all_train_runs.csv",index=False)


df_all_eval = pd.read_csv("all_eval_runs.csv")
df_all_train = pd.read_csv("all_train_runs.csv")
df_all_eval.sort_values(by="timesteps_total",inplace=True,ascending=True)
df_all_train.sort_values(by="timesteps_total",inplace=True,ascending=True)

min_steps = 1000000
def get_smooth_results(df_all_final):
    if 'evaluation/custom_metrics/percentage_complete_mean' in df_all_final.columns:
        df_all_final['perc_completion_mean'] = df_all_final['evaluation/custom_metrics/percentage_complete_mean']
        df_all_final['normalized_mean'] = df_all_final['evaluation/custom_metrics/episode_score_normalized_mean']
    return df_all_final

def get_final_results(df_all_final,min_steps=min_steps):
    df_all_final = df_all_final[df_all_final['timesteps_total'] > min_steps]
    if 'evaluation/custom_metrics/percentage_complete_mean' in df_all_final.columns:
        df_all_final['perc_completion_mean'] = df_all_final['evaluation/custom_metrics/percentage_complete_mean']
        df_all_final['normalized_mean'] = df_all_final['evaluation/custom_metrics/episode_score_normalized_mean']

        df_all_final_results = df_all_final[["run","group","perc_completion_mean","normalized_mean"]].groupby("run").max().groupby("group").aggregate([np.mean,np.std]).reset_index()
    
    elif 'custom_metrics/percentage_complete_mean' in df_all_final.columns:

        df_all_final_results = df_all_final[["run","group","custom_metrics/percentage_complete_mean","custom_metrics/episode_score_normalized_mean"]].groupby(by=["run","group"])

        norm_mean = df_all_final_results.apply(lambda x: max(smooth(x["custom_metrics/episode_score_normalized_mean"].to_list())))
        perc_completion_mean = df_all_final_results.apply(lambda x: max(smooth(x["custom_metrics/percentage_complete_mean"].to_list())))
        df_all_final = pd.concat([perc_completion_mean,norm_mean],axis=1)
        
        df_all_final_results = df_all_final.groupby("group").aggregate([np.mean,np.std]).reset_index()  
    
    
    return df_all_final,df_all_final_results


df_all_eval_final_results = get_final_results(df_all_eval)
df_all_train_final_results = get_final_results(df_all_train)

df_all_eval_final_results[0].to_csv('evaluation_results.csv')
df_all_train_final_results[0].to_csv('training_results.csv')

df_all_eval_final_results[1].to_csv('eval_results_group.csv',index=False)
df_all_train_final_results[1].to_csv('train_results_group.csv',index=False)