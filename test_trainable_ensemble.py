# import argparse
# import gymnasium
# from gymnasium.envs.registration import register
# import numpy as np
# import torch
# from datetime import datetime
# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # --- Local Imports ---
# from agents.ensemble_agent import EnsembleAgent
# from utils.state_management_closed_loop_ensemble import StateRewardManager
# from utils.safety2_closed_loop import SafetyLayer
# from utils.realistic_scenario import RealisticMealScenario
# from simglucose.patient.t1dpatient import T1DPatient

# def get_cohort_patients(cohort_name):
#     if cohort_name == 'adult':
#         return [f'adult#{i:03d}' for i in range(1, 11)]
#     elif cohort_name == 'adolescent':
#         return [f'adolescent#{i:03d}' for i in range(1, 11)]
#     elif cohort_name == 'child':
#         return [f'child#{i:03d}' for i in range(1, 11)]
#     else:
#         raise ValueError(f"Unknown cohort: {cohort_name}")

# def evaluate_trainable_ensemble(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\n{'='*70}")
#     print(f"EVALUATING CLINICAL META-CONTROLLER ON {args.cohort.upper()} COHORT")
#     print(f"Model Path: {args.model_path}")
#     print(f"{'='*70}\n")

#     # 1. SETUP
#     seed = args.seed
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     hyperparameters = {
#         'max_timesteps_per_episode': 288 # 1 Day
#     }

#     results_dir = f'./results/eval_meta_ensemble_{args.cohort}'
#     os.makedirs(results_dir, exist_ok=True)

#     state_dim = 4
#     action_dim = 1

#     # 2. LOAD AGENT
#     if not os.path.exists(args.model_path):
#         raise FileNotFoundError(f"Could not find model at {args.model_path}")
        
#     agent = EnsembleAgent(state_dim, action_dim, max_action=1.0, device=device)
#     agent.load(args.model_path)
    
#     # Set all networks to evaluation mode
#     agent.sac_agent.actor.eval()
#     agent.td3_agent.actor.eval()
#     agent.meta_controller.eval()

#     manager = StateRewardManager(state_dim)
    
#     # NEW: Pass cohort argument to the Safety Layer
#     safety_layer = SafetyLayer(cohort=args.cohort)

#     cohort_patients = get_cohort_patients(args.cohort)
#     all_metrics = []

#     now = datetime.now()
#     start_time = datetime.combine(now.date(), datetime.min.time())

#     # 3. EVALUATION LOOP (Patient by Patient)
#     for patient_name in cohort_patients:
#         print(f"Evaluating Patient: {patient_name}...")
        
#         patient_obj = T1DPatient.withName(patient_name)
#         bw = patient_obj._params['BW']
        
#         eval_scenario = RealisticMealScenario(start_time=start_time, patient=patient_obj, seed=seed)
        
#         env_id = f'simglucose/eval-meta-{patient_name.replace("#", "-")}-v0'
#         try:
#             register(
#                 id=env_id,
#                 entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
#                 max_episode_steps=hyperparameters['max_timesteps_per_episode'],
#                 kwargs={"patient_name": patient_name, "custom_scenario": eval_scenario}
#             )
#         except gymnasium.error.Error:
#             pass
            
#         env = gymnasium.make(env_id)

#         # --- WARMUP PHASE ---
#         obs_array, _ = env.reset(seed=seed)
#         manager.reset()
#         for _ in range(hyperparameters['max_timesteps_per_episode']):
#             u_state = manager.get_full_state(obs_array[0], bw)
#             n_state = manager.get_normalized_state(u_state)
            
#             action, _, _ = agent.select_action(n_state, evaluate=True)
            
#             # --- FINAL TUNED EXTENDED BOLUS LIMITS ---
#             if args.cohort == 'child':
#                 clinical_max = 0.3      
#             elif args.cohort == 'adolescent':
#                 clinical_max = 0.75      # Slightly raised to handle spikes
#             else:
#                 clinical_max = 1.5       # Raised to fix Adult Hyperglycemia
                
#             normalized_action = (action[0] + 1.0) / 2.0 
            
#             if args.cohort == 'adult':
#                 # Linear math for adults so they get the full dose they ask for
#                 insulin_dose = np.array([normalized_action * clinical_max])
#             else:
#                 # Squared math for precise micro-dosing in kids/teens
#                 insulin_dose = np.array([(normalized_action ** 2) * clinical_max])

#             safe_action = safety_layer.apply(insulin_dose, u_state)
            
#             manager.insulin_history.append(safe_action[0])
#             obs_array, _, done, truncated, _ = env.step(safe_action)
#             if done or truncated: break

#         # --- OFFICIAL EVALUATION PHASE ---
#         obs_array, _ = env.reset(seed=seed + 1)
#         manager.reset()
        
#         glucose_history = [obs_array[0]]
#         insulin_history = []
#         sac_weights_history = []
        
#         unnormalized_state = manager.get_full_state(obs_array[0], bw)
#         current_state = manager.get_normalized_state(unnormalized_state)

#         for t in range(hyperparameters['max_timesteps_per_episode']):
#             action, w_sac, w_td3 = agent.select_action(current_state, evaluate=True)
#             sac_weights_history.append(w_sac)
            
#             # --- FINAL TUNED EXTENDED BOLUS LIMITS ---
#             if args.cohort == 'child':
#                 clinical_max = 0.3     
#             elif args.cohort == 'adolescent':
#                 clinical_max = 0.75      # Slightly raised from 0.5 
#             else:
#                 clinical_max = 1.5      # Raised from 1.0 to fix Adult Hyperglycemia
                
#             normalized_action = (action[0] + 1.0) / 2.0  # Use action[0] in test script
            
#             # For adults, we remove the squaring math so they get linear, powerful boluses
#             if args.cohort == 'adult':
#                 insulin_dose = np.array([normalized_action * clinical_max])
#             else:
#                 # Children and Adolescents keep the squared math for micro-dosing precision
#                 insulin_dose = np.array([(normalized_action ** 2) * clinical_max])

            
#             safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
            
#             insulin_history.append(safe_action[0])
#             manager.insulin_history.append(safe_action[0])
            
#             next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
            
#             glucose_history.append(next_obs_array[0])
            
#             unnormalized_state = manager.get_full_state(next_obs_array[0], bw)
#             current_state = manager.get_normalized_state(unnormalized_state)
            
#             if terminated or truncated:
#                 break
        
#         env.close()

#         if len(insulin_history) < len(glucose_history):
#             insulin_history.append(0.0)

#         glucose_arr = np.array(glucose_history)
        
#         # Calculate Clinical Metrics
#         tir = np.sum((glucose_arr >= 70) & (glucose_arr <= 180)) / len(glucose_arr) * 100
#         hypo = np.sum(glucose_arr < 70) / len(glucose_arr) * 100
#         severe_hypo = np.sum(glucose_arr < 54) / len(glucose_arr) * 100
#         hyper = np.sum(glucose_arr > 180) / len(glucose_arr) * 100
#         mean_bg = np.mean(glucose_arr)
#         avg_sac_weight = np.mean(sac_weights_history)
        
#         all_metrics.append({
#             'Patient': patient_name,
#             'TIR (%)': round(tir, 2),
#             'Hypo (%)': round(hypo, 2),
#             'Severe Hypo (%)': round(severe_hypo, 2),
#             'Hyper (%)': round(hyper, 2),
#             'Mean BG': round(mean_bg, 2),
#             'Avg SAC Wt': round(avg_sac_weight, 2)
#         })

#         # --- PLOTTING ---
#         time_axis = np.arange(len(glucose_history)) * 5 / 60 # Convert to hours
        
#         fig, ax1 = plt.subplots(figsize=(12, 6))
#         ax1.set_xlabel('Time (Hours)', fontsize=12)
#         ax1.set_ylabel('Blood Glucose (mg/dL)', color='blue', fontsize=12)
#         ax1.plot(time_axis, glucose_history, color='blue', label='Glucose', linewidth=2)
#         ax1.axhline(y=180, color='red', linestyle='--', alpha=0.5, label='Hyper Limit (180)')
#         ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Hypo Limit (70)')
#         ax1.fill_between(time_axis, 70, 180, color='green', alpha=0.1, label='Target Range')
#         ax1.tick_params(axis='y', labelcolor='blue')
#         ax1.set_ylim(40, max(300, max(glucose_history) + 20))

#         ax2 = ax1.twinx()
#         ax2.set_ylabel('Insulin Dose (U / 5min)', color='orange', fontsize=12)
#         ax2.bar(time_axis, insulin_history, width=0.08, color='orange', alpha=0.6, label='Insulin')
#         ax2.tick_params(axis='y', labelcolor='orange')
#         ax2.set_ylim(0, max(2.0, max(insulin_history) * 2))

#         fig.suptitle(f'Meta-Ensemble: {patient_name} (TIR: {tir:.1f}% | Avg SAC Wt: {avg_sac_weight:.2f})', fontsize=14, weight='bold')
#         fig.tight_layout()
        
#         plot_path = os.path.join(results_dir, f'plot_{patient_name.replace("#", "-")}.png')
#         plt.savefig(plot_path, dpi=300)
#         plt.close()

#     # 4. SAVE SUMMARY CSV
#     df = pd.DataFrame(all_metrics)
#     summary_path = os.path.join(results_dir, 'meta_ensemble_evaluation_summary.csv')
#     df.to_csv(summary_path, index=False)
    
#     print(f"\n{'='*70}")
#     print("EVALUATION COMPLETE")
#     print(f"{'='*70}")
#     print(df.to_string(index=False))
    
#     # --- NEW: CALCULATE AND PRINT COHORT AVERAGES ---
#     print(f"\n{'-'*70}")
#     print(f"🎯 COHORT AVERAGES ({args.cohort.upper()}):")
#     print(f"{'-'*70}")
#     print(f"TIR (70-180)  : {df['TIR (%)'].mean():.2f} %")
#     print(f"Hypo (<70)    : {df['Hypo (%)'].mean():.2f} %")
#     print(f"Sev. Hypo (<54): {df['Severe Hypo (%)'].mean():.2f} %")
#     print(f"Hyper (>180)  : {df['Hyper (%)'].mean():.2f} %")
#     print(f"Mean BG       : {df['Mean BG'].mean():.2f} mg/dL")
#     print(f"Avg SAC Wt    : {df['Avg SAC Wt'].mean():.2f}")
#     print(f"{'-'*70}\n")
    
#     print(f"Plots and summary saved to: {results_dir}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Evaluate Trainable Meta-Controller Ensemble")
#     parser.add_argument('--cohort', type=str, choices=['child', 'adolescent', 'adult'], required=True)
#     parser.add_argument('--model_path', type=str, required=True, help="Path to best_model.pth")
#     parser.add_argument('--seed', type=int, default=100, help="Random seed for deterministic evaluation")
    
#     args = parser.parse_args()
#     evaluate_trainable_ensemble(args)


import argparse
import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import uuid
from collections import namedtuple

# --- Local Imports ---
from agents.ensemble_agent import EnsembleAgent
from utils.state_management_closed_loop_ensemble import StateRewardManager
from utils.safety2_closed_loop import SafetyLayer
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import Scenario

# --- THE BULLETPROOF FIXED SCENARIO ---
PATIENT_ACTION = namedtuple('patient_action', ['meal', 'insulin'])

class FixedMealScenario(Scenario):
    """A custom scenario that completely bypasses simglucose defaults to force exact meals."""
    def __init__(self, start_time):
        super().__init__(start_time)
        # Fixed deterministic meals: 07:00 (45g), 12:00 (70g), 18:00 (80g)
        self.meals = [(7 * 60, 45), (12 * 60, 70), (18 * 60, 80)]
        
    def get_action(self, t):
        current_time_in_day = t.hour * 60 + t.minute
        for meal_time, meal_amount in self.meals:
            if abs(current_time_in_day - meal_time) < 1e-9:
                return PATIENT_ACTION(meal=meal_amount, insulin=0)
        return PATIENT_ACTION(meal=0, insulin=0)
        
    def reset(self):
        pass # Prevent any random generation on reset

def get_cohort_patients(cohort_name):
    if cohort_name == 'adult': return [f'adult#{i:03d}' for i in range(1, 11)]
    elif cohort_name == 'adolescent': return [f'adolescent#{i:03d}' for i in range(1, 11)]
    elif cohort_name == 'child': return [f'child#{i:03d}' for i in range(1, 11)]
    else: raise ValueError(f"Unknown cohort: {cohort_name}")

def evaluate_trainable_ensemble(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"DETERMINISTIC EVALUATION ON {args.cohort.upper()} COHORT")
    print(f"Model Path: {args.model_path}")
    print(f"{'='*70}\n")

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    hyperparameters = {'max_timesteps_per_episode': 288}
    
    # --- REVERTED TO OLD SAVE PATH ---
    results_dir = f'./results/eval_meta_ensemble_{args.cohort}'
    os.makedirs(results_dir, exist_ok=True)

    state_dim, action_dim = 4, 1

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Could not find model at {args.model_path}")
        
    agent = EnsembleAgent(state_dim, action_dim, max_action=1.0, device=device)
    agent.load(args.model_path)
    
    agent.sac_agent.actor.eval()
    agent.td3_agent.actor.eval()
    agent.meta_controller.eval()

    manager = StateRewardManager(state_dim)
    safety_layer = SafetyLayer(cohort=args.cohort)
    cohort_patients = get_cohort_patients(args.cohort)
    all_metrics = []

    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    for patient_name in cohort_patients:
        print(f"Evaluating Patient: {patient_name}...")
        patient_obj = T1DPatient.withName(patient_name)
        bw = patient_obj._params['BW']
        
        # Apply our new bulletproof scenario
        eval_scenario = FixedMealScenario(start_time=start_time)
        
        # Add a UUID to the env_id to completely bypass the Gymnasium cache bug
        unique_id = uuid.uuid4().hex[:6]
        env_id = f'simglucose/eval-fixed-{patient_name.replace("#", "-")}-{unique_id}-v0'
        
        register(
            id=env_id,
            entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
            max_episode_steps=hyperparameters['max_timesteps_per_episode'],
            kwargs={"patient_name": patient_name, "custom_scenario": eval_scenario}
        )
            
        env = gymnasium.make(env_id)

        # --- WARMUP PHASE ---
        obs_array, _ = env.reset(seed=seed)
        manager.reset()
        for _ in range(hyperparameters['max_timesteps_per_episode']):
            u_state = manager.get_full_state(obs_array[0], bw)
            n_state = manager.get_normalized_state(u_state)
            action, _, _ = agent.select_action(n_state, evaluate=True)
            
            if args.cohort == 'child': clinical_max = 0.25      
            elif args.cohort == 'adolescent': clinical_max = 0.75      
            else: clinical_max = 1.5       
                
            normalized_action = (action[0] + 1.0) / 2.0 
            if args.cohort == 'adult': insulin_dose = np.array([normalized_action * clinical_max])
            else: insulin_dose = np.array([(normalized_action ** 2) * clinical_max])

            safe_action = safety_layer.apply(insulin_dose, u_state)
            manager.insulin_history.append(safe_action[0])
            obs_array, _, done, truncated, _ = env.step(safe_action)
            if done or truncated: break

        # --- OFFICIAL EVALUATION PHASE ---
        obs_array, _ = env.reset(seed=seed + 1)
        manager.reset()
        
        glucose_history = [obs_array[0]]
        insulin_history = []
        sac_weights_history = []
        
        u_state = manager.get_full_state(obs_array[0], bw)
        c_state = manager.get_normalized_state(u_state)

        for t in range(hyperparameters['max_timesteps_per_episode']):
            action, w_sac, w_td3 = agent.select_action(c_state, evaluate=True)
            sac_weights_history.append(w_sac)
            
            if args.cohort == 'child': clinical_max = 0.25     
            elif args.cohort == 'adolescent': clinical_max = 0.75      
            else: clinical_max = 1.5      
                
            normalized_action = (action[0] + 1.0) / 2.0  
            if args.cohort == 'adult': insulin_dose = np.array([normalized_action * clinical_max])
            else: insulin_dose = np.array([(normalized_action ** 2) * clinical_max])
            
            safe_action = safety_layer.apply(insulin_dose, u_state)
            insulin_history.append(safe_action[0])
            manager.insulin_history.append(safe_action[0])
            
            next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
            glucose_history.append(next_obs_array[0])
            
            u_state = manager.get_full_state(next_obs_array[0], bw)
            c_state = manager.get_normalized_state(u_state)
            
            if terminated or truncated: break
        
        env.close()
        if len(insulin_history) < len(glucose_history): insulin_history.append(0.0)

        glucose_arr = np.array(glucose_history)
        
        tir = np.sum((glucose_arr >= 70) & (glucose_arr <= 180)) / len(glucose_arr) * 100
        hypo = np.sum(glucose_arr < 70) / len(glucose_arr) * 100
        severe_hypo = np.sum(glucose_arr < 54) / len(glucose_arr) * 100
        hyper = np.sum(glucose_arr > 180) / len(glucose_arr) * 100
        mean_bg = np.mean(glucose_arr)
        avg_sac_weight = np.mean(sac_weights_history)
        
        all_metrics.append({
            'Patient': patient_name,
            'TIR (%)': round(tir, 2),
            'Hypo (%)': round(hypo, 2),
            'Severe Hypo (%)': round(severe_hypo, 2),
            'Hyper (%)': round(hyper, 2),
            'Mean BG': round(mean_bg, 2),
            'Avg SAC Wt': round(avg_sac_weight, 2)
        })

        # --- PERFECTLY ALIGNED PLOTTING ---
        time_axis_minutes = np.arange(len(glucose_history)) * 5
        clean_patient_name = patient_name.replace("#", "-")

        fig, ax1 = plt.subplots(figsize=(15, 7))
        color = 'tab:blue'
        ax1.set_xlabel('Time (minutes)', fontsize=12)
        ax1.set_ylabel('Blood Glucose (mg/dL)', color=color, fontsize=12)
        ax1.plot(time_axis_minutes, glucose_history, color=color, label='Blood Glucose', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(which='major', axis='y', linestyle='--', alpha=0.7)
        ax1.axhline(y=180, color='r', linestyle=':', label='Hyper Threshold (180)', linewidth=2)
        ax1.axhline(y=70, color='orange', linestyle=':', label='Hypo Threshold (70)', linewidth=2)
        ax1.fill_between(time_axis_minutes, 70, 180, alpha=0.1, color='green', label='Target Range')
        
        ax2 = ax1.twinx()
        color = 'tab:gray'
        ax2.set_ylabel('Insulin Dose (U / 5min)', color=color, fontsize=12)
        ax2.bar(time_axis_minutes, insulin_history, width=5, color=color, alpha=0.6, label='Insulin Dose')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Extract straight from our new FixedMealScenario
        meal_labels_seen = set()
        for meal_time, meal_amount in eval_scenario.meals:
            label = f'Meal ({meal_amount}g)'
            if label not in meal_labels_seen:
                ax1.axvline(x=meal_time, color='black', linestyle='--', label=label, linewidth=1.5)
                meal_labels_seen.add(label)
            else:
                ax1.axvline(x=meal_time, color='black', linestyle='--', linewidth=1.5)
        
        fig.suptitle(f'Ensemble SAC-TD3 Performance: {patient_name} (TIR: {tir:.1f}%)', fontsize=16, weight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=10)
        
        # --- REVERTED TO OLD PLOT NAME ---
        plot_path = os.path.join(results_dir, f'evaluation_plot_{clean_patient_name}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()

    df = pd.DataFrame(all_metrics)
    # --- REVERTED TO OLD CSV NAME ---
    summary_path = os.path.join(results_dir, 'meta_ensemble_evaluation_summary.csv')
    df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*70}\nDETERMINISTIC EVALUATION COMPLETE\n{'='*70}")
    print(df.to_string(index=False))
    print(f"\n{'-'*70}\n🎯 COHORT AVERAGES ({args.cohort.upper()}):\n{'-'*70}")
    print(f"TIR (70-180)  : {df['TIR (%)'].mean():.2f} %")
    print(f"Hypo (<70)    : {df['Hypo (%)'].mean():.2f} %")
    print(f"Sev. Hypo (<54): {df['Severe Hypo (%)'].mean():.2f} %")
    print(f"Hyper (>180)  : {df['Hyper (%)'].mean():.2f} %")
    print(f"Mean BG       : {df['Mean BG'].mean():.2f} mg/dL")
    print(f"Avg SAC Wt    : {df['Avg SAC Wt'].mean():.2f}\n{'-'*70}\n")
    print(f"Plots and summary saved to: {results_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Trainable Meta-Controller Ensemble")
    parser.add_argument('--cohort', type=str, choices=['child', 'adolescent', 'adult'], required=True)
    parser.add_argument('--model_path', type=str, required=True, help="Path to best_model.pth")
    parser.add_argument('--seed', type=int, default=100)
    
    args = parser.parse_args()
    evaluate_trainable_ensemble(args)