# import argparse
# import gymnasium
# from gymnasium.envs.registration import register
# import numpy as np
# import torch
# from datetime import datetime
# import os
# import pandas as pd

# # --- Local Imports ---
# from agents.ensemble_agent import EnsembleAgent
# from utils.state_management_closed_loop_ensemble import StateRewardManager
# from utils.safety2_closed_loop import SafetyLayer
# from utils.realistic_scenario import RealisticMealScenario
# from simglucose.patient.t1dpatient import T1DPatient

# # --- Report ---
# from simglucose.analysis.report import report


# # -------------------------------
# # Cohort helper
# # -------------------------------
# def get_cohort_patients(cohort):
#     if cohort == 'adult':
#         return [f'adult#{i:03d}' for i in range(1, 11)]
#     elif cohort == 'adolescent':
#         return [f'adolescent#{i:03d}' for i in range(1, 11)]
#     elif cohort == 'child':
#         return [f'child#{i:03d}' for i in range(1, 11)]


# # -------------------------------
# # Safe Risk
# # -------------------------------
# def compute_risk(bg):
#     bg = np.clip(bg, 40, 400)
#     f = 1.509 * ((np.log(bg))**1.084 - 5.381)
#     return 10 * (f**2)


# # -------------------------------
# # MAIN
# # -------------------------------
# def generate_report(args):

#     print("\n" + "="*70)
#     print(f"Generating REPORT for {args.model_name} on {args.cohort}")
#     print("="*70)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     agent = EnsembleAgent(4, 1, max_action=1.0, device=device)
#     agent.load(args.model_path)

#     # eval mode
#     agent.sac_agent.actor.eval()
#     agent.td3_agent.actor.eval()
#     agent.meta_controller.eval()

#     manager = StateRewardManager(4)
#     safety_layer = SafetyLayer(cohort=args.cohort)

#     patients = get_cohort_patients(args.cohort)

#     results_dir = f'./results/report_{args.model_name}_{args.cohort}'
#     os.makedirs(results_dir, exist_ok=True)

#     all_dfs = {}

#     now = datetime.now()
#     start_time = datetime.combine(now.date(), datetime.min.time())

#     # -------------------------------
#     # PATIENT LOOP
#     # -------------------------------
#     for patient_name in patients:
#         print(f"Simulating {patient_name}...")

#         patient_obj = T1DPatient.withName(patient_name)
#         bw = patient_obj._params['BW']

#         scenario = RealisticMealScenario(start_time=start_time, patient=patient_obj, seed=42)

#         env_id = f'simglucose/report-{patient_name.replace("#", "-")}-v0'

#         try:
#             register(
#                 id=env_id,
#                 entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
#                 max_episode_steps=288,
#                 kwargs={"patient_name": patient_name, "custom_scenario": scenario}
#             )
#         except:
#             pass

#         env = gymnasium.make(env_id)

#         obs_array, _ = env.reset(seed=42)
#         manager.reset()

#         glucose_history = []
#         insulin_history = []

#         u_state = manager.get_full_state(obs_array[0], bw)
#         state = manager.get_normalized_state(u_state)

#         for _ in range(288):

#             action, _, _ = agent.select_action(state, evaluate=True)

#             normalized_action = (action[0] + 1) / 2

#             if args.cohort == 'adult':
#                 insulin = normalized_action * 2.0
#             elif args.cohort == 'adolescent':
#                 insulin = (normalized_action ** 2) * 0.75
#             else:
#                 insulin = (normalized_action ** 2) * 0.25

#             safe_action = safety_layer.apply(np.array([insulin]), u_state)

#             # --- SAFE BG ---
#             bg = float(obs_array[0])
#             if np.isnan(bg) or bg <= 0:
#                 bg = 110.0
#             bg = np.clip(bg, 40, 400)

#             glucose_history.append(bg)
#             insulin_history.append(float(safe_action[0]))

#             obs_array, _, done, truncated, _ = env.step(safe_action)

#             u_state = manager.get_full_state(obs_array[0], bw)
#             state = manager.get_normalized_state(u_state)

#             if done or truncated:
#                 break

#         env.close()

#         # -------------------------------
#         # FIX LENGTHS
#         # -------------------------------
#         min_len = min(len(glucose_history), len(insulin_history))
#         glucose_history = glucose_history[:min_len]
#         insulin_history = insulin_history[:min_len]

#         glucose_arr = np.array(glucose_history)
#         insulin_arr = np.array(insulin_history)

#         # -------------------------------
#         # CREATE RLAP FORMAT DF
#         # -------------------------------
#         time_index = pd.date_range(
#             start="2020-01-01",
#             periods=len(glucose_arr),
#             freq="5min"
#         )

#         df = pd.DataFrame({
#             'BG': glucose_arr,
#             'CGM': glucose_arr,
#             'CHO': np.zeros(len(glucose_arr)),
#             'insulin': insulin_arr
#         }, index=time_index)

#         df['Risk'] = compute_risk(df['BG'].values)

#         all_dfs[patient_name] = df

#     # -------------------------------
#     # ALIGN ALL DATA (CRITICAL FIX)
#     # -------------------------------
#     min_len = min(len(df) for df in all_dfs.values())

#     for k in all_dfs:
#         all_dfs[k] = all_dfs[k].iloc[:min_len]

#     df_concat = pd.concat(all_dfs.values(), keys=all_dfs.keys())

#     # Final cleanup
#     df_concat = df_concat.replace([np.inf, -np.inf], np.nan)
#     df_concat = df_concat.fillna(method='ffill').fillna(method='bfill')

#     # -------------------------------
#     # GENERATE REPORT
#     # -------------------------------
#     report(df_concat, save_path=results_dir)

#     print("\n✅ REPORT GENERATED SUCCESSFULLY")
#     print(f"Saved at: {results_dir}")


# # -------------------------------
# # ENTRY
# # -------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--cohort', type=str, required=True)
#     parser.add_argument('--model_path', type=str, required=True)
#     parser.add_argument('--model_name', type=str, required=True)

#     args = parser.parse_args()

#     generate_report(args)


"""
generate_report_plots.py
========================
Generates RLAP-style clinical analysis plots by feeding simulation data
directly into simglucose.analysis.report.report().

The report() function produces four figures automatically:
  - BG_trace.png     (ensemble BG + CGM + CHO over time)
  - zone_stats.png   (% time in BG ranges per patient)
  - risk_stats.png   (LBGI / HBGI / Risk Index bar chart)
  - CVGA.png         (Control Variability Grid Analysis)

It also saves:
  - performance_stats.csv
  - risk_trace.csv
  - CVGA_stats.csv

Usage
-----
  python generate_report_plots.py --cohort adult \\
         --model_path ./models/trainable_ensemble_adult/best_model.pth

Optional flags
--------------
  --seed      int   Evaluation seed (default 100)
  --days      int   Number of simulation days to run (default 10)
  --save_csv        Also save per-patient raw glucose CSV files
"""

# import argparse
# import os
# import matplotlib
# matplotlib.use("Agg")   # must be set BEFORE importing pyplot (headless)
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta

# import gymnasium
# from gymnasium.envs.registration import register
# import numpy as np
# import pandas as pd
# import torch

# # ── The one and only simglucose import we need ─────────────────────────────
# from simglucose.analysis.report import report

# # ── project-local imports ──────────────────────────────────────────────────
# from agents.ensemble_agent import EnsembleAgent
# from utils.state_management_closed_loop_ensemble import StateRewardManager
# from utils.safety2_closed_loop import SafetyLayer
# from utils.realistic_scenario import RealisticMealScenario
# from simglucose.patient.t1dpatient import T1DPatient

# # ═══════════════════════════════════════════════════════════════════════════
# # CONSTANTS
# # ═══════════════════════════════════════════════════════════════════════════
# STEPS_PER_DAY = 288      # 5-min steps × 288 = 24 h
# SAMPLE_PERIOD = 5        # minutes per step


# # ═══════════════════════════════════════════════════════════════════════════
# # HELPERS
# # ═══════════════════════════════════════════════════════════════════════════

# def get_cohort_patients(cohort_name: str) -> list:
#     mapping = {"adult": "adult", "adolescent": "adolescent", "child": "child"}
#     if cohort_name not in mapping:
#         raise ValueError(f"Unknown cohort: {cohort_name}")
#     return [f"{mapping[cohort_name]}#{i:03d}" for i in range(1, 11)]


# def build_action(raw_action: np.ndarray, cohort: str) -> np.ndarray:
#     """Mirror the exact action-scaling logic from train / test scripts."""
#     clinical_max = {"child": 0.25, "adolescent": 0.75, "adult": 2.0}[cohort]
#     normalized = (raw_action[0] + 1.0) / 2.0
#     if cohort == "adult":
#         return np.array([normalized * clinical_max])
#     return np.array([(normalized ** 2) * clinical_max])


# # ═══════════════════════════════════════════════════════════════════════════
# # SIMULATION — run n_days for one patient, return BG / CGM / CHO arrays
# # ═══════════════════════════════════════════════════════════════════════════

# def run_patient(
#     agent, manager, safety_layer,
#     patient_name: str, cohort: str,
#     seed: int, n_days: int, start_time: datetime,
# ) -> dict:
#     patient_obj = T1DPatient.withName(patient_name)
#     bw = patient_obj._params["BW"]

#     all_bg, all_cgm, all_cho, all_insulin, all_times = [], [], [], [], []

#     for day in range(n_days):
#         day_seed  = seed + day * 100
#         day_start = start_time + timedelta(days=day)

#         meal_scenario = RealisticMealScenario(
#             start_time=day_start, patient=patient_obj, seed=day_seed
#         )

#         env_id = f"simglucose/report-{patient_name.replace('#', '-')}-d{day}-v0"
#         try:
#             register(
#                 id=env_id,
#                 entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
#                 max_episode_steps=STEPS_PER_DAY,
#                 kwargs={"patient_name": patient_name,
#                         "custom_scenario": meal_scenario},
#             )
#         except gymnasium.error.Error:
#             pass

#         env = gymnasium.make(env_id)
#         obs_array, _ = env.reset(seed=day_seed)
#         manager.reset()

#         u_state = manager.get_full_state(obs_array[0], bw)
#         c_state = manager.get_normalized_state(u_state)

#         # Build meal (CHO) map: step_index -> carb_g
#         meal_map = {}
#         if hasattr(meal_scenario, "scenario"):
#             for meal_time, carb_g in meal_scenario.scenario:
#                 delta_min = (meal_time - day_start).total_seconds() / 60
#                 step_idx  = int(delta_min // SAMPLE_PERIOD)
#                 meal_map[step_idx] = meal_map.get(step_idx, 0) + float(carb_g)

#         day_bg, day_cgm, day_cho, day_insulin, day_times = [], [], [], [], []

#         for t in range(STEPS_PER_DAY):
#             timestamp = day_start + timedelta(minutes=t * SAMPLE_PERIOD)

#             action, _, _ = agent.select_action(c_state, evaluate=True)
#             dose      = build_action(action, cohort)
#             safe_dose = safety_layer.apply(dose, u_state)
#             manager.insulin_history.append(safe_dose[0])

#             next_obs, _, terminated, truncated, _ = env.step(safe_dose)

#             day_bg.append(float(next_obs[0]))
#             day_cgm.append(float(next_obs[0]))   # gym wrapper: obs = CGM reading
#             day_cho.append(float(meal_map.get(t, 0.0)))
#             day_insulin.append(float(safe_dose[0]))
#             day_times.append(timestamp)

#             u_state = manager.get_full_state(next_obs[0], bw)
#             c_state = manager.get_normalized_state(u_state)

#             if terminated or truncated:
#                 break

#         all_bg.extend(day_bg)
#         all_cgm.extend(day_cgm)
#         all_cho.extend(day_cho)
#         all_insulin.extend(day_insulin)
#         all_times.extend(day_times)

#         env.close()

#     return {
#         "patient": patient_name,
#         "times":   all_times,
#         "BG":      np.array(all_bg),
#         "CGM":     np.array(all_cgm),
#         "CHO":     np.array(all_cho),
#         "insulin": np.array(all_insulin),
#     }


# # ═══════════════════════════════════════════════════════════════════════════
# # BUILD THE MULTIINDEX DATAFRAME that report() expects
# #
# #   report(df)  where df has:
# #     Index level 0 : patient_name  (e.g. "adult#001")
# #     Index level 1 : timestamp     (datetime)
# #     Columns       : BG, CGM, CHO  (and anything else is ignored)
# # ═══════════════════════════════════════════════════════════════════════════

# def build_report_df(all_data: list) -> pd.DataFrame:
#     """
#     Construct the MultiIndex DataFrame that simglucose's report() needs.

#     Structure required by report():
#         df.unstack(level=0).BG   -> wide DataFrame  [time × patient]
#         df.unstack(level=0).CGM  -> wide DataFrame  [time × patient]
#         df.unstack(level=0).CHO  -> wide DataFrame  [time × patient]

#     So the index must be (patient_name, timestamp) with the same timestamps
#     for every patient (same length simulation, same day_start offsets).
#     """
#     patient_frames = {}
#     for d in all_data:
#         patient_frames[d["patient"]] = pd.DataFrame(
#             {
#                 "BG":      d["BG"],
#                 "CGM":     d["CGM"],
#                 "CHO":     d["CHO"],
#                 "insulin": d["insulin"],
#             },
#             index=pd.DatetimeIndex(d["times"], name="time"),
#         )

#     # Align all patients to a common time index (shortest wins)
#     min_len = min(len(v) for v in patient_frames.values())
#     for key in patient_frames:
#         patient_frames[key] = patient_frames[key].iloc[:min_len]

#     # Build MultiIndex: outer = patient, inner = time
#     df = pd.concat(patient_frames, names=["patient", "time"])
#     return df


# # ═══════════════════════════════════════════════════════════════════════════
# # SUMMARY CSV (printed to console + written to disk)
# # ═══════════════════════════════════════════════════════════════════════════

# def _lbgi(bg: np.ndarray) -> float:
#     f = 1.509 * (np.log(np.maximum(bg, 1.0)) ** 1.084 - 5.381)
#     return float(np.mean(10.0 * (f ** 2) * (f < 0)))

# def _hbgi(bg: np.ndarray) -> float:
#     f = 1.509 * (np.log(np.maximum(bg, 1.0)) ** 1.084 - 5.381)
#     return float(np.mean(10.0 * (f ** 2) * (f > 0)))


# def save_summary(all_data: list, results_dir: str, cohort: str):
#     rows = []
#     for d in all_data:
#         bg = d["BG"]
#         rows.append({
#             "Patient":          d["patient"],
#             "TIR (%)":         round(100 * np.mean((bg >= 70) & (bg <= 180)), 2),
#             "Hypo (%)":        round(100 * np.mean(bg < 70),  2),
#             "Severe Hypo (%)": round(100 * np.mean(bg < 54),  2),
#             "Hyper (%)":       round(100 * np.mean(bg > 180), 2),
#             "Mean BG":         round(float(np.mean(bg)),      2),
#             "LBGI":            round(_lbgi(bg),               3),
#             "HBGI":            round(_hbgi(bg),               3),
#             "Risk Index":      round(_lbgi(bg) + _hbgi(bg),   3),
#         })
#     df   = pd.DataFrame(rows)
#     path = os.path.join(results_dir, "report_summary.csv")
#     df.to_csv(path, index=False)

#     print(f"\n  [OK] Summary CSV  ->  {path}")
#     print("\n" + "=" * 70)
#     print(df.to_string(index=False))
#     print(f"\n{'-'*70}")
#     print(f"COHORT AVERAGES  ({cohort.upper()})")
#     print(f"{'-'*70}")
#     for col in ["TIR (%)", "Hypo (%)", "Severe Hypo (%)",
#                 "Hyper (%)", "Mean BG", "LBGI", "HBGI", "Risk Index"]:
#         print(f"  {col:<22}: {df[col].mean():.3f}")
#     print(f"{'-'*70}\n")


# # ═══════════════════════════════════════════════════════════════════════════
# # MAIN
# # ═══════════════════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser(
#         description="Generate RLAP-style report plots via simglucose.analysis.report"
#     )
#     parser.add_argument("--cohort",     required=True,
#                         choices=["child", "adolescent", "adult"])
#     parser.add_argument("--model_path", required=True,
#                         help="Path to best_model.pth")
#     parser.add_argument("--seed",       type=int, default=100)
#     parser.add_argument("--days",       type=int, default=10,
#                         help="Number of simulation days (default 10)")
#     parser.add_argument("--save_csv",   action="store_true",
#                         help="Save per-patient raw CSV files as well")
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\n{'='*70}")
#     print(f"  REPORT PLOTS -- {args.cohort.upper()} | {args.model_path}")
#     print(f"  Days: {args.days}  |  Seed: {args.seed}  |  Device: {device}")
#     print(f"{'='*70}\n")

#     results_dir = f"./results/report_plots_{args.cohort}"
#     os.makedirs(results_dir, exist_ok=True)

#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     # ── Load agent ──────────────────────────────────────────────────────────
#     if not os.path.exists(args.model_path):
#         raise FileNotFoundError(f"Model not found: {args.model_path}")

#     agent = EnsembleAgent(4, 1, max_action=1.0, device=device)
#     agent.load(args.model_path)
#     agent.sac_agent.actor.eval()
#     agent.td3_agent.actor.eval()
#     agent.meta_controller.eval()

#     manager      = StateRewardManager(4)
#     safety_layer = SafetyLayer(cohort=args.cohort)

#     cohort_patients = get_cohort_patients(args.cohort)
#     start_time = datetime.combine(datetime.now().date(), datetime.min.time())

#     # ── Simulate all patients ───────────────────────────────────────────────
#     all_data = []
#     for patient_name in cohort_patients:
#         print(f"  Simulating {patient_name} ({args.days} days)...")
#         data = run_patient(
#             agent, manager, safety_layer,
#             patient_name, args.cohort,
#             args.seed, args.days, start_time,
#         )
#         all_data.append(data)

#         if args.save_csv:
#             csv_df = pd.DataFrame({
#                 "Time":    data["times"],
#                 "BG":      data["BG"],
#                 "CGM":     data["CGM"],
#                 "CHO":     data["CHO"],
#                 "insulin": data["insulin"],
#             })
#             csv_path = os.path.join(
#                 results_dir,
#                 f"{patient_name.replace('#', '-')}.csv",
#             )
#             csv_df.to_csv(csv_path, index=False)
#             print(f"    CSV -> {csv_path}")

#     # ── Build the MultiIndex DataFrame report() expects ─────────────────────
#     print("\n  Building MultiIndex DataFrame for simglucose report()...")
#     df = build_report_df(all_data)

#     print(f"  DataFrame shape : {df.shape}")
#     print(f"  Patients        : {df.index.get_level_values(0).unique().tolist()}")
#     print(f"  Columns         : {df.columns.tolist()}")

#     # ── Call simglucose.analysis.report.report() ────────────────────────────
#     # Signature: report(df, cgm_sensor=None, save_path=None)
#     #
#     # Internally it calls:
#     #   ensemblePlot(df)        -> BG_trace.png
#     #   percent_stats(BG)       -> zone_stats.png
#     #   risk_index_trace(BG)    -> risk_stats.png
#     #   CVGA(BG)                -> CVGA.png
#     #
#     # All figures are saved to save_path automatically.
#     # plt.show() is called inside report() — with Agg backend this is a no-op.
#     print(f"\n  Calling simglucose report()  ->  {results_dir}/\n")

#     results, ri_per_hour, zone_stats, figs, axes = report(
#         df,
#         cgm_sensor=None,
#         save_path=results_dir,
#     )

#     print(f"\n  [OK] BG_trace.png    saved")
#     print(f"  [OK] zone_stats.png  saved")
#     print(f"  [OK] risk_stats.png  saved")
#     print(f"  [OK] CVGA.png        saved")
#     print(f"  [OK] performance_stats.csv  saved")
#     print(f"  [OK] risk_trace.csv         saved")
#     print(f"  [OK] CVGA_stats.csv         saved")

#     # ── Print report() results to console ───────────────────────────────────
#     print(f"\n{'='*70}")
#     print("  PERFORMANCE STATS (from report())")
#     print(f"{'='*70}")
#     print(results.to_string())

#     print(f"\n{'='*70}")
#     print("  CVGA ZONE STATS")
#     print(f"{'='*70}")
#     print(zone_stats.to_string())

#     # ── Our own TIR / hypo / hyper summary ─────────────────────────────────
#     save_summary(all_data, results_dir, args.cohort)

#     print(f"All outputs saved to: {results_dir}/\n")


# if __name__ == "__main__":
#     main()






"""
generate_report_plots.py
========================
Generates RLAP-style clinical analysis plots and individual patient traces.
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")   # Headless plotting
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import pandas as pd
import torch

from simglucose.analysis.report import report
from agents.ensemble_agent import EnsembleAgent
from utils.state_management_closed_loop_ensemble import StateRewardManager
from utils.safety2_closed_loop import SafetyLayer
from utils.realistic_scenario import RealisticMealScenario
from simglucose.patient.t1dpatient import T1DPatient

# --- CONSTANTS ---
STEPS_PER_DAY = 288      # 5-min steps × 288 = 24 h
SAMPLE_PERIOD = 5        # minutes per step

def get_cohort_patients(cohort_name: str) -> list:
    mapping = {"adult": "adult", "adolescent": "adolescent", "child": "child"}
    if cohort_name not in mapping:
        raise ValueError(f"Unknown cohort: {cohort_name}")
    return [f"{mapping[cohort_name]}#{i:03d}" for i in range(1, 11)]

def build_action(raw_action: np.ndarray, cohort: str) -> np.ndarray:
    """Mirror the exact action-scaling logic from the Hybrid Architecture."""
    clinical_max = {"child": 0.25, "adolescent": 0.75, "adult": 1.5}[cohort] 
    normalized = (raw_action[0] + 1.0) / 2.0
    if cohort == "adult":
        return np.array([normalized * clinical_max])
    return np.array([(normalized ** 2) * clinical_max])

# --- RLAP PLOTTING INTEGRATION ---
def plot_individual_patient(df, patient_name, save_dir):
    """
    Generates a 4-panel plot (BG, CHO, Insulin, Risk) for a single patient.
    """
    df.index = pd.to_datetime(df.index)

    # Calculate Risk Index dynamically for the plot
    f = 1.509 * (np.log(np.maximum(df['BG'], 1.0)) ** 1.084 - 5.381)
    df['Risk'] = 10.0 * (f ** 2)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    
    # 1. Blood Glucose & CGM
    axes[0].plot(df.index, df['BG'], label='BG', color='black', linewidth=1.5)
    axes[0].plot(df.index, df['CGM'], label='CGM', color='orange', linestyle='--', alpha=0.8)
    axes[0].axhspan(70, 180, color='limegreen', alpha=0.3, label='Target (70-180)')
    axes[0].axhspan(0, 70, color='red', alpha=0.15, label='Hypo (<70)')
    axes[0].axhspan(180, 400, color='red', alpha=0.15, label='Hyper (>180)')
    axes[0].set_ylim(40, max(350, df['BG'].max() + 20)) 
    axes[0].set_ylabel('Glucose (mg/dL)', fontsize=10, weight='bold')
    axes[0].legend(loc='upper right', fontsize='small', ncol=2)
    axes[0].set_title(f"Clinical Trace: {patient_name}", fontsize=14, weight='bold')
    axes[0].grid(True, alpha=0.3)

    # 2. CHO (Carbs) - width=0.003 is approx 4.3 minutes on a datetime axis
    axes[1].bar(df.index, df['CHO'], width=0.003, color='tab:orange', label='CHO Intake')
    axes[1].set_ylabel('CHO (g)', fontsize=10, weight='bold')
    axes[1].set_ylim(0, max(10, df['CHO'].max() * 1.2)) 
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # 3. Insulin
    axes[2].bar(df.index, df['insulin'], width=0.003, color='tab:blue', alpha=0.7, label='Insulin Dose')
    axes[2].set_ylabel('Insulin (U)', fontsize=10, weight='bold')
    axes[2].set_ylim(0, max(1.0, df['insulin'].max() * 1.2))
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # 4. Risk
    axes[3].plot(df.index, df['Risk'], color='tab:red', label='Risk Index')
    axes[3].set_ylabel('Risk', fontsize=10, weight='bold')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    
    plt.xlabel('Time', fontsize=12, weight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{patient_name}_RLAP_trace.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig) 
    print(f"    [OK] RLAP Trace -> {save_path}")

def run_patient(
    agent, manager, safety_layer,
    patient_name: str, cohort: str,
    seed: int, n_days: int, start_time: datetime,
) -> dict:
    patient_obj = T1DPatient.withName(patient_name)
    bw = patient_obj._params["BW"]

    total_steps = n_days * STEPS_PER_DAY
    all_bg = np.zeros(total_steps)
    all_cgm = np.zeros(total_steps)
    all_cho = np.zeros(total_steps)
    all_insulin = np.zeros(total_steps)
    all_times = []

    global_step = 0

    for day in range(n_days):
        day_seed  = seed + day * 100
        day_start = start_time + timedelta(days=day)

        meal_scenario = RealisticMealScenario(
            start_time=day_start, patient=patient_obj, seed=day_seed
        )

        env_id = f"simglucose/report-{patient_name.replace('#', '-')}-d{day}-v0"
        try:
            register(
                id=env_id,
                entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
                max_episode_steps=STEPS_PER_DAY,
                kwargs={"patient_name": patient_name, "custom_scenario": meal_scenario},
            )
        except gymnasium.error.Error:
            pass

        env = gymnasium.make(env_id)
        
        # 1. Reset the environment FIRST (This triggers meal_scenario.reset() internally)
        obs_array, _ = env.reset(seed=day_seed)
        manager.reset()

        # 2. THE CHO FIX: Now that the scenario is locked for the day, extract self.meals
        meal_map = {}
        if hasattr(meal_scenario, "meals"):
            for meal_time_min, carb_g in meal_scenario.meals:
                # meal_time_min is minutes from midnight. Divide by 5 to get step_idx
                step_idx = int(meal_time_min // SAMPLE_PERIOD)
                meal_map[step_idx] = meal_map.get(step_idx, 0.0) + float(carb_g)

        u_state = manager.get_full_state(obs_array[0], bw)
        c_state = manager.get_normalized_state(u_state)
        
        last_valid_bg = float(obs_array[0])
        is_dead = False 

        for t in range(STEPS_PER_DAY):
            timestamp = day_start + timedelta(minutes=t * SAMPLE_PERIOD)
            all_times.append(timestamp)

            if is_dead:
                # Pad array to prevent CVGA NaN errors
                all_bg[global_step] = last_valid_bg
                all_cgm[global_step] = last_valid_bg
                all_cho[global_step] = 0.0
                all_insulin[global_step] = 0.0
            else:
                action, _, _ = agent.select_action(c_state, evaluate=True)
                dose      = build_action(action, cohort)
                safe_dose = safety_layer.apply(dose, u_state)
                manager.insulin_history.append(safe_dose[0])

                next_obs, _, terminated, truncated, _ = env.step(safe_dose)

                # 3. Read the exact carbs for this 5-minute window from our locked map
                cho_val = meal_map.get(t, 0.0)

                last_valid_bg = float(next_obs[0])

                all_bg[global_step] = last_valid_bg
                all_cgm[global_step] = last_valid_bg
                all_cho[global_step] = cho_val
                all_insulin[global_step] = float(safe_dose[0])

                u_state = manager.get_full_state(next_obs[0], bw)
                c_state = manager.get_normalized_state(u_state)

                if terminated or truncated:
                    is_dead = True 

            global_step += 1
        env.close()

    return {
        "patient": patient_name,
        "times":   all_times,
        "BG":      all_bg,
        "CGM":     all_cgm,
        "CHO":     all_cho,
        "insulin": all_insulin,
    }

def build_report_df(all_data: list) -> pd.DataFrame:
    patient_frames = {}
    for d in all_data:
        patient_frames[d["patient"]] = pd.DataFrame(
            {
                "BG":      d["BG"],
                "CGM":     d["CGM"],
                "CHO":     d["CHO"],
                "insulin": d["insulin"],
            },
            index=pd.DatetimeIndex(d["times"], name="time"),
        )
    df = pd.concat(patient_frames, names=["patient", "time"])
    return df

def save_summary(all_data: list, results_dir: str, cohort: str):
    rows = []
    for d in all_data:
        bg = d["BG"]
        rows.append({
            "Patient":          d["patient"],
            "TIR (%)":          round(100 * np.mean((bg >= 70) & (bg <= 180)), 2),
            "Hypo (%)":         round(100 * np.mean(bg < 70),  2),
            "Severe Hypo (%)":  round(100 * np.mean(bg < 54),  2),
            "Hyper (%)":        round(100 * np.mean(bg > 180), 2),
            "Mean BG":          round(float(np.mean(bg)),      2),
        })
    df   = pd.DataFrame(rows)
    path = os.path.join(results_dir, "report_summary.csv")
    df.to_csv(path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Generate RLAP-style report plots")
    parser.add_argument("--cohort", required=True, choices=["child", "adolescent", "adult"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--save_csv", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\n  REPORT PLOTS -- {args.cohort.upper()}\n{'='*70}\n")

    results_dir = f"./results/report_plots_{args.cohort}"
    os.makedirs(results_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = EnsembleAgent(4, 1, max_action=1.0, device=device)
    agent.load(args.model_path)
    agent.sac_agent.actor.eval()
    agent.td3_agent.actor.eval()
    agent.meta_controller.eval()

    manager      = StateRewardManager(4)
    safety_layer = SafetyLayer(cohort=args.cohort)
    cohort_patients = get_cohort_patients(args.cohort)
    
    # Optional: Start on Jan 1, 2026 for clean plot axes
    start_time = datetime(2026, 1, 1, 0, 0, 0)

    all_data = []
    
    print("  SIMULATING PATIENTS & GENERATING INDIVIDUAL PLOTS...")
    for patient_name in cohort_patients:
        print(f"\n  Processing {patient_name} ({args.days} days)...")
        
        # 1. Run Simulation
        data = run_patient(agent, manager, safety_layer, patient_name, args.cohort, args.seed, args.days, start_time)
        all_data.append(data)
        
        # 2. Generate RLAP Individual Plot
        patient_df = pd.DataFrame(
            {"BG": data["BG"], "CGM": data["CGM"], "CHO": data["CHO"], "insulin": data["insulin"]},
            index=pd.DatetimeIndex(data["times"])
        )
        plot_individual_patient(patient_df, patient_name, results_dir)
        
        # Save CSV if requested
        if args.save_csv:
            csv_path = os.path.join(results_dir, f"{patient_name.replace('#', '-')}.csv")
            patient_df.to_csv(csv_path)

    print("\n  Building MultiIndex DataFrame for Simglucose Aggregate Report...")
    df = build_report_df(all_data)

    print(f"  Calling simglucose report()  ->  {results_dir}/")
    results, ri_per_hour, zone_stats, figs, axes = report(df, cgm_sensor=None, save_path=results_dir)

    save_summary(all_data, results_dir, args.cohort)
    print(f"\n  [SUCCESS] All outputs saved to: {results_dir}/\n")

if __name__ == "__main__":
    main()