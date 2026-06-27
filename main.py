import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- PROFILE CONSTANTS ---
AGE = 29
WEIGHT_KG = 84.3

def extract_session_metrics(file_path):
    """
    Parses a single TCX and returns a dictionary of key metrics.
    """
    ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        records = []
        trackpoints = root.findall('.//ns:Trackpoint', ns)


        
        for tp in trackpoints:
            time_node = tp.find('ns:Time', ns)
            hr_node = tp.find('ns:HeartRateBpm/ns:Value', ns)
            if time_node is not None and hr_node is not None:
                records.append({'time': time_node.text, 'hr': int(hr_node.text)})

        
        df = pd.DataFrame(records)
        if df.empty: return None
        
        df['time'] = pd.to_datetime(df['time'])
        df['sec'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
        df['dur_min'] = df['sec'].diff().fillna(1) / 60
        
        # Formulas
        hr_max = 208 - (0.7 * AGE)
        hr_rest = df['hr'].iloc[0:120].min() if len(df) > 120 else 65
        
        # TRIMP & Calories
        df['delta_hr'] = ((df['hr'] - hr_rest) / (hr_max - hr_rest)).clip(0, 1)
        trimp = (df['dur_min'] * df['delta_hr'] * 0.64 * np.exp(1.92 * df['delta_hr'])).sum()
        kcal = ((-55.0969 + (0.6309 * df['hr']) + (0.1988 * WEIGHT_KG) + (0.2017 * AGE)) * df['dur_min']).sum() * 0.239
        
        return {
            'date': df['time'].iloc[0],
            'duration_min': df['sec'].max() / 60,
            'avg_hr': df['hr'].mean(),
            'max_hr': df['hr'].max(),
            'trimp': trimp,
            'kcal': kcal,
            'file': os.path.basename(file_path)
        }
    except Exception as e:
        print(f"Error skipping {file_path}: {e}")
        return None

def calculate_fitness_metrics(summary_df):
    """
    Calculates CTL (Fitness), ATL (Fatigue), and TSB (Form).
    """
    summary_df = summary_df.sort_values('date').reset_index(drop=True)
    
    # 7-day and 42-day time constants
    alpha_atl = 1 - np.exp(-1 / 7)
    alpha_ctl = 1 - np.exp(-1 / 42)
    
    atl, ctl = [0.0] * len(summary_df), [0.0] * len(summary_df)
    
    # Simple EWMA
    for i in range(len(summary_df)):
        load = summary_df.loc[i, 'trimp']
        if i == 0:
            atl[i], ctl[i] = load, load
        else:
            atl[i] = atl[i-1] + alpha_atl * (load - atl[i-1])
            ctl[i] = ctl[i-1] + alpha_ctl * (load - ctl[i-1])
            
    summary_df['ATL'] = atl
    summary_df['CTL'] = ctl
    summary_df['TSB'] = summary_df['CTL'] - summary_df['ATL']
    return summary_df

def plot_training_trends(df, output_dir):
    """
    Generates a high-fidelity Athlete Performance Report.
    """
    plt.close('all')
    sns.set_style("white")
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 4, 3], hspace=0.3)
    
    # 1. METRIC CARDS (Top Panel)
    ax_cards = fig.add_subplot(gs[0])
    ax_cards.axis('off')
    
    current_ctl = df['CTL'].iloc[-1]
    current_atl = df['ATL'].iloc[-1]
    current_tsb = df['TSB'].iloc[-1]
    
    # Simple Status Logic
    if current_tsb < -30: status, status_col = "DANGER: OVERREACHING", "#e63946"
    elif -30 <= current_tsb < -10: status, status_col = "OPTIMAL TRAINING ZONE", "#2a9d8f"
    elif -10 <= current_tsb < 5: status, status_col = "NEUTRAL/TRANSITION", "#6c757d"
    else: status, status_col = "FRESH / RECOVERED", "#457b9d"

    card_text = (
        f"ATHLETE PERFORMANCE REPORT\n{'-'*30}\n"
        f"FITNESS (CTL): {current_ctl:.1f}  |  FATIGUE (ATL): {current_atl:.1f}  |  FORM (TSB): {current_tsb:.1f}\n"
        f"CURRENT STATUS: {status}"
    )
    ax_cards.text(0.5, 0.5, card_text, fontsize=22, fontweight='bold', ha='center', va='center',
                  bbox=dict(boxstyle='round,pad=1', facecolor=status_col, alpha=0.15, edgecolor=status_col))

    # 2. PERFORMANCE MANAGER (Middle Panel)
    ax_pmc = fig.add_subplot(gs[1])
    # Use dates for labels but index for plotting to avoid spacing issues
    x_labels = df['date'].dt.strftime('%b %d')
    ax_pmc.plot(df.index, df['CTL'], label='Fitness (Long-term)', color='#1d3557', linewidth=4, zorder=5)
    ax_pmc.fill_between(df.index, df['CTL'], color='#1d3557', alpha=0.1)
    
    ax_pmc.plot(df.index, df['ATL'], label='Fatigue (Short-term)', color='#e63946', linewidth=2, alpha=0.6, linestyle='--')
    ax_pmc.set_title('Fitness vs. Fatigue Evolution', fontsize=18, fontweight='bold', loc='left')
    ax_pmc.set_ylabel('Training Load Units')
    ax_pmc.legend(loc='upper left', frameon=True)
    ax_pmc.set_xticks(df.index)
    ax_pmc.set_xticklabels(x_labels, rotation=35)

    # 3. TRAINING STRESS BALANCE (Bottom Panel)
    ax_tsb = fig.add_subplot(gs[2], sharex=ax_pmc)
    ax_tsb.plot(df.index, df['TSB'], color='#333', linewidth=3, marker='o', markersize=8)
    
    # SHADE THE ZONES
    ax_tsb.axhspan(-10, 10, color='#6c757d', alpha=0.1, label='Neutral') # Neutral
    ax_tsb.axhspan(-30, -10, color='#2a9d8f', alpha=0.2, label='Optimal (Building)') # Optimal
    ax_tsb.axhspan(10, 40, color='#457b9d', alpha=0.2, label='Fresh (Race Ready)') # Fresh
    ax_tsb.axhspan(-60, -30, color='#e63946', alpha=0.1, label='Danger (Over-training)') # Danger
    
    ax_tsb.axhline(0, color='black', linewidth=1, linestyle='-')
    ax_tsb.set_title('Training Stress Balance (Your Recovery Readiness)', fontsize=18, fontweight='bold', loc='left')
    ax_tsb.set_ylabel('Form (TSB)')
    ax_tsb.legend(loc='lower left', ncol=4, fontsize=10)
    
    plt.tight_layout()
    report_path = os.path.join(output_dir, 'ATHLETE_INTELLIGENCE_REPORT.png')
    plt.savefig(report_path, dpi=150)
    plt.close()
    print(f"Intuitive Dashboard saved to: {report_path}")


def analyze_cardiac_trends(df):
    """
    Analyzes how the heart is adapting by comparing early vs late sessions.
    """
    if len(df) < 2:
        return "Insufficient data for trend prediction."
    
    first_session_hr = df['avg_hr'].iloc[0]
    last_session_hr = df['avg_hr'].iloc[-1]
    hr_delta = ((last_session_hr - first_session_hr) / first_session_hr) * 100
    
    # Simple Efficiency Proxy: Duration / Avg HR
    df['efficiency'] = df['duration_min'] / df['avg_hr']
    eff_improvement = ((df['efficiency'].iloc[-1] - df['efficiency'].iloc[0]) / df['efficiency'].iloc[0]) * 100
    
    # Forecast: Project CTL 7 days forward assuming same avg load
    avg_daily_load = df['trimp'].mean()
    alpha_ctl = 1 - np.exp(-1 / 42)
    last_ctl = df['CTL'].iloc[-1]
    
    projected_ctl = last_ctl
    for _ in range(7):
        projected_ctl = projected_ctl + alpha_ctl * (avg_daily_load - projected_ctl)
        
    return {
        'hr_delta': hr_delta,
        'eff_improvement': eff_improvement,
        'projected_ctl': projected_ctl,
        'load_consistency': df['trimp'].std()
    }

def export_full_dataset(file_path, output_dir):
    """
    Parses cleaned and universal Time and Heart Rate data.
    """
    ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        records = []
        trackpoints = root.findall('.//ns:Trackpoint', ns)
        
        for tp in trackpoints:
            time_node = tp.find('ns:Time', ns)
            hr_node = tp.find('ns:HeartRateBpm/ns:Value', ns)
            
            if time_node is not None and hr_node is not None:
                records.append({
                    'raw_time': time_node.text,
                    'heart_rate': int(hr_node.text)
                })
        
        full_df = pd.DataFrame(records)
        if not full_df.empty:
            # Universal Time Conversion
            full_df['timestamp'] = pd.to_datetime(full_df['raw_time'])
            
            # 1. Human Readable Standard (YYYY-MM-DD HH:MM:SS)
            full_df['datetime'] = full_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 2. Universal Unix Epoch (Seconds since 1970)
            full_df['unix_epoch'] = full_df['timestamp'].astype(np.int64) // 10**9
            
            # 3. Relative Seconds (Offset from start of workout)
            full_df['relative_seconds'] = (full_df['timestamp'] - full_df['timestamp'].iloc[0]).dt.total_seconds()
            
            # Clean up and reorder columns for a professional look
            final_df = full_df[['datetime', 'unix_epoch', 'relative_seconds', 'heart_rate']]
            
            # Calculate Total Duration for filename

            total_min = int(full_df['relative_seconds'].iloc[-1] / 60)
            date_str = full_df['timestamp'].iloc[0].strftime('%Y-%m-%d')
            
            # Format filename: 2026-05-11_51min_parsed.csv
            file_name = f"{date_str}_{total_min}min_parsed.csv"
            export_path = os.path.join(output_dir, file_name)
            
            final_df.to_csv(export_path, index=False)
            return True

    except Exception:
        pass
    return False



def main():
    input_dir = '/Users/olegbalabuha/Desktop/Training_Sessions'
    report_dir = '/Users/olegbalabuha/Desktop/HR_Analysis_Results'
    export_dir = '/Users/olegbalabuha/Desktop/Parsed_Training_Data'
    
    for d in [report_dir, export_dir]:
        if not os.path.exists(d): os.makedirs(d)

    print(f"--- INITIALIZING BROAD-SPECTRUM DATA EXTRACTION ---")
    
    session_summaries = []
    files = [f for f in os.listdir(input_dir) if f.endswith('.tcx')]
    
    for file_name in files:
        full_path = os.path.join(input_dir, file_name)
        
        # 1. Export High-Resolution CSV for future projects
        success = export_full_dataset(full_path, export_dir)
        if success:
            print(f"Parsed & Saved: {file_name}")
        
        # 2. Extract summary for current longitudinal report
        res = extract_session_metrics(full_path)
        if res: session_summaries.append(res)
    
    if session_summaries:
        summary_df = pd.DataFrame(session_summaries)
        
        # --- Deduplication ---
        summary_df = summary_df.drop_duplicates(subset=['date', 'duration_min'])
        summary_df = calculate_fitness_metrics(summary_df)
        
        # --- Final Outputs ---
        plot_training_trends(summary_df, report_dir)
        summary_df.to_csv(os.path.join(report_dir, 'training_integrated_summary.csv'), index=False)
        print(f"\nALL DATA CLEANED AND STORED IN: {export_dir}")
        print(f"INTEGRATED SUMMARY SAVED TO: {report_dir}")
    else:
        print("Dataset initialization failed.")



if __name__ == "__main__":
    main()






