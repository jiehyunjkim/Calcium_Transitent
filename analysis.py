#!/usr/bin/env python3
"""
Batch runner for calcium movement analysis.
Usage:
    python analysis.py                    # run all cases
    python analysis.py V_1aav_05_2       # run single case by name
    python analysis.py --drug verapamil  # run all verapamil cases
"""

import os, sys, glob, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no GUI needed for batch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks, peak_prominences

import util as U

# ============================================================
# CONFIG
# ============================================================
BASE = "/raid/mpsych/CACTAS/DATA/FMA/drug"
OUTPUT_DIR = "/raid/mpsych/CACTAS/DATA/FMA/motion"
CSV_PATH = os.path.join(OUTPUT_DIR, "all_results_rebuttal.csv")

def build_case_list():
    """Auto-discover all cases from folder structure."""
    cases = []
    
    for drug_folder in ["verapamil", "nifidepine"]:
        img_base = os.path.join(BASE, drug_folder, "images")
        mask_base = os.path.join(BASE, drug_folder, "masks")
        
        if not os.path.isdir(img_base):
            continue
        
        for folder_name in sorted(os.listdir(img_base)):
            img_path = os.path.join(img_base, folder_name)
            mask_path = os.path.join(mask_base, folder_name)
            
            if not os.path.isdir(img_path) or not os.path.isdir(mask_path):
                continue
            
            # Check has tiff files
            tiffs = glob.glob(os.path.join(img_path, "*.tiff"))
            if len(tiffs) == 0:
                continue
            
            cases.append({
                "name": folder_name,
                "img_path": img_path,
                "mask_path": mask_path,
            })
    
    return cases


def parse_case_name(case_name):
    """Extract drug, concentration, age from folder name."""
    drug = 'verapamil' if case_name.startswith('V') else 'nifedipine'
    
    if '_age_' in case_name:
        age = int(case_name.split('_age_')[1].split('_')[0])
        conc = None
    elif '_B_' in case_name:
        age = None
        conc = 0  # baseline
    else:
        age = None
        parts = case_name.split('_')
        conc_str = parts[2]
        conc_map = {'002': 0.02, '01': 0.1, '05': 0.5, '1': 1.0, '10': 10.0}
        conc = conc_map.get(conc_str, conc_str)
    
    return drug, conc, age


def run_single_case(img_path, mask_path, case_name, output_dir=OUTPUT_DIR):
    """Run full analysis for one case. Returns case_result dict for CSV."""
    
    print(f"\n{'='*60}")
    print(f"  {case_name}")
    print(f"{'='*60}")
    
    drug, conc, age = parse_case_name(case_name)
    
    # --- PDF setup ---
    pdf_path = os.path.join(output_dir, f"{case_name}.pdf")
    pdf = PdfPages(pdf_path)
    U.set_pdf(pdf)
    
    try:
        # 1. Load data
        image_files = sorted(glob.glob(img_path + '/*.tiff'), key=U.extract_frame_number)
        mask_files = sorted(glob.glob(mask_path + '/*.tiff'), key=U.extract_frame_number_mask)
        print(f"  Images: {len(image_files)}, Masks: {len(mask_files)}")
        
        image_stack, mask_stack = U.read_data(image_files, mask_files)
        
        # 2. QC masks
        mask_stack, dbg, best_mask = U.qc_keep_or_replace_from_preprocessed_masks(
            mask_stack,
            top_crop_frac=0.05,
            band_frac=0.35,
            outside_weight=2.0,
            top_weight=2.0,
            raw_min=30000,
            norm_th=0.80,
            debug=True,
        )
        
        # 3. Grid setup
        tissue_mask = (best_mask > 0).astype(np.uint8)
        U.tissue_mask = tissue_mask
        U.GRID_CELLS, U.GRID_BBOXES, U.GRID_LINES = U._build_skewed_grid_from_mask(
            tissue_mask, num_rows=3, num_cols=2)
        
        # 4. Preprocessing
        smooth_data = U.smooth_norm_image(image_stack, 4, 3, 3)
        intensity_signal = smooth_data.mean(axis=(1, 2))
        
        # PDF page: raw signal
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(intensity_signal)
        ax.set_title(f"{case_name} - Mean intensity over time")
        ax.set_xlabel("Frame"); ax.set_ylabel("Average intensity")
        pdf.savefig(fig); plt.close(fig)
        
        # 5. Delta F / F0
        deltaF_over_F0 = U.deltaF_over_F0_fixed(smooth_data, mask_stack, best_mask, baseline_frames=3000)
        deltaF_over_F0 = deltaF_over_F0 - deltaF_over_F0.min()
        
        # 6. Peak detection
        peaks, _ = find_peaks(deltaF_over_F0)
        prominences = peak_prominences(deltaF_over_F0, peaks)[0]
        prom_threshold = np.percentile(prominences, 75) * 0.4
        
        if len(peaks) >= 2:
            intervals = np.diff(peaks)
            min_distance = int(np.median(intervals) * 0.4)
        else:
            min_distance = 25
        
        peaks, _ = find_peaks(deltaF_over_F0, prominence=prom_threshold, distance=min_distance)


        
        if len(peaks) > 3:
            peak_amplitudes = deltaF_over_F0[peaks]
            amp_threshold = np.percentile(peak_amplitudes, 75) * 0.3
            peaks = peaks[peak_amplitudes >= amp_threshold]


        
        boundaries = U.find_peak_boundaries(deltaF_over_F0, peaks)
        left_ips = np.array([b['rise_frame'] for b in boundaries])
        right_ips = np.array([b['decay_frame'] for b in boundaries])
        
        print(f"  Peaks detected: {len(peaks)}")


        # 6.5 Noise floor estimation from quiet windows
        noise_floor_px, nf_dbg = U.estimate_noise_floor_px_from_quiet_windows(
            video=smooth_data,
            mask_stack=mask_stack,
            boundaries=boundaries,
            spacing=10,
            win=15,
            q=95,
            debug_plot=True
        )
        U.NOISE_FLOOR_PX = noise_floor_px
        

        # Signal quality metrics
        peak_amps = deltaF_over_F0[peaks]
        signal_cv = float(np.std(peak_amps) / np.mean(peak_amps)) if len(peaks) > 1 else 0
        beat_rate = len(peaks) / (len(deltaF_over_F0) / 100.0)  # Hz
        rise_durations = [b['peak_frame'] - b['rise_frame'] for b in boundaries]
        decay_durations = [b['decay_frame'] - b['peak_frame'] for b in boundaries]
        mean_rise_frames = np.mean(rise_durations)
        mean_decay_frames = np.mean(decay_durations)

        
        # PDF page: signal + peaks
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(deltaF_over_F0)
        ax.plot(peaks, deltaF_over_F0[peaks], "x")
        for l, r in zip(left_ips, right_ips):
            ax.hlines(deltaF_over_F0[l], l, r, color='r')
        ax.set_title(f"{case_name} - ΔF/F₀ ({len(peaks)} peaks)")
        pdf.savefig(fig); plt.close(fig)
        
        # 7. Peak info + optical flow
        peaks_info = U.peak_info(deltaF_over_F0, peaks, left_ips, right_ips, plot=False)
        trajectories_per_peak = U.LK_per_peak(
            video=smooth_data, mask_stack=mask_stack, peaks_info=peaks_info, spacing=10)
        
        results = U.analyze_peaks_and_velocity(
            video=smooth_data,
            trajectories=trajectories_per_peak,
            peak_info_results=peaks_info,
            fps=100,
            plot=False,
            plot_velocity=False
        )


        # save summary text for each velocity
        vel_rise_all = []
        vel_decay_all = []
        for b in boundaries:
            rise_f = b['peak_frame'] - b['rise_frame']
            decay_f = b['decay_frame'] - b['peak_frame']
            vel_rise_all.append(rise_f)
            vel_decay_all.append(decay_f)


        
        # 8. Vector analysis
        direction = U.capture_print(
            U.generate_rise_decay_flow_images_for_peaks,
            smooth_data, trajectories_per_peak, results,
            output_dir=output_dir,
            num_rows=3, num_cols=2, top_3=False
        )


        # === 8.5 ABLATION (insert) ===========================================
        # Same trajectories, different (grid, phase) configs. No re-run of LK.
        direction_C = U.capture_print(
            U.aggregate_motion_with_config,
            trajectories_per_peak, peaks_info, tissue_mask,
            num_rows=3, num_cols=2, phase_split=False,
            video_shape=smooth_data.shape
        )
        direction_D = U.capture_print(
            U.aggregate_motion_with_config,
            trajectories_per_peak, peaks_info, tissue_mask,
            num_rows=1, num_cols=1, phase_split=True,
            video_shape=smooth_data.shape
        )
        direction_B = U.capture_print(
            U.aggregate_motion_with_config,
            trajectories_per_peak, peaks_info, tissue_mask,
            num_rows=1, num_cols=1, phase_split=False,
            video_shape=smooth_data.shape
        )
        # Restore 3x2 grid for subsequent steps
        U.GRID_CELLS, U.GRID_BBOXES, U.GRID_LINES = U._build_skewed_grid_from_mask(
            tissue_mask, num_rows=3, num_cols=2)

        ablation_E = U.summarize_region_phase(direction)
        ablation_C = U.summarize_region_phase(direction_C)
        ablation_D = U.summarize_region_phase(direction_D)
        ablation_B = U.summarize_region_phase(direction_B)
        # === END ABLATION ====================================================

        
        # 9. Show all rise/decay images
        U.show_all_rise_decay(output_dir, num_peaks=len(peaks), cols=2)
        
        # 10. Direction + displacement plots
        U.plot_unwrapped_direction_by_region(direction)
        U.plot_displacement_bar_charts(direction)
        
        # 11. Quick analysis
        summary = U.capture_print(
            U.run_one_case_quick_analysis,
            direction, num_rows=3, num_cols=2, title_prefix=case_name
        )
        
        # 12. Coupling
        out = U.capture_print(
            U.run_one_case_lr_and_coupling,
            direction, num_rows=3, num_cols=2, title_prefix=case_name
        )
        
        # 13. Histogram
        U.histogram(direction)

        
        # 14. Velocity summary for PDF
        vel_text = f"=== {case_name} Velocity Summary ===\n"
        for (row, col, phase), stats in summary.items():
            vel_text += f"  ({row},{col}) {phase}: vel={stats.get('mean_vel_mm_s', 'N/A'):.4f} mm/s, disp={stats['mean_disp_mm']:.4f} mm\n"
        vel_text += f"\nBeat rate: {beat_rate:.3f} Hz"
        vel_text += f"\nMean rise: {mean_rise_frames/100:.3f}s, decay: {mean_decay_frames/100:.3f}s"
        vel_text += f"\nNoise floor: {U.NOISE_FLOOR_PX:.4f} px"
        U.save_text_page(vel_text, title="Velocity Summary")

        
        # --- Build CSV row ---
        case_result = {
            'case_name': case_name,
            'drug': drug,
            'conc': conc,
            'age': age,
            'n_peaks': len(peaks),
            'signal_cv': round(signal_cv, 4),
            'beat_rate_hz': round(beat_rate, 3),
            'mean_rise_frames': round(mean_rise_frames, 1),
            'mean_decay_frames': round(mean_decay_frames, 1),
            'mean_rise_duration_s': round(mean_rise_frames / 100.0, 4),
            'mean_decay_duration_s': round(mean_decay_frames / 100.0, 4),
            'rise_decay_time_ratio': round(mean_rise_frames / max(mean_decay_frames, 1), 3),
        }
        
        # R, disp, angle per cell
        for (row, col, phase), stats in summary.items():
            prefix = f"r{row}c{col}_{phase}"
            case_result[f"{prefix}_R"] = stats['R']
            case_result[f"{prefix}_disp_mm"] = stats['mean_disp_mm']
            case_result[f"{prefix}_angle"] = stats['mean_angle_deg']
            case_result[f"{prefix}_vel_mm_s"] = stats.get('mean_vel_mm_s', np.nan)
        
        # Coupling
        coup = out['coupling']
        for key, v in coup.items():
            if key == 'global':
                continue
            row, col = key
            case_result[f"r{row}c{col}_delta_angle"] = v.get('mean_abs_angle_diff_deg')
            case_result[f"r{row}c{col}_disp_ratio"] = v.get('mean_disp_ratio_decay_over_rise')
            case_result[f"r{row}c{col}_vel_ratio"] = v.get('mean_vel_ratio_decay_over_rise')

        
        if 'global' in coup:
            g = coup['global']
            case_result['global_mean_delta_angle'] = g['mean_abs_angle_diff_deg']
            case_result['global_disp_ratio'] = g['mean_disp_ratio_decay_over_rise']
            case_result['global_vel_ratio'] = g.get('mean_vel_ratio_decay_over_rise', np.nan)



        # === ABLATION CSV fields (insert) ===================================
        def _abl_disp(s, ph):
            vals = [v['mean_disp_mm'] for k, v in s.items() if k[2] == ph]
            return float(np.mean(vals)) if vals else np.nan
        def _abl_vel(s, ph):
            vals = [v['mean_vel_mm_s'] for k, v in s.items() if k[2] == ph]
            return float(np.mean(vals)) if vals else np.nan
        def _abl_R(s, ph=None):
            if ph is None:
                vals = [v['R'] for v in s.values()]
            else:
                vals = [v['R'] for k, v in s.items() if k[2] == ph]
            return float(np.mean(vals)) if vals else np.nan
        def _safe_ratio(a, b):
            try:
                if b is None or np.isnan(b) or b <= 0 or np.isnan(a):
                    return np.nan
                return float(a / b)
            except Exception:
                return np.nan

        # E (proposed: 3x2 + phase)
        case_result['abl_E_rise_disp']  = _abl_disp(ablation_E, 'rise')
        case_result['abl_E_decay_disp'] = _abl_disp(ablation_E, 'decay')
        case_result['abl_E_rise_vel']   = _abl_vel(ablation_E, 'rise')
        case_result['abl_E_decay_vel']  = _abl_vel(ablation_E, 'decay')
        case_result['abl_E_R_mean']     = _abl_R(ablation_E)

        # C (3x2 + no phase)
        case_result['abl_C_full_disp']  = _abl_disp(ablation_C, 'full')
        case_result['abl_C_full_vel']   = _abl_vel(ablation_C, 'full')
        case_result['abl_C_R_mean']     = _abl_R(ablation_C)

        # D (1x1 + phase)
        case_result['abl_D_rise_disp']  = _abl_disp(ablation_D, 'rise')
        case_result['abl_D_decay_disp'] = _abl_disp(ablation_D, 'decay')
        case_result['abl_D_rise_vel']   = _abl_vel(ablation_D, 'rise')
        case_result['abl_D_decay_vel']  = _abl_vel(ablation_D, 'decay')
        case_result['abl_D_R_mean']     = _abl_R(ablation_D)

        # B (1x1 + no phase) — global baseline
        case_result['abl_B_full_disp']  = _abl_disp(ablation_B, 'full')
        case_result['abl_B_full_vel']   = _abl_vel(ablation_B, 'full')
        case_result['abl_B_R_mean']     = _abl_R(ablation_B)

        # Derived metrics
        case_result['abl_spatial_loss_rise']  = 1 - _safe_ratio(case_result['abl_D_rise_disp'],  case_result['abl_E_rise_disp'])
        case_result['abl_spatial_loss_decay'] = 1 - _safe_ratio(case_result['abl_D_decay_disp'], case_result['abl_E_decay_disp'])
        case_result['abl_phase_asymmetry_vel']  = _safe_ratio(case_result['abl_E_rise_vel'], case_result['abl_E_decay_vel'])
        case_result['abl_total_loss_vel']     = 1 - _safe_ratio(case_result['abl_B_full_vel'],  case_result['abl_E_rise_vel'])
        case_result['abl_total_loss_disp']    = 1 - _safe_ratio(case_result['abl_B_full_disp'], case_result['abl_E_rise_disp'])
        # === END ABLATION CSV fields =========================================
        
        print(f"  ✓ Done: {case_name}")
        return case_result
    
    except Exception as e:
        print(f"  ✗ FAILED: {case_name} - {e}")
        traceback.print_exc()
        return None
    
    finally:
        pdf.close()
        U.set_pdf(None)
        plt.close('all')  # clean up memory


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_cases = build_case_list()
    print(f"Found {len(all_cases)} cases total")
    
    # Filter by command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == '--drug':
            drug_filter = sys.argv[2]
            all_cases = [c for c in all_cases if drug_filter in c['name'].lower() 
                        or (drug_filter == 'verapamil' and c['name'].startswith('V'))
                        or (drug_filter == 'nifidepine' and c['name'].startswith('N'))]
        else:
            # Filter by case name
            name_filter = sys.argv[1]
            all_cases = [c for c in all_cases if name_filter in c['name']]
    
    print(f"Running {len(all_cases)} cases:")
    for c in all_cases:
        print(f"  - {c['name']}")
    
    # Run all
    all_results = []
    for case in all_cases:
        result = run_single_case(
            img_path=case['img_path'],
            mask_path=case['mask_path'],
            case_name=case['name']
        )
        if result is not None:
            all_results.append(result)
    
    # Save CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Append to existing or create new
        if os.path.exists(CSV_PATH):
            df_old = pd.read_csv(CSV_PATH)
            # Remove duplicates (re-run same case)
            df_old = df_old[~df_old['case_name'].isin(df['case_name'])]
            df = pd.concat([df_old, df], ignore_index=True)
        
        df.to_csv(CSV_PATH, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved: {CSV_PATH}")
        print(f"Total cases in CSV: {len(df)}")
        print(f"{'='*60}")
    
    # Summary
    succeeded = len(all_results)
    failed = len(all_cases) - succeeded
    print(f"\nDone: {succeeded} succeeded, {failed} failed")