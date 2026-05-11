import os
import glob
import tifffile
import matplotlib.pyplot as plt
import io, sys
from scipy.signal import find_peaks, peak_widths, peak_prominences, savgol_filter
import scipy.ndimage as ndimage
from scipy.stats import circmean
from scipy.ndimage import gaussian_filter1d, gaussian_filter, binary_fill_holes, label
import re
import time
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import math
import numpy as np
from PIL import Image
import tifffile as tiff
import builtins
from scipy.ndimage import shift as ndi_shift
from skimage.filters import threshold_otsu
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.models import load_model

cv2.setUseOptimized(True)
cv2.setNumThreads(cv2.getNumberOfCPUs())

tissue_mask = None
GRID_CELLS = None 
GRID_BBOXES = None         # shape: [num_rows][num_cols] -> (xmin, xmax, ymin, ymax)
GRID_LINES = None 
_pdf = None
NOISE_FLOOR_PX = 0.0

def set_pdf(pdf_obj):
    global _pdf
    _pdf = pdf_obj

def save_text_page(text, title=""):
    global _pdf
    if _pdf is None:
        return
    
    lines = text.split('\n')
    max_lines = 50 
    
    for page_start in range(0, len(lines), max_lines):
        chunk = '\n'.join(lines[page_start:page_start + max_lines])
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis('off')
        page_title = title if page_start == 0 else f"{title} (cont.)"
        ax.text(0.05, 0.95, f"{page_title}\n\n{chunk}", transform=ax.transAxes,
                fontsize=7, verticalalignment='top', fontfamily='monospace')
        _pdf.savefig(fig)
        plt.close(fig)

def capture_print(func, *args, **kwargs):
    global _pdf
    buf = io.StringIO()
    sys.stdout = buf
    result = func(*args, **kwargs)
    sys.stdout = sys.__stdout__
    
    text = buf.getvalue()
    if text.strip() and _pdf is not None:
        save_text_page(text, title=func.__name__)
    
    return result



def compute_noise_floor(trajectories_all_peaks, results):
    """check noise floor between peaks"""
    baseline_steps = []

    trajs = trajectories_all_peaks[0] if trajectories_all_peaks else []
    
    for traj in trajs:
        traj = np.array(traj)
        for i in range(len(results) - 1):
            rest_start = results[i]['decay_frame']
            rest_end = results[i+1]['rise_frame']
            if rest_end - rest_start < 3:
                continue
            for k in range(rest_start, min(rest_end, len(traj)-1)):
                bx = traj[k+1][0] - traj[k][0]
                by = traj[k+1][1] - traj[k][1]
                baseline_steps.append(np.sqrt(bx**2 + by**2))
    
    if len(baseline_steps) > 10:
        nf = np.mean(baseline_steps) + 2 * np.std(baseline_steps)
    else:
        nf = 0.3
    return nf



def generate_masks(MODEL_PATH, image_folder, masks_folder):
    # Load all test image paths
    image_paths = sorted(glob.glob(os.path.join(image_folder, '*')))

    # Load the pre-trained model
    if MODEL_PATH:
        model = load_model(MODEL_PATH)
    else:
        MODEL_PATH = "/raid/mpsych/CACTAS/DATA/FMA/unet_Fluo_clip_v2.h5"
        model = load_model(MODEL_PATH)

    ###### Generating Masks #######
    for path in image_paths:
        # 1) Load image as grayscale
        image = Image.open(path).convert('L')
        img = np.array(image).astype(np.float32)
    
        # 2) Normalize intensity
        low, high = np.percentile(img, [0.35, 99.65])
        img = np.clip((img - low) * 255.0 / (high - low), 0, 255)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
        # 3) Pad image to match model input size
        padded_img = np.zeros((544, 720), dtype=img.dtype)
        padded_img[2:542, :] = img
        img_input = np.expand_dims(padded_img, axis=(0, -1))
    
        # 4) Predict mask
        pred = model.predict(img_input)[0, :, :, 0]
        pred_binary = (pred > 0.5).astype(np.uint8) * 255
    
        # 5) Crop back to original size and save
        pred_cropped = pred_binary[2:542, :]
        im_name = os.path.splitext(os.path.basename(path))[0]
        tiff.imwrite(os.path.join(masks_folder, f'{im_name}_pred.tiff'), pred_cropped)


def visualize_generated_masks(masks_folder, n=50, remove=0.1):
    # Load all mask file paths
    mask_paths = sorted(glob.glob(os.path.join(masks_folder, '*.tiff')))

    # fill holes, remove small chunck
    for mask_path in mask_paths[:n]:
        mask = tiff.imread(mask_path)
        mask = (mask > 0).astype(np.uint8)  # ensure binary
    
        # fill holes
        mask_filled = binary_fill_holes(mask).astype(np.uint8)
    
        # find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
        sizes = stats[1:, -1]  # skip background (label 0)
        largest = sizes.max() if len(sizes) > 0 else 0
    
        # remove components smaller than 10% of largest
        cleaned = np.zeros_like(mask_filled)
        for i, size in enumerate(sizes):
            if size >= remove * largest:
                cleaned[labels == i + 1] = 1
    
        # visualize
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(mask, cmap='gray')
        plt.title('Original')
        plt.axis('off')
    
        plt.subplot(1, 3, 2)
        plt.imshow(mask_filled, cmap='gray')
        plt.title('Filled')
        plt.axis('off')
    
        plt.subplot(1, 3, 3)
        plt.imshow(cleaned, cmap='gray')
        plt.title('Cleaned (10% filter)')
        plt.axis('off')
    
        plt.suptitle(os.path.basename(mask_path))
        plt.show()


def extract_frame_number(filename):
    match = re.search(r'([0-9]{1,10})\.tiff$', filename)
    return int(match.group(1)) if match else -1

def extract_frame_number_mask(filename):
    match = re.search(r'([0-9]{1,10})\_pred.tiff$', filename)
    return int(match.group(1)) if match else -1

def read_data(image_files, mask_files):
    image_stack = []
    for img_file in image_files:
        # image = tifffile.imread(img_file)
        image = np.array(Image.open(img_file).convert('L'), dtype=np.float32)
        low, high = np.percentile(image, [0.35, 99.65])
        image = np.clip((image - low) * 255.0 / (high - low), 0, 255)
        image_stack.append(image)

    mask_stack = []
    for mask_file in mask_files:
        mask = tifffile.imread(mask_file)
        
        ######fill holes + remove small components ##########
        mask = (mask > 0).astype(np.uint8)
        
        # fill holes
        mask_filled = binary_fill_holes(mask).astype(np.uint8)
    
        # find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
        sizes = stats[1:, -1]  # skip background (label 0)
        largest = sizes.max() if len(sizes) > 0 else 0
    
        # remove components smaller than 10% of largest
        cleaned = np.zeros_like(mask_filled)
        for i, size in enumerate(sizes):
            if size >= 0.1 * largest:
                cleaned[labels == i + 1] = 1
        #######
        
        mask_stack.append(cleaned)

    print(len(image_stack), len(mask_stack))
    
    return image_stack, mask_stack


def _center_band_x(W: int, band_frac: float):
    bw = builtins.max(1, int(round(W * band_frac)))
    halfW = W // 2
    halfbw = bw // 2
    x0 = int(np.subtract(halfW, halfbw))
    if x0 < 0:
        x0 = 0
    x1 = x0 + bw
    if x1 > W:
        x1 = W
    return x0, x1

def keep_components_overlapping_band(mask: np.ndarray, top_crop_frac: float, band_frac: float) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    m = binary_fill_holes(m).astype(np.uint8)

    H, W = m.shape
    y0 = int(round(H * top_crop_frac))
    if y0 + 2 >= H:
        return np.zeros_like(m, dtype=np.uint8)

    x0, x1 = _center_band_x(W, band_frac)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m, dtype=np.uint8)

    out = np.zeros_like(m, dtype=np.uint8)

    for k in range(1, n):
        comp = (lab == k).astype(np.uint8)

        roi = comp[y0:H, x0:x1]
        if int(roi.sum()) > 0:
            out[lab == k] = 1

    return out

def shaft_concentration_score(mask: np.ndarray,
                              top_crop_frac: float = 0.0,
                              band_frac: float = 0.30,
                              outside_weight: float = 2.0,
                              top_weight: float = 4.0) -> float:

    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    y0 = int(round(H * top_crop_frac))
    if y0 + 2 >= H:
        return 0.0

    x0, x1 = _center_band_x(W, band_frac)

    top = m[0:y0, :]
    bottom = m[y0:H, :]

    bottom_in = bottom[:, x0:x1]
    bottom_out_left = bottom[:, 0:x0]
    bottom_out_right = bottom[:, x1:W]

    a_in = float(bottom_in.sum())
    a_out = float(np.add(bottom_out_left.sum(), bottom_out_right.sum()))
    a_top = float(top.sum())

    penalty_out = float(np.multiply(outside_weight, a_out))
    penalty_top = float(np.multiply(top_weight, a_top))

    score = float(np.subtract(np.subtract(a_in, penalty_out), penalty_top))
    return score



def qc_keep_or_replace_from_preprocessed_masks(
    mask_stack,
    top_crop_frac=0.05,
    band_frac=0.35,
    outside_weight=2.0,
    top_weight=2.0,
    raw_min=30000,
    norm_th=0.90,
    debug=True,
):
    scores = []

    for m in mask_stack:
        m = np.asarray(m)
        if m.ndim > 2:
            m = np.squeeze(m)
        m = (m > 0).astype(np.uint8)

        s = shaft_concentration_score(
            m,
            top_crop_frac=top_crop_frac,
            band_frac=band_frac,
            outside_weight=outside_weight,
            top_weight=top_weight,
        )
        scores.append(float(s))

    scores = np.asarray(scores, dtype=np.float32)

    ref_idx = int(np.nanargmax(scores)) if np.any(np.isfinite(scores)) else 0
    best_raw = float(scores[ref_idx]) if np.isfinite(scores[ref_idx]) else float("nan")
    ref_mask = np.asarray(mask_stack[ref_idx]).copy()

    all_fail = (not np.isfinite(best_raw)) or (best_raw < raw_min) or (best_raw <= 0)

    if all_fail:
        score_norm = np.zeros_like(scores, dtype=np.float32)
        qc_pass = np.zeros(len(mask_stack), dtype=np.uint8)
        mask_stack_qc = [ref_mask.copy() for _ in mask_stack]
    else:
        score_norm = scores / best_raw
        qc_pass = (np.isfinite(score_norm) & (score_norm >= norm_th)).astype(np.uint8)

        mask_stack_qc = []
        for i, m in enumerate(mask_stack):
            if qc_pass[i] == 1:
                mask_stack_qc.append(np.asarray(m).copy())      # keep
            else:
                mask_stack_qc.append(ref_mask.copy())           # replace

    dbg = {}
    if debug:
        dbg = {
            "ref_idx": ref_idx,
            "best_raw": best_raw,
            "score_min": float(np.nanmin(scores)) if np.any(np.isfinite(scores)) else float("nan"),
            "score_median": float(np.nanmedian(scores)) if np.any(np.isfinite(scores)) else float("nan"),
            "score_max": float(np.nanmax(scores)) if np.any(np.isfinite(scores)) else float("nan"),
            "all_fail": bool(all_fail),
            "raw_min": float(raw_min),
            "norm_th": float(norm_th),
            "pass_rate": float(qc_pass.mean()) if len(qc_pass) else 0.0,
        }

    return mask_stack_qc, dbg, ref_mask


def smooth_norm_image(image_stack, x=4, y=3, z=3):
    data = np.stack(image_stack, axis=0).astype(np.float32, copy=False)
    
    # normalize by slice
    for i in range(data.shape[0]):
        frame = data[i]
        fmin = frame.min()
        fmax = frame.max()
        if fmax - fmin > 1e-8:
            data[i] = (frame - fmin) / (fmax - fmin)
        else:
            data[i] = 0.0
    
    # smoothing stack
    smooth_data = gaussian_filter(data, sigma=(x, y, z))
    
    return smooth_data


def deltaF_over_F0_fixed(image_stack, mask_stack, best_mask, baseline_frames=100):
    """
    best_mask: set one mask for all (best_mask > 0)
    baseline_frames: first few frames to calculate f0
    """
    fixed_mask = (best_mask > 0).astype(np.uint8)

    T_min_B = []
    for img in image_stack:
        T_mean = img[fixed_mask == 1].mean()
        B_mean = img[fixed_mask == 0].mean()
        T_min_B.append(T_mean - B_mean)

    T_min_B = np.array(T_min_B)
    f0 = np.median(T_min_B[:baseline_frames])
    if abs(f0) < 1e-12:
        f0 = np.percentile(T_min_B, 10)

    delta_f = T_min_B - f0
    result = delta_f / abs(f0)

    return result


def plot_deltaF_signal(deltaF_over_F0, rise, decay, peak, sigma=2.0, show_smooth=True):
    x = np.arange(rise, decay)
    y_raw = deltaF_over_F0[rise:decay].astype(float)

    plt.figure(figsize=(8, 3))
    plt.plot(x, y_raw, label="raw")  # raw line
    plt.plot(peak, deltaF_over_F0[peak], "x", label="peak")

    plt.xticks(np.arange(rise, decay, 10))
    plt.xlabel("Frame ID")
    plt.ylabel("Signal (ΔF/F₀)")
    plt.title("Calcium Signal Trace")
    plt.tight_layout()
    plt.legend()
    if _pdf is not None:
        _pdf.savefig(plt.gcf())
    plt.show()


def find_peak_boundaries(signal, peaks):
    results = []
    
    for i, peak in enumerate(peaks):
        # rise point: lowest point between previous-current peak
        if i == 0:
            search_start = 0
        else:
            search_start = peaks[i - 1]
        
        rise = search_start + np.argmin(signal[search_start:peak])
        
        # decay end: lowest point between current-next peak
        if i == len(peaks) - 1:
            search_end = len(signal)
        else:
            search_end = peaks[i + 1]
        
        decay = peak + np.argmin(signal[peak:search_end])
        
        results.append({
            'peak_index': i + 1,
            'rise_frame': int(rise),
            'peak_frame': int(peak),
            'decay_frame': int(decay),
        })
    
    return results

def peak_info(deltaF_over_F0, peaks, left_ips, right_ips, plot=True):
    results = []

    for i in range(len(peaks)):
        left_ip = math.floor(left_ips[i])
        peak = int(peaks[i])
        right_ip = math.ceil(right_ips[i])

        print("=================================")
        print(f"▶ Peak {i+1}")
        print(f"  Rise frame: {left_ip}")
        print(f"  Peak frame: {peak}")
        print(f"  Decay frame: {right_ip}")

        if plot:
            plot_deltaF_signal(deltaF_over_F0, left_ip, right_ip, peak)
        
        results.append({
                'peak_index': i+1,
                'rise_frame': left_ip,
                'peak_frame': peak,
                'decay_frame': right_ip,
            })

    return results



def LK_per_peak(video, mask_stack, peaks_info, spacing=10):
    T, H, W = video.shape
    all_trajectories = []

    for idx, peak in enumerate(peaks_info):
        rise = peak['rise_frame']
        decay = peak['decay_frame']

        n_frames = decay - rise + 1
        if n_frames < 2:
            print(f"Skipping peak {idx+1}, too short.")
            continue

        # Initial frame and mask
        initial_frame = (video[rise] * 255).astype(np.uint8)
        initial_mask = (mask_stack[rise] > 0).astype(np.uint8)

        # Make point grid
        ys, xs = np.meshgrid(np.arange(0, H, spacing), np.arange(0, W, spacing), indexing='ij')
        points = np.stack([xs.ravel(), ys.ravel()], axis=-1)
        valid_mask = initial_mask[points[:, 1], points[:, 0]] > 0
        points = points[valid_mask]
        prev_pts = points.astype(np.float32).reshape(-1, 1, 2)

        # Initialize trajectory
        trajectories = [[pt.ravel()] for pt in prev_pts]
        prev_gray = initial_frame.copy()

        for t in range(rise + 1, decay + 1):
            frame_gray = (video[t] * 255).astype(np.uint8)
            mask = (mask_stack[t] > 0).astype(np.uint8)

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, frame_gray, prev_pts, None,
                winSize=(25, 25), maxLevel=3,     # winSize=(15, 15) maxLevel=3. 21
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001),    #    0.01
                flags=0, minEigThreshold=1e-5   
            )
            status = status.ravel()
            next_pts = next_pts.reshape(-1, 2)
            prev_pts = prev_pts.reshape(-1, 2)

            new_pts = []
            for i in range(len(prev_pts)):
                if status[i] == 1:
                    x, y = int(round(next_pts[i][0])), int(round(next_pts[i][1]))

                    # Check if the point is within the mask and image bounds
                    if 0 <= x < W and 0 <= y < H and mask[y, x] > 0:
                        trajectories[i].append(next_pts[i])
                        new_pts.append(next_pts[i])
                    else:
                        # If the point is out of bounds or not in the mask, keep the last point
                        trajectories[i].append(trajectories[i][-1])
                        new_pts.append(trajectories[i][-1])
                else:
                    # If the point is lost, keep the last point
                    trajectories[i].append(trajectories[i][-1])
                    new_pts.append(trajectories[i][-1])

            prev_pts = np.array(new_pts, dtype=np.float32).reshape(-1, 1, 2)
            prev_gray = frame_gray.copy()

        all_trajectories.append(trajectories)  # shape: (N_points, n_frames, 2)

    return all_trajectories




#####################################################################
def build_quiet_windows_from_boundaries(boundaries, win=30, margin=3):
    """
    boundaries: find_peak_boundaries list
    win: quiet window length (frames)
    margin: around peak boundary 
    returns: peaks_info list 
    """
    quiet = []
    if len(boundaries) < 2:
        return quiet

    for i in range(len(boundaries) - 1):
        a = int(boundaries[i]["decay_frame"]) + margin
        b = int(boundaries[i + 1]["rise_frame"]) - margin
        if b <= a + 2:
            continue

        # chop between gaps by win size
        mid = (a + b) // 2
        t0 = max(a, mid - win // 2)
        t1 = min(b, t0 + win - 1)
        if t1 - t0 + 1 < 5:
            continue

        quiet.append({
            "peak_index": i + 1, 
            "rise_frame": int(t0),
            "peak_frame": int((t0 + t1) // 2),
            "decay_frame": int(t1),
        })

    return quiet


def collect_step_sizes_from_trajectories(trajectories_per_window, step_min=0.0):
    """
    trajectories_per_window: LK_per_peak output (list of windows, each is list of traj)
    returns: step size list (pixels)
    """
    steps = []
    for window_trajs in trajectories_per_window:
        for traj in window_trajs:
            tr = np.asarray(traj, dtype=np.float32)
            if tr.ndim != 2 or tr.shape[0] < 2:
                continue
            dx = tr[1:, 0] - tr[:-1, 0]
            dy = tr[1:, 1] - tr[:-1, 1]
            st = np.sqrt(dx * dx + dy * dy)
            if step_min > 0:
                st = st[st > step_min]
            steps.extend(st.tolist())
    return steps


def estimate_noise_floor_px_from_quiet_windows(video, mask_stack, boundaries, spacing=10, win=30, q=95, debug_plot=False):
    quiet_info = build_quiet_windows_from_boundaries(boundaries, win=win)
    if len(quiet_info) == 0:
        return 0.0, {"n_windows": 0, "n_steps": 0}

    quiet_trajs = LK_per_peak(video=video, mask_stack=mask_stack, peaks_info=quiet_info, spacing=spacing)
    steps = collect_step_sizes_from_trajectories(quiet_trajs)

    if len(steps) == 0:
        return 0.0, {"n_windows": len(quiet_info), "n_steps": 0}

    nf = float(np.percentile(np.asarray(steps, dtype=np.float32), q))

    if debug_plot:
        plt.figure(figsize=(6, 3))
        plt.hist(steps, bins=60)
        plt.axvline(nf, linewidth=2)
        plt.title(f"Quiet steps histogram, noise floor p{q} = {nf:.3f} px")
        plt.xlabel("step (px)")
        plt.ylabel("count")
        if _pdf is not None:
            _pdf.savefig(plt.gcf())
        plt.show()

    dbg = {"n_windows": len(quiet_info), "n_steps": len(steps), "q": q, "noise_floor_px": nf}
    return nf, dbg

#####################################################################











def velocity(video, trajectories, fps, start_frame=0, end_frame=None):
    # parameter is set based on binning size, magnification, and pixel size on Mai's thesis
    pixel_size = 0.0069 
    binning_size = 2
    magnification = 2 
    pixel_to_mm = pixel_size * binning_size / magnification  # mm per pixel


    velocities = []
    distances = []
    velocity_info = []
    max_distance = 0
    max_speed = 0

    for i, traj in enumerate(trajectories):
        traj = np.asarray(traj)
        if end_frame is not None:
            traj = traj[start_frame:end_frame]
        else:
            traj = traj[start_frame:]
        x = traj[:, 0] # save x-coordinates for euclidean distance 
        y = traj[:, 1] # save y-coordinates for euclidean distance

        distance = 0
        last_frame_with_motion = 0

        # Calculate distance
        for t in range(1, len(x)):
            dx = x[t] - x[t - 1]
            dy = y[t] - y[t - 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > NOISE_FLOOR_PX:  # 추가
                distance += dist
            #distance += dist
            # if the distance is greater than 0, then we consider it as a motion
            if dist > 0:
                last_frame_with_motion = t

        # Convert distance from pixels to mm
        distance_mm = distance * pixel_to_mm
        seconds = (last_frame_with_motion + 1) / fps
        velocity = 0 if seconds == 0 else distance_mm / seconds

        velocities.append(velocity)
        distances.append(distance_mm)

        velocity_info.append({
            "index": i,
            "velocity": velocity,
            "distance": distance_mm
        })

        # Update max distance and speed
        if distance_mm > max_distance:
            max_distance = distance_mm
        if velocity > max_speed:
            max_speed = velocity

    return velocities, max_distance, max_speed, distances, velocity_info


def analyze_peaks_and_velocity(video, trajectories, peak_info_results, fps=100, plot=True, plot_velocity = True):
    results_out = []

    for i, peak_data in enumerate(peak_info_results):
        left_ip = peak_data['rise_frame']
        peak = peak_data['peak_frame']
        right_ip = peak_data['decay_frame']
        peak_index = peak_data['peak_index']

        sub_video = video[left_ip:right_ip]

        # Make trajectories as array for calcualation
        raw_trajs = np.array(trajectories[i])

        if raw_trajs.ndim != 3 or raw_trajs.shape[2] != 2:
            raise ValueError(f"Expected trajectory shape (N, T, 2), got {raw_trajs.shape}")

        # each point -> (T, 2) trajectory
        traj_list = [raw_trajs[n] for n in range(raw_trajs.shape[0])]

        # Compute velocity metrics
        vels, max_dist, max_spd, distances, velocity_info = velocity(sub_video, traj_list, fps=fps)

        if plot:
            print("=================================")
            print(f"▶ Peak {peak_index}")
            print(f"  Rise frame: {left_ip}")
            print(f"  Peak frame: {peak}")
            print(f"  Decay frame: {right_ip}")
            print(f"  First point speed: {vels[0]:.3f} mm/s")
            print(f"  Max distance: {max_dist:.3f} mm")
            print(f"  Max speed: {max_spd:.3f} mm/s")
            print("=================================\n")

        if plot_velocity:
            # velocity
            plt.figure(figsize=(10, 2.5))
            plt.bar(range(len(vels)), vels, color='red', width=1.0)
            plt.xlabel("Point ID in sparse set")
            plt.ylabel("Velocity (mm/s)")
            plt.title(f"Peak {peak_index} Velocity Distribution")
            plt.tight_layout()
            if _pdf is not None:
                _pdf.savefig(plt.gcf())
            plt.show()

            # distance
            plt.figure(figsize=(10, 2.5))
            plt.bar(range(len(distances)), distances, color='blue', width=1.0)
            plt.xlabel("Point ID in sparse set")
            plt.ylabel("Distance (mm)")
            plt.title(f"Peak {peak_index} Total Distance Traveled")
            plt.tight_layout()
            if _pdf is not None:
                _pdf.savefig(plt.gcf())
            plt.show()

        results_out.append({
            'peak_index': peak_index,
            'rise_frame': left_ip,
            'peak_frame': peak,
            'decay_frame': right_ip,
            'first_speed': round(vels[0], 3),
            'max_distance': round(max_dist, 3),
            'max_speed': round(max_spd, 3),
            'velocity_info': velocity_info
        })

    return results_out


def top3_velocity_and_distance(velocity_info):
    # sort velocity_info by velocity and distance
    sorted_by_speed = sorted(velocity_info, key=lambda x: x['velocity'], reverse=True)
    sorted_by_dist = sorted(velocity_info, key=lambda x: x['distance'], reverse=True)

    # get top 3 ids for speed and distance
    top3_speed_ids = set(v['index'] for v in sorted_by_speed[:3])
    top3_dist_ids = set(v['index'] for v in sorted_by_dist[:3])

    # find ids that are in both top 3 speed and distance
    both_ids = top3_speed_ids & top3_dist_ids
    only_speed_ids = top3_speed_ids - both_ids
    only_dist_ids = top3_dist_ids - both_ids

    return both_ids, only_speed_ids, only_dist_ids


def draw_trajectories(img, trajectories, t_start, t_end, both_ids, only_speed_ids, only_dist_ids):
    thr = float(NOISE_FLOOR_PX)
    if not np.isfinite(thr) or thr <= 0:
        thr = 0.0
        
    # Draw trajectories on the image with different colors based on their categories
    for i, traj in enumerate(trajectories):
        traj = np.array(traj)
        if len(traj) <= t_end:
            continue
        # Subset the trajectory for each peak
        sub = traj[t_start:t_end + 1]

        if i in both_ids:
            track_color = (255, 0, 255)  # purple
            dot_color = (255, 0, 255)
            thickness = 3
        elif i in only_speed_ids:
            track_color = (0, 0, 255)    # red
            dot_color = (0, 0, 255)
            thickness = 2
        elif i in only_dist_ids:
            track_color = (255, 0, 0)    # blue
            dot_color = (255, 0, 0)
            thickness = 2
        else:
            track_color = (0, 255, 255)  # yellow
            dot_color = (255, 255, 255)
            thickness = 1

        # Draw the trajectory for each peak
        for j in range(1, len(sub)):
            x1, y1 = sub[j - 1]
            x2, y2 = sub[j]
            step = np.sqrt((x2-x1)**2 + (y2-y1)**2)   # 
            if step <= thr:                           #
                continue                              #
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), track_color, thickness)

        # Check the last point of the trajectory
        x0, y0 = sub[-1]

        # if those are on top 3 speed or distance, draw a star marker, otherwise draw a normal circle
        if i in both_ids or i in only_speed_ids or i in only_dist_ids:
            cv2.drawMarker(
                img, (int(x0), int(y0)), dot_color,
                markerType=cv2.MARKER_STAR,
                markerSize=14, thickness=2, line_type=cv2.LINE_AA
            )
        else:
            cv2.circle(img, (int(x0), int(y0)), 3, dot_color, -1)


def calc_angle(results_list, peak_id, phase_name, row, col, vectors, disp_mm, velocity_mm_s=0.0):
    angles_in_rad = []
    for dx, dy in vectors:
        if dx == 0 and dy == 0:
            continue
        angle = math.atan2(dy, dx)
        angles_in_rad.append(angle)

    if angles_in_rad:
        #mean_angle_rad = circmean(np.unwrap(angles_in_rad))
        mean_angle_rad = circmean(angles_in_rad)
        mean_angle_deg = math.degrees(mean_angle_rad)
        if mean_angle_deg < 0:
            mean_angle_deg += 360
    else:
        mean_angle_deg = 0.0

    results_list.append({
        'peak_id': peak_id,
        'phase': phase_name,
        'row': row,
        'col': col,
        'mean_angle_deg': round(mean_angle_deg, 2),
        'mean_disp_mm' : round(disp_mm, 4),
        'mean_vel_mm_s': round(velocity_mm_s, 4),
    })

    return mean_angle_deg


def get_bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("mask is empty")
    y_top = int(ys.min())
    y_bottom = int(ys.max())
    x_left = int(xs.min())
    x_right = int(xs.max())
    return x_left, x_right, y_top, y_bottom


def build_bbox_grid(mask, num_rows, num_cols):
    """
    divide inside of bbox by num_rows x num_cols 
    GRID_CELLS: save each cells as rectangle polygon (TL TR BR BL)
    GRID_BBOXES: each cell bbox (xmin,xmax,ymin,ymax)
    GRID_LINES: draw lines
    """
    x_left, x_right, y_top, y_bottom = get_bbox_from_mask(mask)

    row_ys = np.linspace(float(y_top), float(y_bottom), num_rows + 1)
    col_xs = np.linspace(float(x_left), float(x_right), num_cols + 1)

    cells = [[None for _ in range(num_cols)] for __ in range(num_rows)]
    bboxes = [[None for _ in range(num_cols)] for __ in range(num_rows)]

    for r in range(num_rows):
        y0 = row_ys[r]
        y1 = row_ys[r + 1]
        for c in range(num_cols):
            x0 = col_xs[c]
            x1 = col_xs[c + 1]

            poly = np.array([
                (x0, y0),  # TL
                (x1, y0),  # TR
                (x1, y1),  # BR
                (x0, y1),  # BL
            ], dtype=np.float64)

            cells[r][c] = poly
            bboxes[r][c] = (float(x0), float(x1), float(y0), float(y1))

    # draw lines
    H_lines = []
    for r in range(num_rows + 1):
        y = float(row_ys[r])
        H_lines.append(((float(x_left), y), (float(x_right), y)))

    V_lines = []
    for c in range(num_cols + 1):
        x = float(col_xs[c])
        V_lines.append(((x, float(y_top)), (x, float(y_bottom))))

    return cells, bboxes, (H_lines, V_lines)



def _polygon_area(poly):
    # poly: (N,2)
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _ensure_ccw(poly):
    # returns CCW ordered polygon
    if _polygon_area(poly) < 0:
        return poly[::-1].copy()
    return poly


def _cyrus_beck_clip_segment_to_convex_polygon(p0, p1, poly):
    """
    Convex polygon clip using Cyrus Beck.
    poly must be CCW.
    Returns (q0, q1) or None.
    """
    poly = _ensure_ccw(poly)
    d = p1 - p0

    t0 = 0.0
    t1 = 1.0

    nverts = poly.shape[0]
    for i in range(nverts):
        a = poly[i]
        b = poly[(i + 1) % nverts]
        e = b - a

        # inward normal for CCW polygon is left normal
        n = np.array([-e[1], e[0]], dtype=np.float64)

        f0 = np.dot(n, p0 - a)
        denom = np.dot(n, d)

        if abs(denom) < 1e-12:
            # segment parallel to edge
            if f0 < 0:
                return None
            continue

        t = -f0 / denom

        if denom > 0:
            # entering
            if t > t0:
                t0 = t
        else:
            # leaving
            if t < t1:
                t1 = t

        if t0 > t1:
            return None

    q0 = p0 + t0 * d
    q1 = p0 + t1 * d
    return q0, q1



def get_grid_boundaries(mask, num_rows, num_cols):
    """
    H_lines: length num_rows+1, each is ((xL,y),(xR,y)) but xL,xR can drift with row
    V_lines: length num_cols+1, each is ((xT,y_top),(xB,y_bottom)) and can be slanted
    """
    ys, xs = np.where(mask > 0)
    y_top = int(ys.min())
    y_bottom = int(ys.max())

    xs_top = np.where(mask[y_top] > 0)[0]
    xs_bot = np.where(mask[y_bottom] > 0)[0]

    x_top_left = float(xs_top.min())
    x_top_right = float(xs_top.max())
    x_bot_left = float(xs_bot.min())
    x_bot_right = float(xs_bot.max())

    # point of top and bottom for col line
    top_xs = np.linspace(x_top_left, x_top_right, num_cols + 1)
    bot_xs = np.linspace(x_bot_left, x_bot_right, num_cols + 1)

    # vertical line: (top_x, y_top) -> (bot_x, y_bottom)
    V_lines = []
    for c in range(num_cols + 1):
        V_lines.append(((float(top_xs[c]), float(y_top)),
                        (float(bot_xs[c]), float(y_bottom))))

    # point of left and right for col line
    y_rows = np.linspace(float(y_top), float(y_bottom), num_rows + 1)
    left_xs = np.linspace(x_top_left, x_bot_left, num_rows + 1)
    right_xs = np.linspace(x_top_right, x_bot_right, num_rows + 1)

    # horizontal line: (left_x, y) -> (right_x, y)
    H_lines = []
    for r in range(num_rows + 1):
        H_lines.append(((float(left_xs[r]), float(y_rows[r])),
                        (float(right_xs[r]), float(y_rows[r]))))

    return H_lines, V_lines



def _line_intersection(p1, p2, p3, p4):
    """
    Intersection of infinite lines (p1,p2) and (p3,p4).
    returns (x,y) float.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        # if it is almost horizontal, get midpoint
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (float(px), float(py))


def _build_skewed_grid_from_mask(mask, num_rows, num_cols):
    m = (mask > 0).astype(np.uint8)
    return build_bbox_grid(m, num_rows, num_cols)


def clip_vector_to_cell_boundary(x1, y1, x2, y2, x0, x1_lim, y0, y1_lim):
    global GRID_CELLS

    if GRID_CELLS is not None:
        row = int(x0)
        col = int(x1_lim)

        if row < 0 or col < 0:
            return x1, y1, x1, y1
        if row >= len(GRID_CELLS) or col >= len(GRID_CELLS[0]):
            return x1, y1, x1, y1

        poly = GRID_CELLS[row][col]
        p0 = np.array([float(x1), float(y1)], dtype=np.float64)
        p1 = np.array([float(x2), float(y2)], dtype=np.float64)

        clipped = _cyrus_beck_clip_segment_to_convex_polygon(p0, p1, poly)
        if clipped is None:
            return x1, y1, x1, y1

        q0, q1 = clipped
        return float(q0[0]), float(q0[1]), float(q1[0]), float(q1[1])

    if x1 == x2:
        if not (x0 <= x1 <= x1_lim):
            return x1, y1, x1, y1
        y1_clip = builtins.max(builtins.min(y1, y1_lim), y0)
        y2_clip = builtins.max(builtins.min(y2, y1_lim), y0)
        return x1, y1_clip, x2, y2_clip

    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1

    new_x1 = builtins.max(builtins.min(x1, x1_lim), x0)
    new_x2 = builtins.max(builtins.min(x2, x1_lim), x0)
    new_y1 = m * new_x1 + c
    new_y2 = m * new_x2 + c

    if not (y0 <= new_y1 <= y1_lim):
        new_y1 = builtins.max(builtins.min(new_y1, y1_lim), y0)
        new_x1 = (new_y1 - c) / m
    if not (y0 <= new_y2 <= y1_lim):
        new_y2 = builtins.max(builtins.min(new_y2, y1_lim), y0)
        new_x2 = (new_y2 - c) / m

    return new_x1, new_y1, new_x2, new_y2


def clip_vector_to_cell_boundary_by_index(x1, y1, x2, y2, row, col):
    global GRID_CELLS
    if GRID_CELLS is None:
        return x1, y1, x1, y1

    if row < 0 or col < 0:
        return x1, y1, x1, y1
    if row >= len(GRID_CELLS) or col >= len(GRID_CELLS[0]):
        return x1, y1, x1, y1

    poly = GRID_CELLS[row][col]
    p0 = np.array([float(x1), float(y1)], dtype=np.float64)
    p1 = np.array([float(x2), float(y2)], dtype=np.float64)

    clipped = _cyrus_beck_clip_segment_to_convex_polygon(p0, p1, poly)
    if clipped is None:
        return x1, y1, x1, y1

    q0, q1 = clipped
    return float(q0[0]), float(q0[1]), float(q1[0]), float(q1[1])




    

def draw_grid_img(img, mask, num_rows, num_cols):
    global GRID_LINES

    if GRID_LINES is None:
        # fallback: call original lines
        ys, xs = np.where(mask > 0)
        y_top = int(ys.min())
        y_bottom = int(ys.max())
        x_left = int(xs.min())
        x_right = int(xs.max())

        row_boundaries = np.linspace(y_top, y_bottom, num_rows + 1)
        col_boundaries = np.linspace(x_left, x_right, num_cols + 1)

        for col in range(1, num_cols):
            x = int(col_boundaries[col])
            cv2.line(img, (x, int(y_top)), (x, int(y_bottom)), (255, 255, 255), 2)

        for row in range(1, num_rows):
            y = int(row_boundaries[row])
            cv2.line(img, (int(x_left), y), (int(x_right), y), (255, 255, 255), 2)
        return

    H_lines, V_lines = GRID_LINES

    # vertical lines
    for c in range(1, num_cols):
        p_top, p_bot = V_lines[c]
        cv2.line(img,
                 (int(p_top[0]), int(p_top[1])),
                 (int(p_bot[0]), int(p_bot[1])),
                 (255, 255, 255), 2)

    # horizontal lines
    for r in range(1, num_rows):
        p_left, p_right = H_lines[r]
        cv2.line(img,
                 (int(p_left[0]), int(p_left[1])),
                 (int(p_right[0]), int(p_right[1])),
                 (255, 255, 255), 2)





    
def draw_average_arrows_on_grid(
        img, trajectories, t_start, t_end,
        num_rows, num_cols,
        arrow_color, peak_id, phase_name,
        result_list=None):

    global tissue_mask, GRID_CELLS, GRID_BBOXES

    mask = tissue_mask

    if GRID_CELLS is None or GRID_BBOXES is None:
        GRID_CELLS, GRID_BBOXES, _ = _build_skewed_grid_from_mask(mask, num_rows, num_cols)

    pixel_size = 0.0069
    binning_factor = 1
    magnification = 1
    pixel_to_mm = (pixel_size * binning_factor) / magnification

    for row in range(num_rows):
        for col in range(num_cols):
            xmin, xmax, ymin, ymax = GRID_BBOXES[row][col]

            net_vectors = []  # (dx, dy) per trajectory

            for traj in trajectories:
                traj = np.array(traj)
                if len(traj) <= t_end:
                    continue

                # check is the starting point is in this cell
                start_pt = traj[t_start]
                sx, sy = start_pt[0], start_pt[1]
                if sx < xmin or sx > xmax or sy < ymin or sy > ymax:
                    continue

                # net displacement (start → end) - direction
                end_pt = traj[t_end]
                dx = end_pt[0] - start_pt[0]
                dy = end_pt[1] - start_pt[1]

                thr = float(NOISE_FLOOR_PX)
                if not np.isfinite(thr) or thr <= 0:
                    thr = 0.05  # px fallback

                # cumulative displacement — velocity
                dx, dy = 0.0, 0.0
                for k in range(t_start, min(t_end, len(traj) - 1)):
                    cx = traj[k+1][0] - traj[k][0]
                    cy = traj[k+1][1] - traj[k][1]
                    step = np.sqrt(cx**2 + cy**2)
                    if step > thr:
                        dx += cx
                        dy += cy
                        
                dist = np.sqrt(dx**2 + dy**2)

                # get rid of not moved point
                if dist < 0.5:
                    continue

                net_vectors.append((dx, dy))

            if net_vectors:
                dxs, dys = zip(*net_vectors)
                mean_dx = np.mean(dxs)
                mean_dy = np.mean(dys)

                disp = np.sqrt(mean_dx**2 + mean_dy**2)
                disp_mm = disp * pixel_to_mm

                n_frames = t_end - t_start
                fps = 100.0
                velocity_mm_s = disp_mm / max(n_frames / fps, 1e-9)


                mean_angle_deg = calc_angle(
                    results_list=result_list,
                    peak_id=peak_id,
                    phase_name=phase_name,
                    row=row,
                    col=col,
                    vectors=net_vectors,
                    # vectors=[(dx, dy) for dx, dy, _ in net_vectors],
                    disp_mm=disp_mm,
                    velocity_mm_s=velocity_mm_s
                )

                # draw arrows
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                scale = 8  # scale it!
                p1 = (int(cx), int(cy))
                p2 = (int(cx + mean_dx * scale), int(cy + mean_dy * scale))
                cv2.arrowedLine(img, p1, p2, arrow_color, 2, tipLength=0.4)

                print(
                    f"Peak {peak_id} {phase_name} - Cell({row},{col}) "
                    f"→ Disp {disp:.2f}px ({disp_mm:.4f}mm), Vel {velocity_mm_s:.4f}mm/s, "
                    f"Angle {mean_angle_deg:.1f}°, n={len(net_vectors)}, noise_thr={NOISE_FLOOR_PX:.3f}px"
                )


def generate_phase_image(video, frame_index, trajectories, t_start, t_end,
                         both_ids, only_speed_ids, only_dist_ids,
                         num_rows, num_cols, arrow_color,
                         peak_id, phase_name, result_list):

    global tissue_mask, GRID_CELLS, GRID_BBOXES, GRID_LINES

    if GRID_CELLS is None or GRID_BBOXES is None or GRID_LINES is None:
        raise RuntimeError("Grid not initialized. Run QC and build grid from best_mask first.")

    bg = (video[frame_index] * 255).astype(np.uint8)
    img_rgb = np.stack([bg] * 3, axis=-1).copy()

    draw_trajectories(img_rgb, trajectories, t_start, t_end, both_ids, only_speed_ids, only_dist_ids)
    draw_average_arrows_on_grid(
        img_rgb, trajectories, t_start, t_end,
        num_rows, num_cols, arrow_color,
        peak_id, phase_name, result_list
    )
    draw_grid_img(img_rgb, tissue_mask, num_rows, num_cols)

    return img_rgb



def save_phase_images(video, trajectories, peak_info, output_dir, both_ids, only_speed_ids, only_dist_ids, num_rows, num_cols):
    rise = peak_info['rise_frame']
    peak = peak_info['peak_frame']
    decay = peak_info['decay_frame']
    peak_id = peak_info['peak_index']

    phases = [
        {"name": "rise", 
         "start": 0, 
         "end": peak - rise, 
         "bg_index": peak, 
         "color": (0, 255, 0)},
        {"name": "decay", 
         "start": peak - rise, 
         "end": decay - rise, 
         "bg_index": decay, 
         "color": (0, 0, 255)}
    ]

    results = []

    for phase in phases:
        img = generate_phase_image(
            video, phase["bg_index"], trajectories,
            phase["start"], phase["end"],
            both_ids, only_speed_ids, only_dist_ids,
            num_rows, num_cols,
            phase["color"], peak_id, phase["name"],
            result_list=results
        )
        out_path = os.path.join(output_dir, f"peak_{peak_id}_{phase['name']}.png")
        cv2.imwrite(out_path, img)
    
    return results



def save_phase_video(video, trajectories, rise, decay, output_path):
    H, W = video.shape[1:]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for f in range(rise, decay + 1):
        bg = (video[f] * 255).astype(np.uint8)
        img_rgb = np.stack([bg] * 3, axis=-1).copy()

        for traj in trajectories:
            traj = np.array(traj)
            if len(traj) <= (f - rise):
                continue
            sub = traj[:f - rise + 1]
            if len(sub) < 2:
                continue

            for j in range(1, len(sub)):
                x1, y1 = sub[j - 1]
                x2, y2 = sub[j]
                step = np.sqrt((x2-x1)**2 + (y2-y1)**2) #
                if step <= NOISE_FLOOR_PX:              #
                    continue                            # 
                cv2.line(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)

            x0, y0 = sub[-1]
            cv2.circle(img_rgb, (int(x0), int(y0)), 3, (255, 255, 255), -1)

        writer.write(img_rgb)

    writer.release()
    print(f"Saved video: {output_path}")


def generate_rise_decay_flow_images_for_peaks(video, trajectories_per_peak, results, output_dir, num_rows=3, num_cols=4, top_3=False):
    os.makedirs(output_dir, exist_ok=True)

    
    global NOISE_FLOOR_PX
    if NOISE_FLOOR_PX <= 0:
        NOISE_FLOOR_PX = compute_noise_floor(trajectories_per_peak, results)
    if NOISE_FLOOR_PX <= 0:
        NOISE_FLOOR_PX = 0.3  # safe fallback

    print(f"Noise floor: {NOISE_FLOOR_PX:.3f} px")

    
    all_direction_results = []

    for peak_idx, result in enumerate(results):
        rise = result['rise_frame']
        peak = result['peak_frame']
        decay = result['decay_frame']
        peak_id = result['peak_index']
        velocity_info = result['velocity_info']
        trajectories = trajectories_per_peak[peak_idx]

        if top_3:
            both_ids, only_speed_ids, only_dist_ids = top3_velocity_and_distance(velocity_info)
        else:
            both_ids, only_speed_ids, only_dist_ids = set(), set(), set()

        direction_results = save_phase_images(
            video, trajectories, result, output_dir,
            both_ids, only_speed_ids, only_dist_ids,
            num_rows, num_cols
        )
        all_direction_results.extend(direction_results)

        # video_path = os.path.join(output_dir, f"peak_{peak_id}_trajectory.mp4")
        # save_phase_video(video, trajectories, rise, decay, video_path)
        # select only 3 videos
        # if peak_idx in [0, len(results)//2, len(results)-1]:
        #     video_path = os.path.join(output_dir, f"peak_{peak_id}_trajectory.mp4")
        #     save_phase_video(video, trajectories, rise, decay, video_path)

    return all_direction_results


def show_all_rise_decay(input_dir, num_peaks=24, cols=2):
    rows = num_peaks
    fig, axs = plt.subplots(rows, cols, figsize=(8, rows * 2))

    for i in range(1, num_peaks + 1):
        rise_path = os.path.join(input_dir, f"peak_{i}_rise.png")
        decay_path = os.path.join(input_dir, f"peak_{i}_decay.png")

        if not os.path.exists(rise_path) or not os.path.exists(decay_path):
            print(f"Missing images for peak {i}")
            continue

        rise_img = cv2.imread(rise_path)
        decay_img = cv2.imread(decay_path)

        rise_img = cv2.cvtColor(rise_img, cv2.COLOR_BGR2RGB)
        decay_img = cv2.cvtColor(decay_img, cv2.COLOR_BGR2RGB)

        axs[i - 1, 0].imshow(rise_img)
        axs[i - 1, 0].axis('off')
        axs[i - 1, 0].set_title(f"Peak {i} - Rise")

        axs[i - 1, 1].imshow(decay_img)
        axs[i - 1, 1].axis('off')
        axs[i - 1, 1].set_title(f"Peak {i} - Decay")

    plt.tight_layout()
    if _pdf is not None:
        _pdf.savefig(plt.gcf())
    plt.show()


def plot_unwrapped_direction_by_region(directions):
    region_labels = {
        (0, 0): "Top left",
        (0, 1): "Top right",

        (1, 0): "Middle left",
        (1, 1): "Middle right",

        (2, 0): "Bottom left",
        (2, 1): "Bottom right"
    }

    region_colors = {
        "Top left": 'magenta',
        "Top right": 'gold',
        "Middle left": 'cyan',
        "Middle right": 'blue',
        "Bottom left": 'brown',
        "Bottom right": 'green'
    }

    phases = ['rise', 'decay']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for i, phase in enumerate(phases):
        ax = axes[i]
        ax.set_title(f"Signal {phase}")
        ax.set_xlabel("Calcium signal ID")
        ax.set_ylabel("Angle")

        for (row, col), label in region_labels.items():
            subset = [d for d in directions if d["phase"] == phase and d["row"] == row and d["col"] == col]
            if not subset:
                continue

            # order py peak id
            subset.sort(key=lambda d: d["peak_id"])
            x = [d["peak_id"] for d in subset]
            y_raw = np.array([d["mean_angle_deg"] % 360 for d in subset])
            center = np.degrees(circmean(np.radians(y_raw)))
            y = ((y_raw - center + 180) % 360 - 180 + center).tolist()
            
            
            ax.plot(
                x, y,
                marker="o",
                linestyle="-",
                linewidth=1,
                label=label,
                color=region_colors[label],
                markersize=3
            )

    axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    if _pdf is not None:
        _pdf.savefig(plt.gcf())
    plt.show()




def plot_displacement_bar_charts(results):
    import matplotlib.pyplot as plt

    region_labels = {
        (0, 0): "Top left",
        (0, 1): "Top right",

        (1, 0): "Middle left",
        (1, 1): "Middle right",

        (2, 0): "Bottom left",
        (2, 1): "Bottom right"
    }

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, ((row, col), label) in enumerate(region_labels.items()):
        ax = axes[idx]
        rise = [r for r in results if r["row"] == row and r["col"] == col and r["phase"] == "rise"]
        decay = [r for r in results if r["row"] == row and r["col"] == col and r["phase"] == "decay"]

        # sort by peak_id
        rise.sort(key=lambda d: d["peak_id"])
        decay.sort(key=lambda d: d["peak_id"])

        x = [r["peak_id"] for r in rise]
        rise_y = [r["mean_vel_mm_s"] for r in rise]
        decay_y = [d["mean_vel_mm_s"] for d in decay]

        n_min = min(len(rise_y), len(decay_y))
        rise_y = rise_y[:n_min]
        decay_y = decay_y[:n_min]
        x = range(n_min)

        width = 0.35
        ax.bar([i - width/2 for i in x], rise_y, width=width, color='navy', label='Signal rise')
        ax.bar([i + width/2 for i in x], decay_y, width=width, color='limegreen', label='Signal decay')

        ax.set_title(f"{label} - Contraction/Relaxation Velocity")
        ax.set_xlabel("Calcium signal ID")
        ax.set_ylabel("Velocity (mm/s)")
        ax.set_ylim(0, max(rise_y + decay_y + [0.01]) * 1.2)  # auto scale
        ax.set_xticks(x)
        #ax.set_ylim(0, max(rise_y + decay_y + [0.1]))
        ax.set_ylim(0, max(rise_y + decay_y + [0.01]) * 1.2)

    axes[0].legend(loc='upper right')
    plt.tight_layout()
    if _pdf is not None:
        _pdf.savefig(plt.gcf())
    plt.show()


def circ_mean_deg(angle_deg_list):
    """
    circular mean in degrees, output range 0..360
    """
    if len(angle_deg_list) == 0:
        return np.nan
    ang = np.deg2rad(np.array(angle_deg_list, dtype=np.float64))
    c = np.mean(np.cos(ang))
    s = np.mean(np.sin(ang))
    m = np.arctan2(s, c)
    mdeg = np.rad2deg(m) % 360.0
    return float(mdeg)

def circ_R(angle_deg_list):
    """
    circular concentration R in 0..1
    1 means highly consistent direction
    0 means random
    """
    if len(angle_deg_list) == 0:
        return np.nan
    ang = np.deg2rad(np.array(angle_deg_list, dtype=np.float64))
    c = np.mean(np.cos(ang))
    s = np.mean(np.sin(ang))
    R = np.sqrt(c * c + s * s)
    return float(R)

def summarize_region_phase(results):
    """
    results: list of dicts, each has at least:
      row, col, phase ("rise" or "decay"),
      mean_angle_deg, mean_disp_mm
    returns: dict summary[(row,col,phase)] = stats dict
    """
    groups = {}
    for r in results:
        key = (int(r["row"]), int(r["col"]), str(r["phase"]))
        if key not in groups:
            groups[key] = {"angles": [], "disp": [], "vel": [], "peak_ids": []}
        # angle
        if "mean_angle_deg" in r and r["mean_angle_deg"] is not None:
            groups[key]["angles"].append(float(r["mean_angle_deg"]) % 360.0)
        # displacement
        if "mean_disp_mm" in r and r["mean_disp_mm"] is not None:
            groups[key]["disp"].append(float(r["mean_disp_mm"]))
        # velocity
        if "mean_vel_mm_s" in r and r["mean_vel_mm_s"] is not None:
            groups[key]["vel"].append(float(r["mean_vel_mm_s"]))
        # peak id optional
        if "peak_id" in r:
            groups[key]["peak_ids"].append(int(r["peak_id"]))

    summary = {}
    for key, v in groups.items():
        angles = v["angles"]
        disp = v["disp"]
        vel = v["vel"]
        stats = {
            "n": int(len(angles)),
            "mean_angle_deg": circ_mean_deg(angles),
            "R": circ_R(angles),
            "mean_disp_mm": float(np.mean(disp)) if len(disp) else np.nan,
            "std_disp_mm": float(np.std(disp, ddof=1)) if len(disp) > 1 else (0.0 if len(disp) == 1 else np.nan),
            "mean_vel_mm_s": float(np.mean(vel)) if len(vel) else np.nan,
            "std_vel_mm_s": float(np.std(vel, ddof=1)) if len(vel) > 1 else (0.0 if len(vel) == 1 else np.nan),
        }
        summary[key] = stats

    return summary

def print_summary_table(summary, num_rows=3, num_cols=2):
    """
    prints a compact table for rise and decay
    """
    def fmt(x, nd=3):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{x:.{nd}f}"

    phases = ["rise", "decay"]
    for phase in phases:
        print("\n====", phase.upper(), "====")
        print("row col | n | mean_angle_deg | R | mean_disp_mm | std_disp_mm")
        for row in range(num_rows):
            for col in range(num_cols):
                key = (row, col, phase)
                if key not in summary:
                    print(f"{row:>3} {col:>3} | 0 | nan | nan | nan | nan")
                    continue
                s = summary[key]
                print(
                    f"{row:>3} {col:>3} | {s['n']:>2} | {fmt(s['mean_angle_deg'],1):>13} | {fmt(s['R'],3):>5} | {fmt(s['mean_disp_mm'],6):>11} | {fmt(s['std_disp_mm'],6):>10}"
                )

def plot_summary_R_and_disp(summary, num_rows=3, num_cols=2, title_prefix=""):
    phases = ["rise", "decay"]

    R_maps = {}
    D_maps = {}
    for phase in phases:
        Rm = np.full((num_rows, num_cols), np.nan, dtype=np.float64)
        Dm = np.full((num_rows, num_cols), np.nan, dtype=np.float64)
        for row in range(num_rows):
            for col in range(num_cols):
                key = (row, col, phase)
                if key in summary:
                    Rm[row, col] = summary[key]["R"]
                    Dm[row, col] = summary[key]["mean_disp_mm"]
        R_maps[phase] = Rm
        D_maps[phase] = Dm

    V_maps = {}
    for phase in phases:
        Vm = np.full((num_rows, num_cols), np.nan, dtype=np.float64)
        for row in range(num_rows):
            for col in range(num_cols):
                key = (row, col, phase)
                if key in summary:
                    Vm[row, col] = summary[key].get("mean_vel_mm_s", np.nan)
        V_maps[phase] = Vm


    # plot R
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    for i, phase in enumerate(phases):
        ax = axes[i]
        im = ax.imshow(R_maps[phase], vmin=0.0, vmax=1.0)
        ax.set_title(f"{title_prefix} R {phase}".strip())
        ax.set_xticks(range(num_cols))
        ax.set_yticks(range(num_rows))
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        for r in range(num_rows):
            for c in range(num_cols):
                val = R_maps[phase][r, c]
                txt = "nan" if np.isnan(val) else f"{val:.2f}"
                ax.text(c, r, txt, ha="center", va="center")
    fig.colorbar(im, ax=axes, location="right", shrink=0.9)
    if _pdf is not None:
        _pdf.savefig(plt.gcf())
    plt.show()

    # plot displacement
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    for i, phase in enumerate(phases):
        ax = axes[i]
        im = ax.imshow(D_maps[phase])
        ax.set_title(f"{title_prefix} mean disp mm {phase}".strip())
        ax.set_xticks(range(num_cols))
        ax.set_yticks(range(num_rows))
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        for r in range(num_rows):
            for c in range(num_cols):
                val = D_maps[phase][r, c]
                txt = "nan" if np.isnan(val) else f"{val:.4f}"
                ax.text(c, r, txt, ha="center", va="center")
    fig.colorbar(im, ax=axes, location="right", shrink=0.9)
    if _pdf is not None:
        _pdf.savefig(plt.gcf())
    plt.show()


    # velocity plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    for i, phase in enumerate(phases):
        ax = axes[i]
        im = ax.imshow(V_maps[phase])
        ax.set_title(f"{title_prefix} mean vel mm/s {phase}".strip())
        ax.set_xticks(range(num_cols))
        ax.set_yticks(range(num_rows))
        ax.set_xlabel("col"); ax.set_ylabel("row")
        for r in range(num_rows):
            for c in range(num_cols):
                val = V_maps[phase][r, c]
                txt = "nan" if np.isnan(val) else f"{val:.4f}"
                ax.text(c, r, txt, ha="center", va="center")
    fig.colorbar(im, ax=axes, location="right", shrink=0.9)
    if _pdf is not None:
        _pdf.savefig(plt.gcf())
    plt.show()



def run_one_case_quick_analysis(results, num_rows=3, num_cols=2, title_prefix=""):
    """
    main entry
    """
    summary = summarize_region_phase(results)
    print_summary_table(summary, num_rows=num_rows, num_cols=num_cols)
    plot_summary_R_and_disp(summary, num_rows=num_rows, num_cols=num_cols, title_prefix=title_prefix)
    return summary





def _circ_mean_deg(angles_deg, weights=None):
    """Circular mean in degrees. angles_deg can be any real; wraps internally."""
    a = np.radians(np.asarray(angles_deg, dtype=np.float64) % 360.0)
    if weights is None:
        weights = np.ones_like(a, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    C = np.sum(weights * np.cos(a))
    S = np.sum(weights * np.sin(a))
    mu = np.degrees(np.arctan2(S, C)) % 360.0
    return float(mu)

def _circ_diff_deg(a_deg, b_deg):
    """Signed smallest circular difference a-b in degrees, result in [-180, 180]."""
    d = (float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0
    return float(d)



def summarize_region_phase_coup(direction, num_rows=3, num_cols=2):
    """
    Returns dict keyed by (row, col, phase) with:
      n, mean_angle_deg (circular), R, mean_disp_mm, std_disp_mm
    """
    out = {}
    for phase in ("rise", "decay"):
        for r in range(num_rows):
            for c in range(num_cols):
                subset = [d for d in direction
                          if d.get("phase") == phase and d.get("row") == r and d.get("col") == c]
                if not subset:
                    continue

                ang = np.array([float(d["mean_angle_deg"]) for d in subset], dtype=np.float64) % 360.0
                disp = np.array([float(d.get("mean_disp_mm", np.nan)) for d in subset], dtype=np.float64)
                vel = np.array([float(d.get("mean_vel_mm_s", np.nan)) for d in subset], dtype=np.float64)

                # circular mean + resultant length R
                a = np.radians(ang)
                C = np.mean(np.cos(a))
                S = np.mean(np.sin(a))
                R = float(np.sqrt(C*C + S*S))
                mean_angle = float(np.degrees(np.arctan2(S, C)) % 360.0)

                out[(r, c, phase)] = {
                    "n": int(len(subset)),
                    "mean_angle_deg": mean_angle,
                    "R": R,
                    "mean_disp_mm": float(np.nanmean(disp)),
                    "std_disp_mm": float(np.nanstd(disp)),
                    "mean_vel_mm_s": float(np.nanmean(vel)), 
                    "std_vel_mm_s": float(np.nanstd(vel)),  
                }
    return out



def compute_left_right_deltas(summary, num_rows=3, num_cols=2):
    """
    Assumes num_cols==2 (col 0 = left, col 1 = right).
    Returns:
      deltas[phase] = dict with arrays per row for:
        delta_R (L-R), delta_disp (L-R), delta_angle_deg (signed, L-R, circular)
      plus global means across rows.
    """
    assert num_cols == 2, "This helper assumes 2 columns: left(0), right(1)."

    deltas = {}
    for phase in ("rise", "decay"):
        dR = []
        dD = []
        dA = []
        dV = []
        rows_used = []
        for r in range(num_rows):
            kL = (r, 0, phase)
            kR = (r, 1, phase)
            if (kL not in summary) or (kR not in summary):
                continue
            RL = summary[kL]["R"]; RR = summary[kR]["R"]
            DL = summary[kL]["mean_disp_mm"]; DR = summary[kR]["mean_disp_mm"]
            AL = summary[kL]["mean_angle_deg"]; AR = summary[kR]["mean_angle_deg"]
            VL = summary[kL].get("mean_vel_mm_s", np.nan)
            VR = summary[kR].get("mean_vel_mm_s", np.nan)

            dR.append(float(RL - RR))
            dD.append(float(DL - DR))
            dA.append(_circ_diff_deg(AL, AR))  # signed circular difference
            dV.append(float(VL - VR))
            rows_used.append(r)

        deltas[phase] = {
            "rows": rows_used,
            "delta_R_by_row": np.array(dR, dtype=np.float64),
            "delta_disp_by_row": np.array(dD, dtype=np.float64),
            "delta_angle_deg_by_row": np.array(dA, dtype=np.float64),
            "delta_R_mean": float(np.nanmean(dR)) if len(dR) else np.nan,
            "delta_disp_mean": float(np.nanmean(dD)) if len(dD) else np.nan,
            "delta_angle_deg_mean": float(_circ_mean_deg(dA)) if len(dA) else np.nan,  # mean of differences (circular)
            "delta_vel_by_row": np.array(dV, dtype=np.float64),
            "delta_vel_mean": float(np.nanmean(dV)) if len(dV) else np.nan,
        }
    return deltas


def compute_rise_decay_coupling(direction, num_rows=3, num_cols=2):
    """
    Per (row,col), match rise and decay by peak_id.
    Outputs coupling metrics:
      - corr_disp: Pearson corr between rise disp and decay disp across peaks
      - corr_angle: circular-ish proxy: corr of cos/sin components (average)
      - mean_abs_angle_diff_deg: mean abs circular diff between rise and decay angles
      - mean_disp_ratio: mean(decay_disp / rise_disp)
    Returns dict keyed by (row,col) and also 'global'.
    """
    out = {}

    for r in range(num_rows):
        for c in range(num_cols):
            rise = [d for d in direction if d.get("phase") == "rise" and d.get("row") == r and d.get("col") == c]
            decay = [d for d in direction if d.get("phase") == "decay" and d.get("row") == r and d.get("col") == c]
            if (not rise) or (not decay):
                continue

            rise_by_id = {int(d["peak_id"]): d for d in rise if "peak_id" in d}
            decay_by_id = {int(d["peak_id"]): d for d in decay if "peak_id" in d}
            common = sorted(set(rise_by_id.keys()) & set(decay_by_id.keys()))
            if len(common) < 3:
                continue

            rise_disp = np.array([float(rise_by_id[i].get("mean_disp_mm", np.nan)) for i in common], dtype=np.float64)
            decay_disp = np.array([float(decay_by_id[i].get("mean_disp_mm", np.nan)) for i in common], dtype=np.float64)
            rise_vel = np.array([float(rise_by_id[i].get("mean_vel_mm_s", np.nan)) for i in common], dtype=np.float64)
            decay_vel = np.array([float(decay_by_id[i].get("mean_vel_mm_s", np.nan)) for i in common], dtype=np.float64)

            rise_ang = np.array([float(rise_by_id[i]["mean_angle_deg"]) % 360.0 for i in common], dtype=np.float64)
            decay_ang = np.array([float(decay_by_id[i]["mean_angle_deg"]) % 360.0 for i in common], dtype=np.float64)

            # displacement coupling (Pearson)
            def _pearson(x, y):
                x = np.asarray(x); y = np.asarray(y)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 3:
                    return np.nan
                x = x[ok]; y = y[ok]
                if (np.std(x) < 1e-12) or (np.std(y) < 1e-12):
                    return np.nan
                return float(np.corrcoef(x, y)[0, 1])

            corr_disp = _pearson(rise_disp, decay_disp)

            # angle coupling: correlate cos and sin separately, then average
            rise_rad = np.radians(rise_ang)
            decay_rad = np.radians(decay_ang)
            corr_cos = _pearson(np.cos(rise_rad), np.cos(decay_rad))
            corr_sin = _pearson(np.sin(rise_rad), np.sin(decay_rad))
            corr_angle = float(np.nanmean([corr_cos, corr_sin]))

            # per-peak circular diffs (rise - decay)
            diffs = np.array([_circ_diff_deg(a, b) for a, b in zip(rise_ang, decay_ang)], dtype=np.float64)
            mean_abs_angle_diff = float(np.nanmean(np.abs(diffs)))

            # disp ratio decay/rise (robust: avoid divide by 0)
            eps = 1e-12
            ratio = decay_disp / (rise_disp + eps)
            mean_disp_ratio = float(np.nanmean(ratio[np.isfinite(ratio)]))
            vel_ratio = decay_vel / (rise_vel + eps)
            mean_vel_ratio = float(np.nanmean(vel_ratio[np.isfinite(vel_ratio)]))
            corr_vel = _pearson(rise_vel, decay_vel)

            out[(r, c)] = {
                "n_common_peaks": int(len(common)),
                "corr_disp": corr_disp,
                "corr_angle": corr_angle,
                "mean_abs_angle_diff_deg": mean_abs_angle_diff,
                "mean_disp_ratio_decay_over_rise": mean_disp_ratio,
                "mean_vel_ratio_decay_over_rise": mean_vel_ratio,
                "corr_vel": corr_vel,
            }

    # global aggregation: simple mean over cells
    if out:
        out["global"] = {
            "cells": [k for k in out.keys() if k != "global"],
            "mean_corr_disp": float(np.nanmean([v["corr_disp"] for k, v in out.items() if k != "global"])),
            "mean_corr_angle": float(np.nanmean([v["corr_angle"] for k, v in out.items() if k != "global"])),
            "mean_corr_vel": float(np.nanmean([v["corr_vel"] for k, v in out.items() if k != "global"])),
            "mean_abs_angle_diff_deg": float(np.nanmean([v["mean_abs_angle_diff_deg"] for k, v in out.items() if k != "global"])),
            "mean_disp_ratio_decay_over_rise": float(np.nanmean([v["mean_disp_ratio_decay_over_rise"] for k, v in out.items() if k != "global"])),
            "mean_vel_ratio_decay_over_rise": float(np.nanmean([v["mean_vel_ratio_decay_over_rise"] for k, v in out.items() if k != "global"])),
        }
    return out


def plot_left_right_deltas(deltas, title_prefix=""):
    """
    Simple bar plots of delta_R and delta_disp per row for rise/decay.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), constrained_layout=True)
    phases = ["rise", "decay"]

    for i, phase in enumerate(phases):
        rows = deltas[phase]["rows"]
        dR = deltas[phase]["delta_R_by_row"]
        dD = deltas[phase]["delta_disp_by_row"]

        ax = axes[i, 0]
        ax.bar(rows, dR)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"{title_prefix} ΔR (L-R) {phase}".strip())
        ax.set_xlabel("row")
        ax.set_ylabel("ΔR")

        ax = axes[i, 1]
        ax.bar(rows, dD)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"{title_prefix} Δdisp (L-R) {phase}".strip())
        ax.set_xlabel("row")
        ax.set_ylabel("Δdisp (mm)")

        # 세 번째 열 추가
        ax = axes[i, 2]
        dV = deltas[phase]["delta_vel_by_row"]
        ax.bar(rows, dV)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"{title_prefix} Δvel (L-R) {phase}".strip())
        ax.set_xlabel("row")
        ax.set_ylabel("Δvel (mm/s)")

    if _pdf is not None:
        _pdf.savefig(plt.gcf())
    plt.show()


def print_coupling_table(coupling, num_rows=3, num_cols=2):
    print("row col | n_common | corr_disp | corr_vel | corr_angle | mean|Δangle| | disp_ratio | vel_ratio")
    for r in range(num_rows):
        for c in range(num_cols):
            key = (r, c)
            if key not in coupling:
                continue
            v = coupling[key]
            print(f"{r:3d} {c:3d} |"
                  f" {v['n_common_peaks']:8d} |"
                  f" {v['corr_disp']:9.3f} |"
                  f" {v['corr_angle']:10.3f} |"
                  f" {v['mean_abs_angle_diff_deg']:10.1f} |"
                  f" {v['mean_disp_ratio_decay_over_rise']:14.3f}"
                  f" {v['corr_vel']:9.3f} |"
                  f" {v['mean_vel_ratio_decay_over_rise']:9.3f}")
    if "global" in coupling:
        g = coupling["global"]
        print("\nGLOBAL (mean over cells)")
        print(f"mean_corr_disp: {g['mean_corr_disp']:.3f}")
        print(f"mean_corr_angle: {g['mean_corr_angle']:.3f}")
        print(f"mean_abs_angle_diff_deg: {g['mean_abs_angle_diff_deg']:.1f}")
        print(f"mean_disp_ratio_decay_over_rise: {g['mean_disp_ratio_decay_over_rise']:.3f}")
        print(f"mean_disp_ratio_decay_over_rise: {g['mean_disp_ratio_decay_over_rise']:.3f}")
        print(f"mean_vel_ratio_decay_over_rise: {g['mean_vel_ratio_decay_over_rise']:.3f}")


def run_one_case_lr_and_coupling(direction, num_rows=3, num_cols=2, title_prefix="case"):
    """
    Input: direction (your list of dicts)
    Output: dict with summary, deltas, coupling
    Also plots deltas and prints coupling table.
    """
    summary = summarize_region_phase_coup(direction, num_rows=num_rows, num_cols=num_cols)
    deltas = compute_left_right_deltas(summary, num_rows=num_rows, num_cols=num_cols)
    coupling = compute_rise_decay_coupling(direction, num_rows=num_rows, num_cols=num_cols)

    plot_left_right_deltas(deltas, title_prefix=title_prefix)
    print_coupling_table(coupling, num_rows=num_rows, num_cols=num_cols)

    return {
        "summary": summary,
        "deltas": deltas,
        "coupling": coupling,
    }


def histogram(direction):
    abs_diffs = []
    for peak_id in set(d['peak_id'] for d in direction):
        for row in range(3):
            for col in range(2):
                rise = [d for d in direction if d['peak_id']==peak_id and d['phase']=='rise' and d['row']==row and d['col']==col]
                decay = [d for d in direction if d['peak_id']==peak_id and d['phase']=='decay' and d['row']==row and d['col']==col]
                if rise and decay:
                    diff = (rise[0]['mean_angle_deg'] - decay[0]['mean_angle_deg'] + 180) % 360 - 180
                    abs_diffs.append(abs(diff))
    
    plt.hist(abs_diffs, bins=18, range=(0, 180))
    plt.xlabel("Rise-Decay angle difference (degrees)")
    plt.ylabel("Count")
    plt.title("Distribution of rise-decay direction differences")
    if _pdf is not None:
        _pdf.savefig(plt.gcf())
    plt.show()



def parse_case_name(case_name):
    # "V_1aav_05_2" → verapamil, conc=0.5, age=None
    # "N_1aav_age_11_1" → nifedipine, conc=None, age=11
    # "V_1aav_B_1" → verapamil, conc=0 (baseline), age=None
    
    drug = 'verapamil' if case_name.startswith('V') else 'nifidepine'
    
    if '_age_' in case_name:
        age = int(case_name.split('_age_')[1].split('_')[0])
        conc = None
    elif '_B_' in case_name:
        age = None
        conc = 0  # baseline
    else:
        age = None
        # V_1aav_05_2 → "05"
        parts = case_name.split('_')
        conc_str = parts[2]  # "002", "01", "05", "1", "10"
        conc_map = {'002': 0.02, '01': 0.1, '05': 0.5, '1': 1.0, '10': 10.0}
        conc = conc_map.get(conc_str, conc_str)
    
    return drug, conc, age






########################################################################################

def motion_debug_stats(trajs, t_start, t_end, pixel_to_mm=1.0, step_thr_px=0.0):
    """
    trajs: list of trajectories, each is (T,2) or list of (x,y)
    t_start, t_end: inclusive indices within this peak window (0..peak_len-1)
    Returns dict with distributions + summary.
    """
    nets = []
    cumuls = []
    step_means = []
    step_maxs = []

    for traj in trajs:
        traj = np.asarray(traj, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] != 2:
            continue
        if t_end >= len(traj):
            continue
        if t_end <= t_start:
            continue

        sub = traj[t_start:t_end+1]                 # (L,2)
        d = sub[1:] - sub[:-1]                      # (L-1,2)
        steps = np.sqrt((d[:,0]**2 + d[:,1]**2))    # (L-1,)

        if step_thr_px > 0:
            steps = steps[steps > step_thr_px]

        net = sub[-1] - sub[0]
        net_mag = float(np.sqrt(net[0]**2 + net[1]**2))
        cumul = float(np.sum(steps)) if steps.size else 0.0
        step_mean = float(np.mean(steps)) if steps.size else 0.0
        step_max = float(np.max(steps)) if steps.size else 0.0

        nets.append(net_mag)
        cumuls.append(cumul)
        step_means.append(step_mean)
        step_maxs.append(step_max)

    nets = np.asarray(nets)
    cumuls = np.asarray(cumuls)
    step_means = np.asarray(step_means)
    step_maxs = np.asarray(step_maxs)

    def _summ(a):
        if a.size == 0:
            return {"n": 0}
        return {
            "n": int(a.size),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "std": float(np.std(a)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
        }

    out = {
        "net_px": _summ(nets),
        "cumul_px": _summ(cumuls),
        "step_mean_px": _summ(step_means),
        "step_max_px": _summ(step_maxs),
        "ratio_cumul_over_net_mean": float(np.mean(cumuls / (nets + 1e-9))) if nets.size else np.nan,
        "net_mm_mean": float(np.mean(nets) * pixel_to_mm) if nets.size else np.nan,
        "cumul_mm_mean": float(np.mean(cumuls) * pixel_to_mm) if cumuls.size else np.nan,
        "step_mean_mm_mean": float(np.mean(step_means) * pixel_to_mm) if step_means.size else np.nan,
    }
    return out




def debug_net_cumul_step_for_all_peaks(trajectories_per_peak, peak_info_results, fps=100, step_thr_px=0.1):
    pixel_size = 0.0069
    binning_factor = 2
    magnification = 2
    pixel_to_mm = (pixel_size * binning_factor) / magnification

    for peak_idx, p in enumerate(peak_info_results):
        rise = int(p["rise_frame"])
        peak = int(p["peak_frame"])
        decay = int(p["decay_frame"])

        trajs = trajectories_per_peak[peak_idx]  # list of trajectories for that peak
        peak_len = len(trajs[0]) if len(trajs) else 0

        # within-peak indices
        rise_start = 0
        rise_end = max(0, (peak - rise))               # inclusive end for rise
        decay_start = max(0, (peak - rise))            # start at peak frame within window
        decay_end = max(0, (decay - rise))             # inclusive end for decay

        # clamp
        rise_end = min(rise_end, peak_len - 1)
        decay_start = min(decay_start, peak_len - 1)
        decay_end = min(decay_end, peak_len - 1)
        rise_n_steps = max(rise_end - rise_start, 0)
        decay_n_steps = max(decay_end - decay_start, 0)

        rise_stats = motion_debug_stats(trajs, rise_start, rise_end, pixel_to_mm=pixel_to_mm, step_thr_px=step_thr_px)
        decay_stats = motion_debug_stats(trajs, decay_start, decay_end, pixel_to_mm=pixel_to_mm, step_thr_px=step_thr_px)

        print("=================================")
        print(f"Peak {p['peak_index']}  frames: rise={rise}, peak={peak}, decay={decay}  (peak_len={peak_len})")
        print(f"Peak {p['peak_index']}  rise_frames={rise_n_steps}, decay_frames={decay_n_steps} (peak_len={peak_len})")
        print(f"  RISE  net_px mean={rise_stats['net_px'].get('mean',np.nan):.3f}  "
              f"cumul_px mean={rise_stats['cumul_px'].get('mean',np.nan):.3f}  "
              f"step_mean_px mean={rise_stats['step_mean_px'].get('mean',np.nan):.3f}  "
              f"cumul/net ratio mean={rise_stats['ratio_cumul_over_net_mean']:.2f}")
        print(f"  DECAY net_px mean={decay_stats['net_px'].get('mean',np.nan):.3f}  "
              f"cumul_px mean={decay_stats['cumul_px'].get('mean',np.nan):.3f}  "
              f"step_mean_px mean={decay_stats['step_mean_px'].get('mean',np.nan):.3f}  "
              f"cumul/net ratio mean={decay_stats['ratio_cumul_over_net_mean']:.2f}")
        print(f"  RISE mm mean:  net={rise_stats['net_mm_mean']:.5f}  cumul={rise_stats['cumul_mm_mean']:.5f}")
        print(f"  DECAY mm mean: net={decay_stats['net_mm_mean']:.5f}  cumul={decay_stats['cumul_mm_mean']:.5f}")
        print("  RISE mean start x:", np.mean([traj[rise_start][0] for traj in trajs]))
        print("  DECAY mean start x:", np.mean([traj[decay_start][0] for traj in trajs]))



######################################################################################################################
################ABALATION###########################################
def aggregate_motion_with_config(trajectories_per_peak, peaks_info,
                                  tissue_mask_in, num_rows, num_cols,
                                  phase_split=True, video_shape=None):
    """
    Re-aggregate per-peak LK trajectories with given grid + phase config.
    
    For ablation: vary (num_rows, num_cols, phase_split) to test each component.
    
    Returns: list of per-(peak, phase, cell) dicts (same format as 
             generate_rise_decay_flow_images_for_peaks output)
    """
    global tissue_mask, GRID_CELLS, GRID_BBOXES, GRID_LINES
    
    # Set tissue mask + rebuild grid for this config
    tissue_mask = tissue_mask_in
    GRID_CELLS, GRID_BBOXES, GRID_LINES = _build_skewed_grid_from_mask(
        tissue_mask, num_rows=num_rows, num_cols=num_cols)
    
    # Dummy background image (arrows drawn on it but discarded)
    if video_shape is None:
        H, W = tissue_mask.shape
    else:
        H, W = video_shape[1], video_shape[2]
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    
    direction_results = []
    
    for peak_idx, info in enumerate(peaks_info):
        rise = info['rise_frame']
        peak = info['peak_frame']
        decay = info['decay_frame']
        peak_id = info['peak_index']
        trajs = trajectories_per_peak[peak_idx]
        
        if phase_split:
            phases = [
                {"name": "rise",  "start": 0,           "end": peak - rise},
                {"name": "decay", "start": peak - rise, "end": decay - rise},
            ]
        else:
            phases = [
                {"name": "full",  "start": 0,           "end": decay - rise},
            ]
        
        for ph in phases:
            draw_average_arrows_on_grid(
                bg, trajs,
                t_start=ph["start"], t_end=ph["end"],
                num_rows=num_rows, num_cols=num_cols,
                arrow_color=(0, 255, 0),
                peak_id=peak_id, phase_name=ph["name"],
                result_list=direction_results
            )
    
    return direction_results

    

    
    