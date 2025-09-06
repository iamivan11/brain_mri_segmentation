import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import jaccard_score, classification_report
from skimage.filters import threshold_multiotsu
from skimage.morphology import closing, area_closing
from skimage.segmentation import (
    morphological_chan_vese,
    morphological_geodesic_active_contour,
)

# BASIC

def invert_binary(img_bin):
    """Invert a binary-like image array (0 <-> max)."""
    result = (img_bin.max() - img_bin)
    return result.astype(np.float32)

def blackout_pixels(img, img_bin):
    """Set pixels to 0 where img_bin is above its midpoint threshold."""
    result = img.copy()
    threshold = 0.5 if img_bin.max() <= 1 else img_bin.max() / 2
    result[img_bin > threshold] = 0
    return result.astype(np.float32)

def raise_nonblack_pixels(img, increment):
    """Increase non-zero pixels by increment, clipping to original max."""
    result = img.copy()
    result[result > 0] += increment
    result = np.clip(result, 0, img.max())
    return result.astype(np.float32)

def combine_regions(T1_0, T1_1, T1_2, T1_3, T1_4, T1_5):
    """Combine six binary region masks into a single labeled mask 0...5."""
    T1_0[T1_0 == 1] = 0
    T1_1[T1_1 == 1] = 1
    T1_2[T1_2 == 1] = 2
    T1_3[T1_3 == 1] = 3
    T1_4[T1_4 == 1] = 4
    T1_5[T1_5 == 1] = 5
    result = T1_0 + T1_1 + T1_2 + T1_3 + T1_4 + T1_5
    return result


# SEGMENTATION

def multi_otsu_thresholding(img, classes):
    """Segment image into given classes using Multi-Otsu Thresholding."""
    thresholds = threshold_multiotsu(img, classes=classes)
    result = np.digitize(img, bins=thresholds)
    return result.astype(np.float32)

def chan_vese(img, n_iter=300, smoothing=1, iter_callback=lambda u: None):
    """Morphological Chan–Vese segmentation (MCV)."""
    result = morphological_chan_vese(img, num_iter=n_iter, init_level_set='disk',
                                   smoothing=smoothing, iter_callback=iter_callback)
    return result.astype(np.float32)

def geod_active_contour(img, n_iter=300, smoothing=1, balloon=1, threshold='auto', iter_callback=lambda u: None):
    """Morphological Geodesic Active Contour segmentation (MGAC)."""
    result = morphological_geodesic_active_contour(
        img, num_iter=n_iter, init_level_set='disk', smoothing=smoothing,
        balloon=balloon, threshold=threshold, iter_callback=iter_callback,
    )
    return result.astype(np.float32)

def area_close(img_bin, area_threshold):
    """Remove small holes in a binary volume by morphological area closing."""
    result = area_closing(img_bin > 0, area_threshold=area_threshold)
    return result.astype(np.float32)

def close(img):
    """Apply morphological closing to an image/volume."""
    result = closing(img)
    return result.astype(np.float32)

def kmeans_segmentation(img, k):
    pixel_values = img.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented = labels.reshape(img.shape)
    # Rank clusters by increasing center intensity.
    sorted_i = np.argsort(centers.ravel())
    rank_map = np.zeros_like(sorted_i)
    for rank, i in enumerate(sorted_i):
        rank_map[i] = rank
    # Remap segmentation labels to 0, 1, 2, ... etc.
    result = rank_map[segmented]
    return result.astype(np.float32)


# METRICS

def compute_jaccard(segmented, GT, labels=None, average='macro'):
    """Return per-region Jaccard and overall mean (sklearn)."""
    if labels is None:
        labels = [0, 1, 2, 3, 4, 5]
    pred = segmented.ravel()
    truth = GT.ravel()
    per_label = jaccard_score(truth, pred, labels=labels, average=None)
    result = {}
    for label, score in zip(labels, per_label):
        result[f'Region {label}'] = float(score)
    result['Overall'] = float(jaccard_score(truth, pred, labels=labels, average=average))
    return result

def compute_P_R_F1(segmented, GT):
    """Print precision (P) / recall (R) / F1-Score (F1) classification report (sklearn)."""
    result = classification_report(GT.ravel(), segmented.ravel())
    print(result)


# VISUALIZATION

def visualize_slice(img, s=0, cap='', cmap='jet'):
    """Show a single slice from a volume/image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(img[:,:,s], cmap=cmap)
    plt.title(f'{cap} Slice {s+1}')
    plt.axis('off')
    plt.show()

def visualize_row(img, cap='', cmap='jet'):
    """Show a single row of slices for the provided image/volume."""
    n_slices = img.shape[-1]
    _, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
    if n_slices == 1:
        axes = [axes]
    for i in range(n_slices):
        axes[i].imshow(img[..., i], cmap=cmap)
        axes[i].axis('off')
        axes[i].set_title(f'{cap} Slice {i+1}')
    plt.tight_layout()
    plt.show()

def visualize_rows(img1, img2, img3, cap_1='', cap_2='', cap_3='', cmap_1='jet', cmap_2='jet', cmap_3='jet'):
    """Show rows of slices for img1 (e.g. T1), img2 (e.g. GT), img3 (e.g. segmented)."""
    z = img2.shape[-1]
    _, axes = plt.subplots(3, z, figsize=(3 * z, 8))
    for i in range(z):
        axes[0, i].imshow(img1[..., i], cmap=cmap_1)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'{cap_1} Slice {i+1}')
        axes[1, i].imshow(img2[..., i], cmap=cmap_2)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'{cap_2} Slice {i+1}')
        axes[2, i].imshow(img3[..., i], cmap=cmap_3)
        axes[2, i].axis('off')
        axes[2, i].set_title(f'{cap_3} Slice {i+1}')
    plt.tight_layout()
    plt.show()