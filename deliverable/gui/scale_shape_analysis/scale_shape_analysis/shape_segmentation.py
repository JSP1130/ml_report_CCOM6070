from pathlib import Path
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
from skimage.measure import find_contours
from skimage.segmentation import find_boundaries

# @title Scale mask segmentation
from rembg import remove
from skimage.morphology import binary_erosion, disk
from skimage.measure import label, regionprops
from skimage.morphology import reconstruction, remove_small_holes
from skimage.segmentation import find_boundaries

def extract_scale_mask(img, alpha_threshold=128, alpha_output=False):
  # Apply rembg to remove the background
  output_img_rgba = remove(img)

  # Extract the alpha channel which serves as the mask
  alpha_channel = output_img_rgba[:, :, 3]

  # Create a binary mask (0 or 255) from the alpha channel
  binary_mask = (alpha_channel >= alpha_threshold).astype(np.bool)

  # Label connected components in the binary mask
  labeled_mask = label(binary_mask)

  # Get region properties
  regions = regionprops(labeled_mask)

  # Find the largest region by area
  largest_region = None
  if regions:
      largest_region = max(regions, key=lambda r: r.area)

  # Create a new mask containing only the largest region directly from its label
  largest_region_mask = np.zeros_like(binary_mask, dtype=np.uint8)
  if largest_region is not None:
      largest_region_mask = (labeled_mask == largest_region.label)
      largest_region_mask = remove_small_holes( largest_region_mask, max_size = 10000 )


  if (alpha_output):
    # Do morphological reconstruction to get alpha mask only for the largest region
    alpha_uint8 = img_as_ubyte(alpha_channel)
    mask_uint8 = img_as_ubyte(largest_region_mask)
    seed_uint8 = np.minimum( alpha_uint8, mask_uint8, dtype=np.uint8 )  # Seed cannot be larger than reconstruction mask (alpha)
    largest_region_alpha = reconstruction(seed_uint8, alpha_uint8, method='dilation').astype(np.uint8)

    return largest_region_mask, largest_region_alpha
  else:
    return largest_region_mask

def overlay_contour(img, mask, color=(255,0,0)):

  # Find the boundaries of the mask
  contour_pixels = find_boundaries(mask.astype(np.bool), mode='outer')

  # Create a copy of the image to draw the contour on
  img_with_contour = img.copy()

  # Set the contour pixels to red
  # img_with_contour expects RGB, so [R, G, B] = [255, 0, 0] for red
  img_with_contour[contour_pixels] = color

  return img_with_contour



  # @title Contour extraction functions
from skimage import measure

def extract_contour(mask):
    contours = measure.find_contours(mask, level=0.5)
    if not contours:
        raise ValueError("No contour found")

    contour = max(contours, key=len)

    # convert (row,col) → (x,y)
    contour = contour[:, ::-1]

    # ensure closed
    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    return contour

from scipy.interpolate import interp1d

def resample_contour(contour, n_points=300):
    deltas = np.diff(contour, axis=0)
    dists = np.sqrt((deltas**2).sum(axis=1))
    cumulative = np.concatenate([[0], np.cumsum(dists)])

    total_length = cumulative[-1]
    cumulative /= total_length

    fx = interp1d(cumulative, contour[:,0])
    fy = interp1d(cumulative, contour[:,1])

    uniform = np.linspace(0, 1, n_points)
    resampled = np.column_stack([fx(uniform), fy(uniform)])

    return resampled

def ensure_clockwise(contour):
    x = contour[:,0]
    y = contour[:,1]
    area = np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])
    if area > 0:
        contour = contour[::-1]
    return contour

from scipy.interpolate import splprep

def fit_periodic_spline(contour, smooth=1.0):
    x = contour[:,0]
    y = contour[:,1]

    tck, u = splprep([x, y], s=smooth, per=True)
    return tck

def full_outline_pipeline(mask,
                          n_points=300,
                          smooth=1.0):
    contour = extract_contour(mask)
    contour = resample_contour(contour, n_points)
    contour = ensure_clockwise(contour)
    spline = fit_periodic_spline(contour, smooth)

    return contour, spline


# @title Contour curvature normalization
from scipy.interpolate import splprep, splev

def spline_curvature(contour, smooth=0):
    tck, u = splprep([contour[:,0], contour[:,1]], s=smooth, per=True)

    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)

    numerator = np.array(dx)*np.array(ddy) - np.array(dy)*np.array(ddx)
    denominator = (np.array(dx)**2 + np.array(dy)**2)**1.5
    eps = 1e-12

    curvature = numerator / (denominator+eps)
    return curvature

import numpy as np

import numpy as np

def rotate_contour_by_max_curvature(contour, curvature, kind='absmax'):
    """
    Rotate a closed contour so that the point with the highest |curvature| becomes the first point.
    Curvature is computed internally using finite differences.

    Parameters
    ----------
    contour : (N,2) array
        x,y coordinates of the contour (may be closed)

    Returns
    -------
    rotated_contour : (N,2) array
        rotated, closed contour
    curvature : (N,) array
        curvature along rotated contour
    peak_idx : int
        index of max |curvature| in original contour
    """

    # Remove duplicate endpoint if contour is closed
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]

    # Find max absolute curvature
    if (kind=='min'):
      peak_idx = np.argmin(curvature)
    elif (kind=='max'):
      peak_idx = np.argmax(curvature)
    elif (kind=='absmax'):
      peak_idx = np.argmax(np.abs(curvature))
    else:
      raise 'invalid kind'

    # Circularly rotate contour and curvature
    rotated_contour = np.roll(contour, -peak_idx, axis=0)
    rotated_curvature = np.roll(curvature, -peak_idx)

    # Re-close contour
    rotated_contour = np.vstack([rotated_contour, rotated_contour[0]])

    return rotated_contour, rotated_curvature, peak_idx

# @title Code for contour split into subarcs
import numpy as np

def main_axes_from_anchor(contour, anchor):
    """
    Compute main axes of a shape using a custom anchor point.

    Parameters
    ----------
    contour : (N,2) array
        x,y coordinates of the shape (closed or open)
    anchor : (2,) array-like
        The anchor/origin point (x0, y0)

    Returns
    -------
    eigvals : array, shape (2,)
        Eigenvalues of covariance matrix (variance along each axis)
    eigvecs : array, shape (2,2)
        Eigenvectors (columns) corresponding to main axes
    """
    # Shift coordinates
    X = contour[:,0] - anchor[0]
    Y = contour[:,1] - anchor[1]

    coords = np.column_stack([X, Y])

    # Covariance
    C = np.cov(coords.T)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending order
    # largest eigenvalue first
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    displacement = np.mean(coords, axis=0)
    if np.dot(displacement, eigvecs[:,0]) < 0:
        eigvecs[:,0] *= -1  # flip direction
    # Second eigenvector orthogonal
    eigvecs[:,1] = np.array([-eigvecs[1,0], eigvecs[0,0]])

    return eigvals, eigvecs

def rotate_coords_main_axes(contour, anchor, eigvecs):
  # Shift coordinates
  XY = contour - anchor.reshape(1,2)

  RXY = XY @ eigvecs

  return RXY

import numpy as np

def extreme_points_perpendicular(contour, anchor, eigvecs):
    """
    Find the two points furthest from the main axis (first eigenvector) on each side.

    Parameters
    ----------
    contour : (N,2) array
    anchor : (2,) array
    eigvecs : (2,2) array
        Columns = eigenvectors, first = main axis

    Returns
    -------
    pt_pos_side : (2,) array
        Point on positive side of axis furthest away
    pt_neg_side : (2,) array
        Point on negative side of axis furthest away
    distances : (N,) array
        perpendicular distances of all points
    """
    v0 = eigvecs[:,0]  # main axis
    displacement = contour - anchor  # (N,2)

    # Signed perpendicular distance in 2D
    # Using cross product to get sign: v0 x disp
    # cross2D(a,b) = a_x*b_y - a_y*b_x
    signed_dist = v0[0]*displacement[:,1] - v0[1]*displacement[:,0]

    # Perpendicular distance magnitude
    #distances = np.abs(signed_dist)

    # Split by side
    #pos_mask = signed_dist > 0
    #neg_mask = signed_dist < 0

    id_right = np.argmax(signed_dist)
    id_left  = np.argmin(signed_dist)

    #pt_pos_side = contour[pos_mask][np.argmax(distances[pos_mask])]
    #pt_neg_side = contour[neg_mask][np.argmax(distances[neg_mask])]

    return id_right, id_left

import numpy as np

def low_curvature_intervals_include_points(curvature, threshold, indices_to_include):
    """
    Find contiguous intervals where curvature < threshold that include given indices.
    Works for a closed contour.

    Parameters
    ----------
    curvature : (N,) array
        curvature vector
    threshold : float
        threshold for "low curvature"
    indices_to_include : list or array of int
        indices that must be included in the intervals

    Returns
    -------
    intervals : list of tuples
        Each tuple = (start_idx, end_idx) of low-curvature interval (inclusive start, exclusive end)
    """
    N = len(curvature)

    # 1️⃣ Boolean mask of low curvature
    low_mask = curvature < threshold

    # 2️⃣ Duplicate mask to handle circular wrap-around
    doubled = np.concatenate([low_mask, low_mask])

    intervals = []
    start = None
    for i, val in enumerate(doubled):
        if val and start is None:
            start = i
        elif not val and start is not None:
            intervals.append((start, i))
            start = None
    if start is not None:
        intervals.append((start, len(doubled)))

    # 3️⃣ Map intervals back to original index space
    valid_intervals = []
    for s, e in intervals:
        length = e - s
        if length > 0:
            s_mod = s % N
            valid_intervals.append((s_mod, length))

    # 4️⃣ Keep only intervals that contain the specified indices
    result = []
    for idx in indices_to_include:
        # find interval containing idx
        found = False
        for s, length in valid_intervals:
            interval_indices = np.arange(s, s+length) % N
            if idx in interval_indices:
                result.append((s, (s+length)%N))
                found = True
                break
        if not found:
            result.append(None)  # fallback if no interval includes this index

    return result

def local_extreme_x_intervals(up_contour, seeds):
    N = len(up_contour)

    y = np.concatenate([up_contour[:,0], up_contour[:,0]])  # double for easy boundary handling

    intervals = []
    for seed in seeds:
        # ---- Forward search ----
        idx_forward = seed
        for i in range(seed + 1, N - 1):
            if not (y[i-1] <= y[i]) ^ (y[i] >= y[i+1]): # any weak extremum
                idx_forward = i
                break

        # ---- Backward search ----
        idx_backward = seed
        for i in range(seed - 1, 0, -1):
            if not (y[i-1] <= y[i]) ^ (y[i] >= y[i+1]): # any weak extremum
                idx_backward = i
                break

        intervals.append( (idx_backward, idx_forward) )

    return intervals

def plot_serration_cut(up_contour, id_right_max, id_left_max, id_right_expanded, id_left_expanded):
  p1 = up_contour[id_right_max,:]
  p2 = up_contour[id_left_max,:]
  p1e = up_contour[id_right_expanded,:]
  p2e = up_contour[id_left_expanded,:]

  plt.plot( [p1[0],p2[0]], [p1[1],p2[1]], 'b*-' )
  plt.plot( [p1e[0],p2e[0]], [p1e[1],p2e[1]], 'k*-' )

def expand_serration(up_contour, id_right_max, id_left_max):
  p1 = up_contour[id_right_max,:]
  p2 = up_contour[id_left_max,:]

  if (abs(p2[1] - p1[1]) <= 2):
      return id_right_max, id_left_max

  invslope = (p2[0] - p1[0]) / (p2[1] - p1[1])

  deltas = up_contour[:,0] - invslope * (up_contour[:,1] - p1[1])

  dd = deltas[id_right_max:id_left_max+1]
  if (dd.size <= 2):
      return id_right_max, id_left_max
  
  mindelta = np.min( dd )

  N = len(up_contour)

  # ---- Forward search from left ----
  seed = id_left_max
  idx_forward = seed
  for i in range(seed + 1, N - 1):
      if deltas[i] < mindelta: # further back than threshold
          idx_forward = i
          break

  # ---- Backward search ----
  seed = id_right_max
  idx_backward = seed
  for i in range(seed - 1, 0, -1):
      if deltas[i] < mindelta: # further back than threshold
          idx_backward = i
          break

  return idx_backward, idx_forward  # expand_right, expand_left

import numpy as np

def trim_low_curvature_intervals(curvature, intervals, negative_threshold):
    """
    Trim multiple low-curvature intervals from both ends until curvature
    drops below negative_threshold.

    Parameters
    ----------
    curvature : (N,) array
        curvature vector
    intervals : list of tuples
        Each tuple = (start_idx, end_idx) of a low-curvature interval
        (indices modulo N for closed contour)
    negative_threshold : float
        curvature threshold to stop trimming

    Returns
    -------
    trimmed_intervals : list of tuples
        Each tuple = (new_start_idx, new_end_idx) after trimming
    """
    N = len(curvature)
    trimmed_intervals = []

    for start, end in intervals:
        # create array of indices for the interval
        indices = np.arange(start, end) % N
        curv = curvature[indices]

        # trim from start
        for i, val in enumerate(curv):
            if val > negative_threshold:
                break
        new_start = indices[i]

        # trim from end
        for j, val in enumerate(curv[::-1]):
            if val > negative_threshold:
                break
        new_end = indices[-(j+1)] + 1  # exclusive

        # modulo N for closed contour
        new_start %= N
        new_end %= N

        trimmed_intervals.append((new_start, new_end))

    return trimmed_intervals

import numpy as np

def generate_labels_closed_contour(N, anchor_idx, right_interval, left_interval):
    """
    Generate a label vector for a closed contour with implicit top interval.

    Labels:
        0 = anchor/start
        1 = right interval
        2 = top interval (between right and left)
        3 = left interval

    Parameters
    ----------
    N : int
        Number of points in the contour
    anchor_idx : int
        Index of the starting point
    left_interval : tuple
        (start_idx, end_idx) of left interval
    right_interval : tuple
        (start_idx, end_idx) of right interval

    Returns
    -------
    labels : (N,) array of int
    """
    labels = np.zeros(N, dtype=int)  # default = 0 (anchor)

    def mark_interval(interval, label_id):
        s, e = interval
        idxs = np.arange(s, e+1) % N
        labels[idxs] = label_id

    # assign right and left intervals
    mark_interval(right_interval, 1)
    mark_interval(left_interval, 3)

    top_interval = (right_interval[1]+1,left_interval[0]-1)
    mark_interval(top_interval, 2)

    return labels

from scipy.signal import hilbert

def reconstruct_envelope(signal):
    """
    Given a signal that is a single quadrature component, e.g., cos(phi(t)) * A(t),
    reconstruct the amplitude envelope using the analytic signal.

    Parameters
    ----------
    signal : (N,) array
        Observed cos component of a (cos,sin) quadrature signal

    Returns
    -------
    envelope : (N,) array
        Smoothed amplitude envelope
    """
    # Compute analytic signal
    analytic = hilbert(signal)

    # Amplitude envelope
    envelope = np.abs(analytic)

    return envelope

def split_contour(contour, curvature_threshold=0.02, smooth=10.0):
  anchor = contour[0]
  eigvals, eigvecs = main_axes_from_anchor(contour, anchor)

  id_right, id_left = extreme_points_perpendicular(contour, anchor, eigvecs)

  curvature = spline_curvature(contour, smooth=smooth)
  envelope = reconstruct_envelope(curvature)

  intervals = low_curvature_intervals_include_points(envelope, threshold=curvature_threshold, indices_to_include=[id_right, id_left])

  trimmed_intervals = intervals
  #trimmed_intervals = trim_low_curvature_intervals(curvature, intervals, negative_threshold=-0.005)

  labels = generate_labels_closed_contour(len(contour), 0, trimmed_intervals[0], trimmed_intervals[1])

  return trimmed_intervals, labels

def plot_split_contours(contour, labels):
  plt.scatter(contour[:,0], contour[:,1], c=labels)
  plt.plot(contour[0,0], contour[0,1], 'r*')  # stem anchor


def smooth_cyclic(curvature, window_size=5):
    """
    Smooth the curvature using a cyclic (circular) convolution.

    Parameters
    ----------
    curvature : (N,) array
        curvature vector
    window_size : int
        Size of smoothing kernel (odd preferred)

    Returns
    -------
    smoothed : (N,) array
        Smoothed |curvature| vector
    """
    #abs_curv = np.abs(curvature)
    N = len(curvature)

    # simple uniform kernel
    kernel = np.ones(window_size) / window_size

    # pad for circular convolution
    pad = window_size // 2
    extended = np.concatenate([curvature[-pad:], curvature, curvature[:pad]])

    # convolve
    smoothed = np.convolve(extended, kernel, mode='valid')

    return smoothed

def plot_up_xy(up_contour, curvature, ids=None, cols=None):

    fig, axes = plt.subplots(
        3, 1,
        sharex=True,
        figsize=(8, 6)
    )

    u = range(len(up_contour))
    if (ids is not None and cols is None):
      nids = len(ids)
      cols = ['r'] * nids

    # x(u)
    axes[0].plot(u, up_contour[:,0])
    axes[0].set_ylabel("x(u)")
    axes[0].grid(True)
    if (ids is not None):
        for id,col in zip(ids,cols):
          axes[0].plot(id, up_contour[id,0], '*', color=col)

    # y(u)
    axes[1].plot(u, up_contour[:,1])
    axes[1].set_ylabel("y(u)")
    axes[1].grid(True)
    if (ids is not None):
        for id,col in zip(ids,cols):
          axes[1].plot(id, up_contour[id,1], '*', color=col)

    # curvature(u)
    axes[2].plot(u, curvature)
    axes[2].set_ylabel("curvature")
    axes[2].set_xlabel("u (arc-length parameter)")
    axes[2].grid(True)
    if (ids is not None):
        for id,col in zip(ids,cols):
          axes[2].plot(id, curvature[id], '*', color=col)

    plt.tight_layout()

def plot_contour_xy(up_contour, ids=None, cols=None):

    axes = plt.gca()

    u = range(len(up_contour))
    if (ids is not None and cols is None):
      nids = len(ids)
      cols = ['r'] * nids

    # x(u)
    axes.plot(up_contour[:,0], up_contour[:,1],'.-')
    axes.set_xlabel("x(u)")
    axes.set_ylabel("y(u)")
    axes.grid(True)
    if (ids is not None):
        for id,col in zip(ids,cols):
          axes.plot(up_contour[id,0], up_contour[id,1], '*', color=col)

    plt.tight_layout()


from skimage.draw import circle_perimeter, line, polygon

import numpy as np
import matplotlib.colors as mcolors

def to_rgb255(color):
    """
    Convert various color formats to RGB [0,255].

    Supports:
        - short names: 'r'
        - full names: 'red'
        - hex: '#ff0000'
        - 0-1 floats: (1,0,0)
        - 0-255 ints: (255,0,0)
        - numpy arrays / lists

    Returns
    -------
    np.ndarray shape (3,) dtype uint8
    """
    # If already array-like
    if isinstance(color, (list, tuple, np.ndarray)):
        arr = np.array(color)
        return arr.astype(np.uint8)

    # Otherwise assume string-like → use matplotlib
    rgb01 = mcolors.to_rgb(color)  # returns floats in [0,1]
    return (np.array(rgb01) * 255).astype(np.uint8)

def draw_keypoints( img, contour, ids=None, cols=None ):
    u = range(len(contour))

    if (ids is not None and cols is None):
      nids = len(ids)
      cols = ['r'] * nids

    if (ids is not None):
      for id,col in zip(ids,cols):
          X,Y = np.round(contour[id,0]).astype(int), np.round(contour[id,1]).astype(int)
          rr, cc = circle_perimeter(Y,X, 5, shape=img.shape)
          img[rr, cc] = to_rgb255(col)
          rr, cc = circle_perimeter(Y,X, 6, shape=img.shape)
          img[rr, cc] = to_rgb255(col)

def draw_axis( img, anchor, eigvecs, sz=200, col='b'):
  U = eigvecs[:,0]
  V = eigvecs[:,1]

  col = to_rgb255(col)

  c0,c1,r0,r1 = (anchor[0], anchor[0]+sz*U[0], anchor[1], anchor[1]+sz*U[1])
  #c0,c1,r0,r1 = anchor[0], anchor[0]+10, anchor[1], anchor[1]+10
  c0 = int(c0)
  c1 = int(c1)
  r0 = int(r0)
  r1 = int(r1)
  rr, cc = line( r0,c0,r1,c1 )
  mask = (
      (rr >= 0) & (rr < img.shape[0]) &
      (cc >= 0) & (cc < img.shape[1])
  )
  rr = rr[mask]
  cc = cc[mask]
  img[rr, cc] = col
  #print( rr, cc)

def draw_contour( img, contour, col = 'b'):

    col = to_rgb255(col)

    x,y = contour[:,0].astype(int),contour[:,1].astype(int)

    rr_all = []
    cc_all = []
    for i in range(len(contour) - 1):
        rr, cc = line(y[i], x[i], y[i+1], x[i+1])
        rr_all.append(rr)
        cc_all.append(cc)

    rr_all = np.concatenate(rr_all)
    cc_all = np.concatenate(cc_all)

    img[rr_all, cc_all] = col

import h5py
from types import SimpleNamespace


class ShapeExtractor:
    def __init__(self, n_points = 512, smooth=1.0, curvature_smooth=5.0):
        self.n_points = n_points
        self.smooth = smooth
        self.alpha_threshold = 200
        self.curvature_smooth = curvature_smooth
   
    def process(self, img):
   
        mask, alpha = extract_scale_mask(img, alpha_threshold=self.alpha_threshold, alpha_output = True)

        overlay = overlay_contour(img, mask, color=(0,0,255))

        # Extract advanced keypoints
        contour, spline = full_outline_pipeline(mask,
                                n_points=self.n_points,
                                smooth=self.smooth)

        curvature = spline_curvature( contour, smooth=self.curvature_smooth)
        contour, curvature, _ = rotate_contour_by_max_curvature(contour, curvature, kind='min')
        u = range(len(contour))

        anchor = contour[0]
        eigvals, eigvecs = main_axes_from_anchor(contour, anchor)

        up_contour = rotate_coords_main_axes(contour, anchor, eigvecs)

        id_right, id_left = extreme_points_perpendicular(contour, anchor, eigvecs)
        id_sides = [id_right, id_left]
        intervals = local_extreme_x_intervals(up_contour, seeds=[id_right, id_left])

        id_right_expanded, id_left_expanded = expand_serration(up_contour, intervals[0][1], intervals[1][0])
        intervals_expanded = [ [ intervals[0][0], id_right_expanded ], [id_left_expanded, intervals[1][1]] ]

        labels = generate_labels_closed_contour(len(contour), 0, intervals_expanded[0], intervals_expanded[1])

        results = {}
        fields = ['mask', 'alpha', 'contour', 'curvature', 'up_contour','anchor','eigvecs','eigvals','id_sides','intervals','intervals_expanded']
        for field in fields:
            results[field] = locals()[field]

        return results

    def draw(self, results, overlay):

        data = SimpleNamespace(**results)

        sz = np.max(data.up_contour[:,0]) # max x coordinate, gives the length from the anchor

        ids=[0, data.id_sides[0], data.id_sides[1], data.intervals[0][0], data.intervals[0][1], data.intervals[1][0], data.intervals[1][1], data.intervals_expanded[0][1], data.intervals_expanded[1][0]]
        cols=['k','r','g','m','k','k','c', 'm','c']

        draw_contour( overlay, data.contour, col='b' )

        draw_keypoints( overlay, data.contour, ids, cols )
        draw_axis( overlay, data.anchor, data.eigvecs, sz, col='k')

    def h5_save(self, h5path, results):
        fields = ['contour','curvature','up_contour', 'anchor', 'eigvecs', 'eigvals', 'id_sides', 'intervals', 'intervals_expanded']
        with h5py.File(h5path, "w") as f:
            
            for field in fields:
                f.create_dataset(field, data=results[field])

            # Optional metadata
            f.attrs["version"] = "1.0"
            f.attrs["description"] = "Butterfly scale contour analysis"
