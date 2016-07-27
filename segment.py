# !~/env/bin/python

from __future__ import division
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import vmo
import librosa
import vmo.analysis.segmentation as vse
import numpy as np
import scipy
import scipy.stats as stats
import scipy.cluster.hierarchy as hierarchy
import mir_eval
from sklearn.cluster import KMeans
from sklearn.mixture import DPGMM
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
import json
import collections
import scipy.linalg as linalg
import pyhsmm
import pyhsmm.basic.distributions as distributions
import pyhsmm.util.text as textutil
import sklearn.preprocessing as pre


SAMPLE_RATE = 22050
HOP_SIZE = 512
SILENCE = 'se'
K_MIN = 5
K_MAX = 10
MEDIAN_WIDTH = 17
GAUSS_WIDTH = 17
Nmax = 20


def load_audio(audio_file, sr=SAMPLE_RATE, silence=SILENCE):
    y, sr = librosa.load(audio_file, sr=sr)
    y_len = len(y)  # number of samples

    sound = np.where(20 * np.log10(np.abs(y)) > -60)
    start = 0
    ending = -1
    if silence == 's':
        start = sound[0][0]
    elif silence == 'e':
        ending = sound[0][-1]
    elif silence == 'se':
        start = sound[0][0]
        ending = sound[0][-1]
    else:
        start = 0
        ending = -1

    if float(start) / sr < 0.1:
        start = 0
    if float(len(y) - ending) / sr < 0.3:
        ending = -1
        y_sound = y[start:]
    else:
        y_sound = y[start:ending]

    return y_len, y_sound, start, ending


def find_boundaries(frame_labels, width=9):
    frame_labels = np.pad(frame_labels, (int(width / 2), int(width / 2) + 1), mode='reflect')
    frame_labels = np.array([stats.mode(frame_labels[i:j])[0][0]
                             for (i, j) in zip(range(0, len(frame_labels) - width),
                                               range(width, len(frame_labels)))])
    boundaries = 1 + np.asarray(np.where(frame_labels[:-1] != frame_labels[1:])).reshape((-1,))
    boundaries = np.unique(np.concatenate([[0], boundaries, [len(frame_labels)-1]]))
    return boundaries, frame_labels


def segment_labeling(x, boundaries, c_method='kmeans', k=5):
    x_sync = librosa.feature.sync(x.T, boundaries)

    if c_method == 'kmeans':
        c = KMeans(n_clusters=k, n_init=100)
        seg_labels = c.fit_predict(x_sync.T)
    elif c_method == 'agglomerative':
        z = hierarchy.linkage(x_sync.T, method='ward')
        t = k * np.max(z[:, 2])
        seg_labels = hierarchy.fcluster(z, t=t, criterion='distance')
    else:
        c = KMeans(n_clusters=k, n_init=100)
        seg_labels = c.fit_predict(x_sync.T)

    return seg_labels


def structure_feature_segmenter(feature=None, sf=None, boundaries=None, width=GAUSS_WIDTH, method='rec', delta=0.05, m='rsfx'):

    if sf is None:
        ssm = recurrence_plot_extraction(feature, method=method, m=m)

        lag_sim = librosa.segment.recurrence_to_lag(ssm, pad=False)
        sf = gaussian_filter(lag_sim, [0.5, width], order=0, mode='reflect')
    else:
        ssm = recurrence_plot_extraction(sf, method=method, m=m)
        fake_lag_sim = sf.T[:K_MAX]
        sf = gaussian_filter1d(fake_lag_sim, width)

    ssm = diag_median_filter(ssm, width=9)
    ssm = np.maximum(ssm, ssm.T)

    if boundaries is None:
        novelty_curve = np.sqrt(np.mean(np.diff(sf, axis=1) ** 2, axis=0))
        novelty_curve = pre.minmax_scale(novelty_curve)
        novelty_curve = np.insert(novelty_curve, 0, 0)

        bound_width = 9
        offset = int((bound_width - 1) / 2)
        tmp_novelty = np.pad(novelty_curve, [offset], mode='reflect')
        boundaries = np.array([0])
        for i in range(len(novelty_curve)):
            if (np.greater(tmp_novelty[i + offset], tmp_novelty[i:i + offset]).all() and
                    np.greater(tmp_novelty[i + offset], tmp_novelty[i + offset + 1:i + bound_width]).all() and
                    tmp_novelty[i + offset] > delta):
                boundaries = np.append(boundaries, i)
        boundaries = np.unique(np.append(boundaries, sf.shape[1]-1))

    seg_sim_mat = np.zeros((len(boundaries) - 1, len(boundaries) - 1))
    intervals = zip(boundaries[:-1], boundaries[1:])
    ssm[ssm > 1.0] = 1.0
    for i in range(len(boundaries) - 1):
        for j in range(len(boundaries) - 1):
            seg_sim_mat[i, j] = _segment_sim(ssm[intervals[i][0]:intervals[i][1], intervals[j][0]:intervals[j][1]])

    seg_sim_mat = (seg_sim_mat + seg_sim_mat.T) / 2
    seg_sim_mat[seg_sim_mat < (np.mean(seg_sim_mat) + np.std(seg_sim_mat))] = 0.0

    new_seg_mat = seg_sim_mat
    while True:
        new_seg_mat = np.dot(new_seg_mat, new_seg_mat)
        thresh_seg_mat = new_seg_mat
        new_seg_mat[new_seg_mat < 1.0] = 0.0
        new_seg_mat[new_seg_mat >= 1.0] = 1.0
        if np.array_equal(new_seg_mat, thresh_seg_mat):
            break

    labels = np.zeros(len(boundaries) - 1)
    for i in range(thresh_seg_mat.shape[0]):
        ind = np.nonzero(thresh_seg_mat[i, :])
        label_ind = 0
        for idx in ind[0]:
            if labels[idx]:
                if label_ind:
                    labels[idx] = label_ind
                else:
                    label_ind = labels[idx]
            else:
                if label_ind:
                    labels[idx] = label_ind
                else:
                    labels[idx] = i + 1
                    label_ind = i + 1
    return boundaries, labels


def _segment_sim(mat):
    u, v = mat.shape
    qmat = np.zeros((u, v))
    for i in range(u):
        for j in range(v):
            if i < 1 or j < 1:
                qmat[i, j] = mat[i, j]
            else:
                qmat[i, j] = np.max([qmat[i - 1, j - 1],
                                     qmat[i - 2, j - 1],
                                     qmat[i - 1, j - 2]]) + mat[i, j]
    return np.max(qmat) / np.min([u, v])


def hsmm_segmenter(factors, width=MEDIAN_WIDTH):
    obs = pre.normalize(factors, axis=1)
    obs_dim = obs.shape[1]
    obs_len = obs.shape[0]
    obs_hypparams = {
        'mu_0': np.mean(obs, axis=0),
        'sigma_0': np.cov(obs, rowvar=0),
        'kappa_0': 0.25,
        'nu_0': obs_dim + 2}

    dur_hypparams = {'alpha_0': 48,
                     'beta_0': 2}

    obs_distns = [distributions.Gaussian(**obs_hypparams) for state in xrange(Nmax)]
    dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

    posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        alpha_a_0=1., alpha_b_0=1./4,
        gamma_a_0=1., gamma_b_0=1./4,
        init_state_concentration=1.,  # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns)

    posteriormodel.add_data(factors, trunc=int(obs_len / 3))

    for idx in textutil.progprint_xrange(150):
        posteriormodel.resample_model()

    labels = posteriormodel.stateseqs
    boundaries, labels = find_boundaries(labels, width)
    best_n_types = len(np.unique(labels))

    if len(boundaries) < best_n_types + 1:
        best_n_types = len(boundaries) - 1

    best_labels = segment_labeling(factors, boundaries, c_method='kmeans', k=best_n_types)
    best_boundaries = np.array(boundaries)

    return best_boundaries, best_labels


def dpgmm_segmenter(factors, width=MEDIAN_WIDTH):
    factors = median_filter(factors, size=(MEDIAN_WIDTH, 1), mode='mirror')
    factors = pre.scale(factors, axis=1)
    best_boundaries = [0, factors.shape[0] - 1]
    best_n_types = 1

    dpgmm = DPGMM(n_components=10, covariance_type='diag', alpha=10, n_iter=100)
    dpgmm.fit(np.tile(factors, (10, 1)))
    labels = dpgmm.predict(factors)
    boundaries, labels = find_boundaries(labels, width)

    if len(np.unique(labels)) > 1:
        best_boundaries = boundaries
        best_n_types = len(np.unique(labels))

    if len(best_boundaries) < best_n_types + 1:
        best_n_types = len(best_boundaries) - 1

    best_labels = segment_labeling(factors, best_boundaries, c_method='kmeans', k=best_n_types)
    best_boundaries = np.array(best_boundaries)

    return best_boundaries, best_labels


def spectral_clustering_segmenter(feature=None, factor=None, k_min=K_MIN, k_max=K_MAX,
                                  width=MEDIAN_WIDTH, method='rec', m='sfx',
                                  sort=False, slice_basis=True):

    if factor is None:
        connectivity = recurrence_plot_extraction(feature, method=method, m=m)
        connectivity = diag_median_filter(connectivity, width=width)

        obs_len = connectivity.shape[0]
        connectivity[range(1, obs_len), range(obs_len - 1)] = 1.0
        connectivity[range(obs_len - 1), range(1, obs_len)] = 1.0
        connectivity[np.diag_indices(obs_len)] = 0

        graph_lap = normalized_graph_laplacian(connectivity)

        factors = eigen_decomposition(graph_lap, k=k_max)
    else:
        factors = factor

    factors = median_filter(factors, size=(width, 1), mode='mirror')
    best_score = -np.inf
    best_boundaries = [0, factors.shape[0] - 1]
    best_n_types = 1
    y_best = factors[np.argmax(np.std(factors, axis=1))]

    if sort:
        arg_indices = np.argsort(np.std(factors, axis=0))
        arg_indices = arg_indices[::-1]
        factors = factors[:, arg_indices]

    for n_types in range(k_min, int(1 + k_max)):
        if slice_basis:
            y = librosa.util.normalize(factors[:, :n_types], norm=2, axis=1)
        else:
            y = librosa.util.normalize(factors, norm=2, axis=1)

        # Try to label the data with n_types
        c = KMeans(n_clusters=n_types, n_init=100)
        labels = c.fit_predict(y)

        # Find the label change-points
        boundaries, labels = find_boundaries(labels, width)

        # boundaries now include start and end markers; n-1 is the number of segments
        if len(boundaries) < n_types + 1:
            n_types = len(boundaries) - 1

        values = np.unique(labels)
        hits = np.zeros(max(values) + 1)
        for v in values:
            hits[v] = np.sum(labels == v)
        hits = hits[hits != 0]
        hits /= float(hits.sum())
        score = stats.entropy(hits) / np.log(n_types)

        if score > best_score:
            best_boundaries = boundaries
            best_n_types = n_types
            best_score = score
            y_best = y

    # Classify each segment centroid

    labels = segment_labeling(y_best, best_boundaries, c_method='kmeans', k=best_n_types)
    best_labels = labels

    return best_boundaries, best_labels


def recurrence_plot_extraction(feature, d='cosine', method='rec', **kwargs):
    f = copy.copy(feature)
    if method == 'rec':
        return _self_sim(f, d)
    elif method == 'vmo':
        return _self_sim_vmo(f, d, **kwargs)
    else:
        return _self_sim(f, d)


def ssm_extraction(feature, d='cosine', method='default'):
    f = copy.copy(feature)
    if method == 'default':
        return _self_sim_naive(feature, d)
    elif method == 'sc':
        return _self_sim_sc(f, d)
    else:
        return _self_sim_naive(f, d)


def _self_sim_vmo(feature, d, m='lrs'):
    if d == 'cosine' or 'correlation':
        r = (0., 1., 0.02)
    elif d == 'euclidean':
        r = (0., 30., .5)
    elif d == 'sqeuclidean':
        r = (0., 800., 16.)

    ideal_t = vmo.find_threshold(feature, r=r, flag='a', dfunc=d, dim=feature.shape[1])

    oracle = vmo.build_oracle(feature,
                              flag='a',
                              threshold=ideal_t[0][1],
                              dfunc=d, dim=feature.shape[1])
    return vse.create_selfsim(oracle, method=m)


def _self_sim(feature, d):
    k_link = 1 + np.ceil(2 * np.log2(feature.shape[0]))
    recurrence = librosa.segment.recurrence_matrix(feature.T,
                                                   k=k_link,
                                                   width=1,
                                                   metric=d,
                                                   sym=True).astype(np.float32)

    # filter the recurrence plot by diagonal majority vote

    return recurrence


def _self_sim_sc(feature, d):
    k = 1 + np.ceil(2 * np.log2(feature.shape[0]))
    feature += np.finfo('float').eps
    D = scipy.spatial.distance.cdist(feature, feature, metric=d)
    sigma = estimate_bandwidth(D, int(k))
    A = np.exp(-0.5 * (D / sigma))
    return A

def _self_sim_naive(feature, d):
    return 1.0 - scipy.spatial.distance.cdist(feature, feature, metric=d)

def estimate_bandwidth(D, k):
    '''Estimate the bandwidth of a gaussian kernel.

    sigma is computed as the average distance between
    each point and its kth nearest neighbor.

    :parameters:
        - D : np.ndarray [shape=(n, n)]
          A squared euclidean distance matrix

        - k : int > 0
          Number of neighbors to use

    :returns:
        - sigma : float > 0
          Estimated bandwidth
    '''

    D_sort = np.sort(D, axis=1)

    if 1 + k >= len(D):
        k = len(D) - 2

    sigma = np.mean(D_sort[:, 1 + k])
    return sigma


def diag_median_filter(mat, width=9):
    diagonal_median = librosa.segment.timelag_filter(median_filter)
    rec_filtered = diagonal_median(mat, size=(1, width), mode='mirror')

    return rec_filtered


def diag_gauss_filter(mat, width=9):
    diagonal_median = librosa.segment.timelag_filter(gaussian_filter)
    rec_filtered = diagonal_median(mat, sigma=(1, width), mode='mirror')

    return rec_filtered


def normalized_graph_laplacian(mat):
    mat_inv = 1. / np.sum(mat, axis=1)
    mat_inv[~np.isfinite(mat_inv)] = 1.
    mat_inv = np.diag(mat_inv ** 0.5)
    laplacian = np.eye(len(mat)) - mat_inv.dot(mat.dot(mat_inv))

    return laplacian


def eigen_decomposition(mat, k=5):  # Changed from 11 to 8 then to 6(7/22)
    vals, vecs = linalg.eig(mat)
    vals = vals.real
    vecs = vecs.real
    idx = np.argsort(vals)

    vals = vals[idx]
    vecs = vecs[:, idx]

    if len(vals) < k + 1:
        k = -1

    return vecs[:, :k]


# def process_label(lab, sr=22050, hop_size=512):
#     (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(lab)
#     truth_dict = {}
#     for label, interval in zip(ref_labels, ref_intervals):
#         frames = librosa.time_to_frames(interval, sr=sr, hop_length=hop_size)
#         if label in truth_dict:
#             truth_dict[label]['times'].append(interval)
#             truth_dict[label]['frames'].append(frames)
#         else:
#             truth_dict[label] = {'times': [interval], 'frames': [frames]}
#     return truth_dict


def save_segments(outfile, boundaries, beat_intervals, labels=None):
    """Save detected segments to a .lab file.

    :parameters:
        - outfile : str
            Path to output file

        - boundaries : list of int
            Beat indices of detected segment boundaries

        - beat_intervals : np.ndarray [shape=(n, 2)]
            Intervals of beats

        - labels : None or list of str
            Labels of detected segments
    """

    if labels is None:
        labels = [('Seg#%03d' % idx) for idx in range(1, len(boundaries))]

    times = [beat_intervals[b, 0] for b in boundaries[:-1]]
    times.append(beat_intervals[-1, -1])

    with open(outfile, 'w') as f:
        for idx, (start, end, lab) in enumerate(zip(times[:-1],
                                                    times[1:],
                                                    labels), 1):
            f.write('%.3f\t%.3f\t%s\n' % (start, end, lab))


def save_results(outfile_path, results):
    """
    Write a result dictionary out as a .json file.
    :parameters:
        - results : dict
            Results dictionary, where keys are metric names and values are
            the corresponding scores
        - output_file : str
            Path to .json file to write to
    """
    with open(outfile_path, 'w+') as f:
        json.dump(results, f)


def save_seg_plot(outfile_path, title_str1, title_str2,
                  ref_intervals, ref_labels, est_intervals, est_labels,
                  cmap='Accent'):
    fig = plt.figure(figsize=(9, 3))

    ulabel, invind = np.unique(ref_labels, return_inverse=True)
    norm = mpl.colors.Normalize(vmin=np.min(invind), vmax=np.max(invind))
    cmap = cm.get_cmap(cmap)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    plt.subplot(211)
    seg = [interval[0] for interval in ref_intervals[1:]]
    plt.vlines(seg, 0, 1, linewidth=1)
    for interval, i in zip(ref_intervals, invind):
        plt.fill_between([interval[0], interval[1]], 0, 1, color=m.to_rgba(i))
    plt.title('Ground truth segmentation - ' + title_str1)
    plt.xlim(0, ref_intervals[-1][1])
    plt.yticks([])
    plt.xticks([])

    ax = fig.add_subplot(212)
    ulabel, invind = np.unique(est_labels, return_inverse=True)
    norm = mpl.colors.Normalize(vmin=np.min(invind), vmax=np.max(invind))
    cmap = cm.get_cmap(cmap)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    seg = [interval[0] for interval in est_intervals[1:]]
    plt.vlines(seg, 0, 1, linewidth=1)
    for interval, i in zip(est_intervals, invind):
        plt.fill_between([interval[0], interval[1]], 0, 1, color=m.to_rgba(i))
    plt.title('Detected segmentation - ' + title_str2)
    plt.xlim(0, est_intervals[-1][1])
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(outfile_path)


def process_border(boundary, label, start, ending, n_beat):
    z = np.max(label) + 1
    if start > 0:
        label = np.insert(label, 0, z)
        boundary += 1
        boundary = np.insert(boundary, 0, 0)
    if ending != -1:
        label = np.append(label, z)
        boundary = np.append(boundary, n_beat)
    return boundary, label


def create_score_dict():
    average_scores = collections.OrderedDict()
    average_scores['Precision@0.5'] = 0.0
    average_scores['Recall@0.5'] = 0.0
    average_scores['F-measure@0.5'] = 0.0
    average_scores['Precision@3.0'] = 0.0
    average_scores['Recall@3.0'] = 0.0
    average_scores['F-measure@3.0'] = 0.0
    average_scores['Ref-to-est deviation'] = 0.0
    average_scores['Est-to-ref deviation'] = 0.0
    average_scores['Pairwise Precision'] = 0.0
    average_scores['Pairwise Recall'] = 0.0
    average_scores['Pairwise F-measure'] = 0.0
    average_scores['Rand Index'] = 0.0
    average_scores['Adjusted Rand Index'] = 0.0
    average_scores['Mutual Information'] = 0.0
    average_scores['Adjusted Mutual Information'] = 0.0
    average_scores['Normalized Mutual Information'] = 0.0
    average_scores['NCE Over'] = 0.0
    average_scores['NCE Under'] = 0.0
    average_scores['NCE F-measure'] = 0.0

    return average_scores


def ground_truth_split(ref_intervals, ref_labels, beats, sr=SAMPLE_RATE, hop_length=HOP_SIZE):

    ref_boundaries = mir_eval.util.intervals_to_boundaries(ref_intervals)

    new_bound_one = [0.0]
    new_bound_two = [0.0]

    new_label_one = []
    new_label_two = []

    for i, label in enumerate(ref_labels):
        choice = np.random.choice([0, 1, 2], size=1)
        if choice[0] == 1:
            new_label_two.append(label)
            new_bound_two.append(ref_boundaries[i+1])
        elif choice[0] == 0:
            new_label_one.append(label)
            new_bound_one.append(ref_boundaries[i+1])
        else:
            new_label_two.append(label)
            new_bound_two.append(ref_boundaries[i+1])
            new_label_one.append(label)
            new_bound_one.append(ref_boundaries[i+1])
    if choice[0] == 1:
        new_label_one.append(ref_labels[-1])
        new_bound_one.append(ref_boundaries[-1])
    elif choice[0] == 0:
        new_label_two.append(ref_labels[-1])
        new_bound_two.append(ref_boundaries[-1])
    new_interval_one = mir_eval.util.boundaries_to_intervals(new_bound_one)
    new_interval_two = mir_eval.util.boundaries_to_intervals(new_bound_two)

    mat_one = annotation_to_mat(new_interval_one, new_label_one, beats, sr=sr, hop_length=hop_length)
    mat_two = annotation_to_mat(new_interval_two, new_label_two, beats, sr=sr, hop_length=hop_length)

    return mat_one, mat_two


def annotation_to_mat(ref_intervals, ref_labels, beats, sr=SAMPLE_RATE, hop_length=HOP_SIZE):

    truth_dict = {}
    for label, interval in zip(ref_labels, ref_intervals):
        frames = librosa.time_to_frames(interval, sr=sr, hop_length=hop_length)
        if label in truth_dict:
            truth_dict[label]['times'].append(interval)
            truth_dict[label]['frames'].append(frames)
        else:
            truth_dict[label] = {'times': [interval], 'frames': [frames]}

    for k in truth_dict:
        truth_dict[k]['beats'] = []
        for interval in truth_dict[k]['frames']:
            beat_interval = np.argmin(np.abs([interval[0]-beats, interval[1]-beats]), axis=1)
            truth_dict[k]['beats'].append(beat_interval)

    truth_mat_beat = np.zeros((len(beats)-1, len(beats)-1))
    for k in truth_dict:
        for i_b in truth_dict[k]['beats']:
            for j_b in truth_dict[k]['beats']:
                if np.array_equal(i_b, j_b):
                    truth_mat_beat[i_b[0]:i_b[1], j_b[0]:j_b[1]] = 1
                else:
                    truth_mat_beat[i_b[0]:i_b[1], j_b[0]:j_b[1]] = 0.9
    return truth_mat_beat
