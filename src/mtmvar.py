"""
MVAR Analysis Functions for EEG Connectivity Analysis

This module contains functions for Multivariate Autoregressive (MVAR) modeling
and Directed Transfer Function (DTF) analysis of EEG data.

ACKNOWLEDGMENTS:
Some functions in this module are based on algorithms and code originally 
developed by Prof. Maciej Kamiński from the Department of Biomedical Physics 
at the University of Warsaw. These contributions form part of the theoretical 
and methodological foundation for MVAR-based connectivity analysis.

Original MVAR methodology references:
- Kamiński, M., & Blinowska, K. J. (1991). A new method of the description 
  of the information flow in the brain structures. Biological Cybernetics, 
  65(3), 203-210.
- Kamiński, M., Ding, M., Truccolo, W. A., & Bressler, S. L. (2001).
  Evaluating causal relations in neural systems: Granger causality, directed 
  transfer function and statistical assessment of significance. Biological 
  Cybernetics, 85(2), 145-157.

Modified and adapted for educational purposes by:
Jarosław Żygierewicz, University of Warsaw
SYNCCIN 2025 Summer School

License: Educational use
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def count_corr(x, ip, iwhat):
    """
    Internal procedure
    """
    sdt = np.shape(x)
    m = sdt[0]
    n = sdt[1]
    if len(sdt) > 2:
        trials = sdt[2]
    else:
        trials = 1
    mip = m * ip
    r_left = np.zeros((mip, mip))
    r_right = np.zeros((mip, m))
    r = np.zeros((m, m))
    r_left_tot = np.zeros((mip, mip))
    r_right_tot = np.zeros((mip, m))
    r_tot = np.zeros((m, m))

    for trial in range(trials):
        for k in range(ip):
            if iwhat == 1:
                corr_scale = 1 / n
                nn = n - k - 1
                r[:, :] = np.dot(x[:, :nn, trial], x[:, k + 1:k + nn + 1, trial].T) * corr_scale
            elif iwhat == 2:
                corr_scale = 1 / (n - k)
                nn = n - k - 1
                r[:, :] = np.dot(x[:, :nn, trial], x[:, k + 1:k + nn + 1, trial].T) * corr_scale

            r_right[k * m:k * m + m, :] = r[:, :]

            if k < ip:
                for i in range(1, ip - k):
                    r_left[(k + i) * m:(k + i) * m + m, (i - 1) * m:(i - 1) * m + m] = r[:, :]
                    r_left[(i - 1) * m:(i - 1) * m + m, (k + i) * m:(k + i) * m + m] = r[:, :].T

        corr_scale = 1 / n
        r[:, :] = np.dot(x[:, :, trial], x[:, :, trial].T) * corr_scale

        for k in range(ip):
            r_left[k * m:k * m + m, k * m:k * m + m] = r[:, :]

        r_left_tot = r_left_tot + r_left
        r_right_tot = r_right_tot + r_right
        r_tot = r_tot + r

    if trials > 1:
        r_left_tot = r_left_tot / trials
        r_right_tot = r_right_tot / trials
        r_tot = r_tot / trials

    return r_left_tot, r_right_tot, r_tot


def ar_coeff(data, model_order=5):
    """
    Estimate MVAR coefficients for multivariate / multi-trial data.

    Parameters:
    data : np.ndarray
        Input time series with shape (channels, samples) or (channels, samples, trials).
    model_order : int
        Model order.

    Returns:
    ar_coeffs : np.ndarray
        AR coefficients with shape (channels, channels, model_order).
    variance : np.ndarray
        Residual covariance matrix with shape (channels, channels).
    """
    # Ensure data has a trials dimension: (channels, samples, trials)
    if data.ndim < 3:
        data = data[:, :, None]

    n_channels = data.shape[0]

    # Compute correlation/block-correlation matrices
    r_left, r_right, r_zero = count_corr(data, model_order, 1)

    # Solve for stacked AR coefficients (shape: (n_channels * p, n_channels))
    x = np.linalg.solve(r_left, r_right).T

    # Residual covariance: r_zero - x @ r_right
    variance = r_zero - x.dot(r_right)

    # Reshape coefficients into (channels, channels, p) to match previous behavior
    ar_coeffs = x.reshape(n_channels, model_order, n_channels).transpose((0, 2, 1))
    return ar_coeffs, variance


def mvar_transfer_function(ar_coeffs, freqs, fs):
    """
    Calculate the transfer function H from multivariate autoregressive coefficients AR
    
    Parameters:
    ar_coeffs : numpy.ndarray
        AR coefficient matrix with shape (chan, chan, p), where p is the model order.
    freqs : numpy.ndarray
        Frequency vector.
    fs : float
        Sampling frequency.

    Returns:
    transfer_function_matrix : numpy.ndarray
        Transfer function matrix with shape (chan, chan, len(freqs)).
    ar_matrix : numpy.ndarray
        Frequency-dependent AR matrix with shape (chan, chan, len(freqs)).
    """
    model_order = ar_coeffs.shape[2]
    n_freqs = len(freqs)
    chan = ar_coeffs.shape[0]

    transfer_function_matrix = np.zeros((chan, chan, n_freqs), dtype=complex)
    ar_matrix = np.zeros((chan, chan, n_freqs), dtype=complex)

    z = np.zeros((model_order, n_freqs), dtype=complex)
    for m in range(1, model_order + 1):
        z[m - 1, :] = np.exp(-m * 2 * np.pi * 1j * freqs / fs)

    for fi in range(n_freqs):
        a = np.eye(chan, dtype=complex)
        for m in range(model_order):
            a -= ar_coeffs[:, :, m] * z[m, fi].item()
        transfer_function_matrix[:, :, fi] = np.linalg.inv(a)
        ar_matrix[:, :, fi] = a

    return transfer_function_matrix, ar_matrix


def multivariate_spectra(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC'):
    """
    Compute the multivariate spectra for all channels in signals.
    
    Parameters:
    signals : np.ndarray
        Input signals of shape (N_chan, N_samp).
    freqs : np.ndarray
        Frequency vector.
    fs : float
        Sampling frequency.
    max_model_order : int
        Maximum model order.
    optimal_model_order : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
    
    Returns:
    spectra : np.ndarray
        Multivariate spectra of shape (N_chan, N_chan, N_f).
    """
    if optimal_model_order is None:
        _, _, optimal_model_order = mvar_criterion(signals, max_model_order, crit_type, True)
        print('Optimal model order for all channels: p = ', str(optimal_model_order))
    else:
        print('Using provided model order: p = ', str(optimal_model_order))
    # Estimate AR coefficients and residual variance
    ar_coeffs, variance = ar_coeff(signals, optimal_model_order)
    transfer_function_matrix, _ = mvar_transfer_function(ar_coeffs, freqs, fs)
    n_chan = signals.shape[0]
    n_freqs = freqs.shape[0]
    spectra = np.zeros((n_chan, n_chan, n_freqs), dtype=np.complex128)  # initialize the multivariate spectrum
    for fi in range(n_freqs):  # compute spectrum for all channels
        spectra[:, :, fi] = transfer_function_matrix[:, :, fi].dot(variance.dot(transfer_function_matrix[:, :, fi].T))

    return spectra


def dtf_multivariate(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC', comment=None):
    """
    Compute the directed transfer function (DTF) for the multivariate case.
    Parameters:
    signals : np.ndarray    
        Input signals of shape (N_chan, N_samp).
    freqs : np.ndarray
        Frequency vector.
    fs : float
        Sampling frequency.
    max_model_order : int
        Maximum model order.
    optimal_model_order : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
    Returns:
    np.ndarray
        Multivariate DTF of shape (N_chan, N_chan, N_f).
    """
    if optimal_model_order is None:
        _, _, optimal_model_order = mvar_criterion(signals, max_model_order, crit_type, False)
        print(
            f'Optimal model order for all {'' if comment is None else comment + ' '}channels: p = {optimal_model_order}')
    else:
        print(f'Using provided model order: p = {optimal_model_order}')
    ar_coeffs, _ = ar_coeff(signals, optimal_model_order)
    transfer_function_matrix, _ = mvar_transfer_function(ar_coeffs, freqs, fs)
    dtf = np.abs(transfer_function_matrix) ** 2

    return dtf


def full_freq_dtf(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC'):
    """
    Compute the full-frequency directed transfer function (ffDTF) for the multivariate case.
    
    The ffDTF modifies the standard DTF normalization to integrate over the entire frequency 
    range, making cross-frequency comparisons more interpretable. Unlike standard DTF which 
    normalizes by frequency-specific inflows, ffDTF normalizes by the sum over all frequencies.
    
    Mathematical formula:
    Standard DTF: DTF_ij(f) = |H_ij(f)|² / Σ_k |H_ik(f)|²
    Full-frequency DTF: ffDTF_ij(f) = |H_ij(f)|² / Σ_f Σ_k |H_ik(f)|²
    
    The normalization takes into account inflows to channel i across the entire frequency range,
    allowing for more meaningful comparison of information flow at different frequencies.
    
    Reference:
    Korzeniewska, A., Mańczak, M., Kamiński, M., Blinowska, K. J., & Kasicki, S. (2003).
    Determination of information flow direction among brain structures by a modified directed transfer function (dDTF) method. 
    Journal of neuroscience methods, 125(1-2), 195-207.
    https://doi.org/10.1016/S0165-0270(03)00052-9
    
    Parameters:
    signals : np.ndarray
        Input signals of shape (N_chan, N_samp).
    freqs : np.ndarray
        Frequency vector.
    fs : float
        Sampling frequency.
    max_model_order : int
        Maximum model order.
    optimal_model_order : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
        
    Returns:
    np.ndarray
        Full-frequency DTF of shape (N_chan, N_chan, N_f).
    """
    dtf = dtf_multivariate(signals, freqs, fs, max_model_order, optimal_model_order, crit_type)

    n_chan, _, n_freqs = dtf.shape
    # Normalize DTF to get full-frequency DTF (ffDTF)
    ff_dtf = np.zeros((n_chan, n_chan, n_freqs))
    for i in range(n_chan):  # rows
        for j in range(n_chan):  # columns
            ff_dtf[i, j, :] = dtf[i, j, :] / np.sum(dtf[i, :, :])
    return ff_dtf


def partial_coherence(spectra):
    """
    Compute the partial coherence using efficient boolean indexing for minors.
    
    Parameters:
    spectra : np.ndarray
        Spectral density matrix of shape (N_chan, N_chan, N_f).

    Returns:
    np.ndarray
        Partial coherence of shape (N_chan, N_chan, N_f).
    """
    n_chan, _, n_f = spectra.shape

    # Compute minors of spectra using boolean indexing (Method 2)
    spectra_minors = np.zeros((n_chan, n_chan, n_f), dtype=np.complex128)

    for i in range(n_chan):  # rows to delete
        for j in range(n_chan):  # columns to delete
            for fi in range(n_f):  # for each frequency
                # Extract the (N_chan x N_chan) matrix at frequency fi
                spectra_freq = spectra[:, :, fi]

                # Create minor by deleting row i and column j using boolean indexing
                if n_chan > 1:  # Only compute minor if matrix is larger than 1x1
                    # Create boolean masks
                    row_mask = np.ones(n_chan, dtype=bool)
                    col_mask = np.ones(n_chan, dtype=bool)
                    row_mask[i] = False
                    col_mask[j] = False

                    # Extract submatrix using boolean indexing
                    minor_matrix = spectra_freq[np.ix_(row_mask, col_mask)]
                    spectra_minors[i, j, fi] = np.linalg.det(minor_matrix)
                else:
                    spectra_minors[i, j, fi] = 1.0  # For 1x1 matrix, minor is 1

    # Compute partial coherence
    kappa = np.zeros((n_chan, n_chan, n_f), dtype=np.complex128)
    for i in range(n_chan):
        for j in range(n_chan):
            if i != j:  # Only compute for off-diagonal elements
                # Avoid division by zero
                denominator = np.sqrt(spectra_minors[i, i, :] * spectra_minors[j, j, :])
                kappa[i, j, :] = np.where(denominator != 0,
                                          spectra_minors[i, j, :] / denominator,
                                          0)
            # Diagonal elements are set to 1 (perfect coherence with itself)
            else:
                kappa[i, j, :] = 1.0

    return kappa


def direct_dtf(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC'):
    """
    Compute the direct directed transfer function (dDTF) for the multivariate case.
    
    The dDTF combines DTF with partial coherence to distinguish direct from indirect 
    connections, filtering out spurious connections that may arise from common inputs
    or indirect pathways.
    
    Mathematical formula:
    dDTF_ij(f) = DTF_ij(f) × |partial_coherence_ij(f)|
    
    Reference:
    Korzeniewska, A., Mańczak, M., Kamiński, M., Blinowska, K. J., & Kasicki, S. (2003).
    Determination of information flow direction among brain structures by a modified 
    directed transfer function (dDTF) method. Journal of Neuroscience Methods, 125(1-2), 195-207.
    https://doi.org/10.1016/S0165-0270(03)00052-9
    
    Parameters:
    signals : np.ndarray    
        Input signals of shape (N_chan, N_samp).
    freqs : np.ndarray
        Frequency vector.
    fs : float
        Sampling frequency.
    max_model_order : int
        Maximum model order.
    optimal_model_order : int, optional
        Optimal model order (if known).
    crit_type : str
        Criterion type for model order selection (e.g., 'AIC', 'BIC').
        
    Returns:
    np.ndarray
        Direct directed transfer function (dDTF) of shape (N_chan, N_chan, N_f).
    """
    # compute spectral density matrix
    spectra = multivariate_spectra(signals, freqs, fs, max_model_order, optimal_model_order, crit_type)
    # Compute partial coherence
    kappa = partial_coherence(spectra)
    # compute the ffDTF
    ff_dtf = full_freq_dtf(signals, freqs, fs, max_model_order, optimal_model_order, crit_type)
    # Compute dDTF using the formula: dDTF = ff_DTF * |kappa|
    d_dtf = ff_dtf * np.abs(kappa)

    return d_dtf


def gen_partial_directed_coherence(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC'):
    """
    Compute the generalized partial directed coherence (GPDC) for the multivariate case.
    
    GPDC is a frequency-domain measure that quantifies the directed influence of one 
    signal on another, based on the autoregressive coefficients in the frequency domain.
    This generalized version addresses the limitations of standard PDC when dealing with 
    systems that have unbalanced noise variances across channels by incorporating 
    proper normalization using the noise covariance structure.
    
    Mathematical formula:
    GPDC_ij(f) = |A_ij(f)|/σ_i / sqrt(Σ_k (|A_kj(f)|² / σ²_k))

    Where:
    - A(f) is the frequency domain AR coefficient matrix (inverse of H)
    - |A_ij(f)| is the absolute value of the AR coefficient from source channel i to target channel j at frequency f
    - σ_i is the noise standard deviation for source channel i (sqrt of diagonal elements of noise covariance matrix V)
    - σ²_k is the noise variance for source channel k (diagonal elements of noise covariance matrix V)
    
    This formulation ensures that channels with different noise levels do not 
    artificially bias connectivity estimates, making GPDC more robust than standard PDC 
    for practical applications with heterogeneous noise characteristics.
    
    References:
    Original PDC: Baccala, L. A., & Sameshima, K. (2001). Partial directed coherence: a new concept 
    in neural structure determination. Biological Cybernetics, 84(6), 463-474.
    https://doi.org/10.1007/PL00007990
    
    Generalized PDC: Baccalá, L. A., & Sameshima, K. (2007). Generalized partial directed coherence. 
    15th International Conference on Digital Signal Processing (DSP 2007), 163-166.
    https://doi.org/10.1109/ICDSP.2007.4288544
    
    Parameters:
    signals : np.ndarray    
        Input signals of shape (N_chan, N_samp).
    freqs : np.ndarray
        Frequency vector.
    fs : float
        Sampling frequency.
    max_model_order : int
        Maximum model order.
    optimal_model_order : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
        
    Returns:
    np.ndarray
        Generalized partial directed coherence (GPDC) of shape (N_chan, N_chan, N_f).
    """
    if optimal_model_order is None:
        _, _, optimal_model_order = mvar_criterion(signals, max_model_order, crit_type, False)
        print('Optimal model order for all channels: p = ', str(optimal_model_order))
    else:
        print('Using provided model order: p = ', str(optimal_model_order))

    # Estimate AR coefficients and residual variance
    ar_coeffs, variance = ar_coeff(signals, optimal_model_order)
    _, ar_matrix = mvar_transfer_function(ar_coeffs, freqs, fs)  # A is the frequency domain AR matrix

    n_chan, _, n_f = ar_matrix.shape
    gpdc = np.zeros((n_chan, n_chan, n_f))

    # Extract noise variances (diagonal of V)
    sigma_squared = np.diag(variance)

    # Compute GPDC 
    for i in range(n_chan):  # source
        for j in range(n_chan):  # target
            # Numerator: |A_ij(f)| / σ_i
            numerator = np.abs(ar_matrix[i, j, :]) / np.sqrt(sigma_squared[i])

            # Denominator: sqrt(Σ_k (|A_kj(f)|² / σ²_k))
            denominator = np.sqrt(np.sum(np.abs(ar_matrix[:, j, :]) ** 2 / sigma_squared[:, np.newaxis], axis=0))

            # Avoid division by zero
            gpdc[i, j, :] = np.where(denominator != 0,
                                     numerator / denominator,
                                     0)

    return gpdc


# Plotting function for graph visualization
def mvar_plot(on_diag, off_diag, freqs, x_label, y_label, chan_names, top_title, scale='linear'):
    """
    Plot MVAR results using bar plots for diagonal (auto) and off-diagonal (cross) terms.

    Parameters:
    on_diag : np.ndarray
        Auto components (shape: N_chan x N_chan x len(freqs))
    off_diag : np.ndarray
        Cross components (shape: N_chan x N_chan x len(freqs))
    freqs : np.ndarray
        Frequency vector
    x_label : str
        Label for x-axis
    y_label : str
        Label for y-axis
    chan_names : list of str
        Names of channels
    top_title : str
        Main plot title
    scale : str
        'linear', 'sqrt', or 'log'
    """
    on_diag = np.abs(on_diag)
    off_diag = np.abs(off_diag)

    if scale == 'sqrt':
        on_diag = np.sqrt(on_diag)
        off_diag = np.sqrt(off_diag)
    elif scale == 'log':
        on_diag = np.log(on_diag + 1e-12)  # Avoid log(0)
        off_diag = np.log(off_diag + 1e-12)

    n_chan = on_diag.shape[0]

    # Zero-out irrelevant parts
    for i in range(n_chan):
        for j in range(n_chan):
            if i != j:
                on_diag[i, j, :] = 0
            else:
                off_diag[i, i, :] = 0

    max_on_diag = np.max(on_diag)
    max_off_diag = np.max(off_diag)

    _, axs = plt.subplots(n_chan, n_chan, figsize=(8, 8),
                          gridspec_kw={'wspace': 0, 'hspace': 0})

    for i in range(n_chan):
        for j in range(n_chan):
            ax = axs[i, j] if n_chan > 1 else axs
            if i != j:
                y = np.real(off_diag[i, j, :])
                ax.plot(freqs, off_diag[i, j, :])
                ax.fill_between(freqs, y, 0, color='skyblue', alpha=0.4)
                ax.set_ylim([0, max_off_diag])
                ax.set_yticks([0, max_off_diag // 2])
            else:
                y = np.real(on_diag[i, j, :])
                ax.plot(freqs, y, color=[0.7, 0.7, 0.7])
                ax.fill_between(freqs, y, 0, color=[0.7, 0.7, 0.7], alpha=0.4)
                ax.set_ylim([0, max_on_diag])
                ax.set_yticks([0, max_on_diag // 2])

            ax.set_xticks([freqs[0], int(freqs[len(freqs) // 2])])
            ax.tick_params(labelleft=(j == 0), labelbottom=(i == n_chan - 1))

            if i == n_chan - 1:
                ax.set_xlabel(f"{x_label}{chan_names[j]}")
            if j == 0:
                ax.set_ylabel(f"{y_label}{chan_names[i]}")

    if n_chan > 1:
        axs[0, 0].set_title(top_title)
    else:
        axs.set_title(top_title)
    # plt.tight_layout()


def mvar_criterion(data, max_model_order, crit_type='AIC', plot=False):
    """
    Compute model order selection criteria (AIC, HQ, SC) for MVAR modeling.

    Parameters:
    data : np.ndarray
        Input data of shape (channels, samples).
    max_model_order : int
        Maximum model order to evaluate.
    crit_type : str
        Criterion type: 'AIC', 'HQ', or 'SC'.
    plot : bool
        Whether to plot the criterion values.

    Returns:
    crit : np.ndarray
        Criterion values for each model order.
    p_range : np.ndarray
        Evaluated model order range (1:max_model_order).
    optimal_model_order : int
        Optimal model order (minimizing the criterion).
    """
    n_channels, n_samples = data.shape
    model_order_range = np.arange(1, max_model_order + 1, dtype=int)
    crit = np.zeros(max_model_order)

    for model_order in model_order_range:
        _, variance = ar_coeff(data, model_order)  # You must define or import AR_coeff
        if crit_type == 'AIC':
            crit[model_order - 1] = np.log(np.linalg.det(variance)) + 2 * model_order * n_channels ** 2 / n_samples
        elif crit_type == 'HQ':
            crit[model_order - 1] = np.log(np.linalg.det(variance)) + 2 * np.log(
                np.log(n_samples)) * model_order * n_channels ** 2 / n_samples
        elif crit_type == 'SC':
            crit[model_order - 1] = np.log(np.linalg.det(variance)) + np.log(
                n_samples) * model_order * n_channels ** 2 / n_samples
        else:
            raise ValueError("Invalid criterion type. Choose from 'AIC', 'HQ', 'SC'.")

    optimal_model_range = model_order_range[np.argmin(crit)]
    if plot:
        plt.figure()
        plt.plot(model_order_range, crit, marker='o')
        plt.plot(optimal_model_range, np.min(crit), 'ro')
        plt.xlabel('Model order p')
        plt.ylabel(f'{crit_type} criterion')
        plt.title(f'MVAR Model Order Selection ({crit_type}). The best order = {optimal_model_range}')
        plt.grid(True)
        plt.show()

    return crit, model_order_range, optimal_model_range


# Compute linewidths
def get_linewidths(graph):
    weights = np.array([d['weight'] for u, v, d in graph.edges(data=True)])
    return 5 * weights / weights.max()


def graph_plot(connectivity_matrix, ax, freqs, freq_range, chan_names, title):
    """
    Plot connectivity matrix as a graph.
    Parameters:
    connectivity_matrix : np.ndarray, shape (N_chan, N_chan, N_f)
        Connectivity matrix with complex values. e.g. Directed Transfer Function (DTF).
    ax : matplotlib.axes.Axes
        Axes on which to plot the graph.
    freqs : np.ndarray
        Frequency vector.
    f_range : tuple
        Frequency range for the plot (min, max).
    chan_names : list
        List of channel names.
    title : str
        Title for the plot. 
    Returns:
    G : networkx.DiGraph
        Directed graph created from the connectivity matrix.

    """
    # Convert complex DTF to real for visualization
    connectivity = connectivity_matrix.real
    # Sum over the frequencies in f_range and transpose to match the directionality of edges (from row to column) in the directed graph
    # find the indices of the frequency range
    f_indices = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
    if len(f_indices) == 0:
        raise ValueError("No frequencies found in the specified range.")
    adj = np.sum(connectivity[:, :, f_indices], axis=2).T
    # Remove self-loops by setting diagonal to zero
    np.fill_diagonal(adj, 0)
    # Create directed graphs
    graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    # Plotting
    pos = nx.spring_layout(graph)  # use the same layout for both
    # Map chan_names to node labels
    labels = {i: chan_names[i] for i in range(len(chan_names))}
    nx.draw(graph, pos, ax=ax, with_labels=True, labels=labels, arrows=True,
            width=get_linewidths(graph), node_size=500,
            arrowstyle='->',
            arrowsize=35,
            connectionstyle='arc3,rad=0.2')
    ax.set_title(title)

    return graph
