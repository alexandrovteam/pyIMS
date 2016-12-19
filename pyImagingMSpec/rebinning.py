import numpy as np

def get_peak_in_bin_idx(mzs, edge_lower, edge_upper):
    mz_count = np.arange(np.searchsorted(mzs, edge_lower), np.searchsorted(mzs, edge_upper), dtype=int)
    return mz_count


def peaks_from_spectra(mzs, specix, edge_lower, edge_upper):
    peaks_in_this_bin_ix = specix[get_peak_in_bin_idx(mzs, edge_lower, edge_upper)]
    return np.max(np.bincount(peaks_in_this_bin_ix, minlength=1))


def get_edges(mzs, specix, max_width=0.1, max_ups=1):
    bin_edges_pos = [mzs[0],mzs[1]]
    for mz in mzs[1:]:
        ups = peaks_from_spectra(mzs, specix, bin_edges_pos[-1], bin_edges_pos[-2])
        if ups > max_ups:
            bin_edges_pos.append(mz)
        bin_edges_pos[-1] = mz

    bin_edges_neg = [mzs[-1],mzs[-2]]
    for mz in mzs[::-1]:
        ups = peaks_from_spectra(mzs, specix, bin_edges_neg[-1], bin_edges_neg[-2])
        if ups > max_ups:
            bin_edges_neg.append(mz)
        bin_edges_neg[-1] = mz
    edges = np.unique(np.round(np.concatenate((bin_edges_pos, bin_edges_neg)), decimals=5))
    delta_edges = np.diff(edges)
    extra_edges = []
    for ii, delta in enumerate(delta_edges):
        if delta >= max_width:
            extra_edges.extend(np.linspace(edges[ii], edges[ii+1], int(delta/max_width))[1:-1])
    edges = np.unique(np.concatenate([edges, np.asarray(extra_edges)]))
    return edges


def get_some_real_spectra(imsDataset, n):
    #get n spectra from an object of type imsDataset (or any class that has a get_spectrum method)
    # returns a sorted list of mzs, intensities, pixel_index
    all_mzs = []
    all_intensities = []
    all_ix = []
    if n > len(imsDataset.coordinates):
        n = len(imsDataset.coordinates)
    for ii in np.linspace(0, len(imsDataset.coordinates)-1, n, dtype=int):
        real_mzs, real_intensities = map(np.asarray, imsDataset.get_spectrum(ii))
        all_mzs.extend([mz for mz in real_mzs])
        all_intensities.extend([ri for ri in real_intensities])
        all_ix.extend([ii for mz in real_mzs])
    mzs = np.asarray(all_mzs).flatten()
    ints = np.asarray(all_intensities).flatten()
    specix = np.asarray(all_ix).flatten()
    six = np.argsort(mzs)
    mzs = mzs[six]
    specix = specix[six]
    return mzs, ints, specix


def generate_mz_bin_edges(imsDataset, n_spectra=100, max_width = 0.1, max_ups=1):
    spec = get_some_real_spectra(imsDataset, n_spectra)
    edges = get_edges(spec[0], spec[2], max_width=max_width, max_ups=max_ups)
    return edges


def generate_mz_axis_from_imzml(imzml, step_size=100):
    spec = get_some_real_spectra(imzml, step_size)
    edges = get_edges(spec[0], spec[2])
    mz_axis = []
    for ii in range(1, len(edges)):
        mz = np.mean(spec[0][get_peak_in_bin_idx(spec[0], edges[ii-1], edges[ii])])
        if np.isnan(mz):
            mz =  np.mean([edges[ii-1], edges[ii]])
        ppm = 0.5 * 1e6 * (edges[ii] - edges[ii-1]) / mz
        mz_axis.append((float(mz), float(ppm)))
    return mz_axis