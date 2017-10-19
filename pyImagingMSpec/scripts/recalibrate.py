import argparse
import numpy as np
from pyimzml.ImzMLWriter import ImzMLWriter
from pyImagingMSpec.inMemoryIMS import inMemoryIMS
from scipy.optimize import least_squares
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from scipy.signal import medfilt2d

def fun(x, t, y):
    e = np.polynomial.polynomial.polyval(t,x) - y
    #e = x[0]+x[1]*t+x[2]*t**2 - y
    return e


def find_nearest(v, t):
    """
    v: vector of values
    t: target value
    """
    v=np.asarray(v)
    ix = np.searchsorted(v,t)
    if any([ix == len(v), ix == 0]):
        return ix
    return ix-1+np.argmin(np.abs(v[[ix-1,ix]]-t))


def find_max_of_nearest(v, i, t, n):
    """
    return the most intense of nearby peaks
    """
    assert(v.shape==i.shape)
    v=np.asarray(v)
    ix = np.searchsorted(v,t)
    ix_l = np.max([0, ix-n])
    ix_u = np.min([v.shape[0], ix+n])
    _v = v[ix_l:ix_u]
    _i = i[ix_l:ix_u]
    ix = np.arange(ix_l, ix_u, dtype=int)
    return ix[np.argmax(_i)] if _v.shape[0]!=0 else None


def find_max_in_range(v, i, t, r):
    """
    v: vector of values
    i: vector of intensities
    t: target value
    r: range to consider
    """
    assert(v.shape==i.shape)
    ix_l = np.searchsorted(v, t-r)
    ix_u = np.searchsorted(v, t+r)
    _v = v[ix_l:ix_u]
    _i = i[ix_l:ix_u]
    tf = (_v>=t-r) & (_v<t+r)
    _v = _v[tf]
    _i = _i[tf]
    ix = np.arange(ix_l, ix_u)[tf]
    return ix[np.argmax(_i)] if _v.shape[0]!=0 else None


def get_delta(ref_mzs, mzs, intensities, max_delta_ppm=20.):
    delta = []
    for mz in ref_mzs:
        delta_mz = 1e-6*mz*max_delta_ppm
        # ix = find_max_of_nearest(mzs, intensities, mz, n=3)
        ix = find_max_in_range(mzs, intensities, mz, delta_mz)
        if ix:
            delta.append(1e6*(mz - mzs[ix])/mz)
            #delta.append(mz-mzs[ix])
        else:
            delta.append(np.nan)
    return np.asarray(delta)


def generate_data(t, x, noise=0, n_outliers=0, random_state=30):
#    y = x[0]+x[1]*t+x[2]*t**2
    y = np.polynomial.polynomial.polyval(t,x)
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *= 35
    return y + error


def fit_spectrum(mzs, intensities, ref_mzs, ref_pct, max_delta_ppm, mz_min, mz_max, x0=[1, 1, 1],
                 weight_by_occurance=True):
    delta = get_delta(ref_mzs, mzs, intensities, max_delta_ppm=max_delta_ppm)
    _x = [mz_min, mz_max]
    _y = [0, 0]
    for ref_mz, ref_pct, dt in zip(ref_mzs, ref_pct.astype('int'), delta):
        if not weight_by_occurance:
            ref_pct = 1
        for ii in np.arange(ref_pct):
            _x.append(ref_mz)
            _y.append(dt)
    _x, _y = map(np.asarray, (_x, _y))
    _r = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(_x[np.abs(_y) < 500], _y[np.abs(_y) < 500]))
    return _r


def recalibrate_spectrum(mzs, r):
    est_error = generate_data(mzs, r)
    mzs = mzs + 1e-6 * est_error * mzs
    return mzs


def calculate_mass_deviation(input_filename, known_peaks):
    ims_dataset = inMemoryIMS(input_filename)
    delta = np.zeros([len(ims_dataset.coords), len(known_peaks)])
    for ii in range(len(ims_dataset.coords)):
        spec = ims_dataset.get_spectrum(ii).get_spectrum(source='centroids')
        delta[ii,:] = get_delta(known_peaks, spec[0], spec[1])
    return delta


def poly_from_deltas(known_mzs, delta, max_ppm=100, polyorder=3):
    f = lambda x: np.median(x[np.abs(x) < max_ppm])
    median_delta = [f(delta[:, ii]) for ii in range(len(known_mzs))]
    z = np.polyfit(known_mzs, median_delta, polyorder)
    p = np.poly1d(z)
    return p


def do_recalibration(input_filename, output_filename, p):
    ims_dataset = inMemoryIMS(input_filename)
    with ImzMLWriter(output_filename) as file_out:
        for ii in range(len(ims_dataset.coords)):
            spec = ims_dataset.get_spectrum(ii).get_spectrum(source='centroids')
            mzs = spec[0]
            mzs_recal = [m-(1e-6)*m*p(m) for m in mzs]
            file_out.addSpectrum(mzs_recal, spec[1], ims_dataset.coords[ii,[1,0,2]])


def recalibrate_dataset(input_filename, output_filename, known_peaks):
    deltas = calculate_mass_deviation(input_filename, known_peaks)
    p = poly_from_deltas(known_peaks, deltas)
    do_recalibration(input_filename, output_filename, p)


def fit_dataset(imzml, ref_formula, x0=[1,1,1], max_delta_ppm=3., mz_min=200, mz_max=1000):
    fit = []
    for spec_ix, coords in enumerate(imzml.coordinates):
        if spec_ix%500==0:
            print spec_ix/float(len(imzml.coordinates))
        spec = imzml.getspectrum(index=spec_ix)
        mzs, intensities = np.asarray(spec[0]), np.asarray(spec[1])
        _r = fit_spectrum(mzs, intensities, ref_formula.mz, ref_formula.percent, max_delta_ppm, mz_min, mz_max, x0)
        fit.append(_r.x)
    return fit


def recal(imzml_out_fn, imzml, fit, m=3):
    # Write recalibrated dataset
    # spatial smoothing on recal params
    im3 = []
    for ii in range(len(fit[0])):
        im = np.mean([fit[spec_ix][ii] for spec_ix in range(len(imzml.coordinates))]) + np.zeros((imzml.imzmldict["max count of pixels y"], imzml.imzmldict["max count of pixels x"]))
        for spec_ix, (x,y,z) in enumerate(imzml.coordinates):
            im[y - 1, x - 1] = fit[spec_ix][ii]
        im = medfilt2d(im, m)
        im3.append(im)
    im3 = np.dstack(im3)
    # recal and write
    with ImzMLWriter(imzml_out_fn) as imzml_out:
        for spec_ix, coords in enumerate(imzml.coordinates):
            if spec_ix%500==0:
                print spec_ix/float(len(imzml.coordinates))
            mzs, intensities = imzml.getspectrum(index=spec_ix)
            mzs = np.asarray(mzs)
            mzs = recalibrate_spectrum(mzs, im3[coords[1]-1, coords[0]-1, :])
            imzml_out.addSpectrum(mzs, intensities, coords)

def robust_recalibration(imzml_fn, imzml_fn_r, ref_formula, numpeaks, smoothing, x0=[1, 1]):
    import os
    imzml = ImzMLParser(imzml_fn)
    # calculate fit parameters with varying numbers of peaks
    fit = fit_dataset(imzml, ref_formula, x0=x0, max_delta_ppm=numpeaks)
    # do fit with different spatial smoothing
    recal(imzml_fn_r, imzml, fit, m=smoothing)
    return fit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="recalibrate centroided imaging MS file")
    parser.add_argument('input', type=str, help="input imaging MS file")
    parser.add_argument('output', type=str, help="output filename")
    parser.add_argument('known_peaks', metavar='N', type=float, nargs='+',
                        help='an integer for the accumulator')
    args = parser.parse_args()

    recalibrate_dataset(args.input, args.output, args.known_peaks)
