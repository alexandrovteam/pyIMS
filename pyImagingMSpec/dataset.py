import os
import numpy as np
import bisect
from pyImagingMSpec.ion_datacube import ion_datacube
from pyMSpec import instrument

def imsDataset(filename):
    def clean_ext(ext):
        return ext.lower().strip('.')
    KNOWN_EXT = {'imzml': ImzmlDataset,}  # 'imzb':imzbDataset, 'sl': scilsLabDataset
    [root, ext] = os.path.splitext(filename)
    if clean_ext(ext) in KNOWN_EXT:
        return KNOWN_EXT[clean_ext(ext)](filename)
    else:
        print(KNOWN_EXT)
        raise IOError('file extention {} not known'.format(clean_ext(ext)))


class BaseDataset(object):
    " base class for ims datasets. should not be directly instantiated"

    def __init__(self, filename):
        self.filename = filename
        self.coordinates = []
        self.histogram_mz_axis = {}
        self.step_size = [] #pixel dimension
        self.instrument = None

    def get_spectrum(self):
        raise NotImplementedError

    def get_image(self):
        raise NotImplementedError

    def rebin(self, n_spectra=100, max_width = 0.1, max_ups=1):
        """
           Returns m/z bins formatted as a Pandas dataframe with the following columns:
            * left, right - bin boundaries;
            * count - number of peaks in the bin;
            * mean - average m/z;
            * sd - standard deviation of m/z;
            * intensity - total intensity.
        """
        from rebinning import generate_mz_bin_edges
        self.rebin_info = generate_mz_bin_edges(self, n_spectra=n_spectra, max_width=max_width, max_ups=max_ups)

    def generate_histogram_axis(self, ppm=1.):
        print 'generating histogram axis for ppm {} [{}-{}]'.format(ppm, self.mz_min, self.mz_max)
        assert self.mz_min > 0
        ppm_mult = ppm * 1e-6
        mz_current = self.mz_min
        mz_list = [mz_current, ]
        while mz_current <= self.mz_max:
            mz_current = mz_current + mz_current * ppm_mult
            mz_list.append(mz_current)
        self.histogram_mz_axis[ppm] = np.asarray(mz_list)

    def get_histogram_axis(self, ppm=1.):
        try:
            mz_axis = self.histogram_mz_axis[ppm]
        except KeyError as e:
            self.generate_histogram_axis(ppm=ppm)
        return self.histogram_mz_axis[ppm]

    def generate_summary_spectrum(self, summary_type='mean', ppm=1., hist_axis=[]):
        if hist_axis == []:
            hist_axis = self.get_histogram_axis(ppm=ppm)
        # calcualte mean along some m/z axis
        mean_spec = np.zeros(np.shape(hist_axis))
        for ix in range(len(self.coordinates)):
            spec = self.get_spectrum(ix)
            for ii in range(0, len(hist_axis) - 1):
                mz_upper = hist_axis[ii + 1]
                mz_lower = hist_axis[ii]
                idx_left = bisect.bisect_left(spec[0], mz_lower)
                idx_right = bisect.bisect_right(spec[0], mz_upper)
                # slice list for code clarity
                if summary_type == 'mean':
                    count_vect = spec[1][idx_left:idx_right]
                    mean_spec[ii] += np.sum(count_vect)
                elif summary_type == 'freq':
                    if not idx_left == idx_right:
                        mean_spec[ii] += 1
                else:
                    raise ValueError('Summary type not recognised; {}'.format(summary_type))
        if summary_type == 'mean':
            mean_spec = mean_spec / len(self.index_list)
        elif summary_type == 'freq':
            mean_spec = mean_spec / len(self.index_list)
        return hist_axis, mean_spec

    def empty_datacube(self):
        data_out = ion_datacube()
        # add precomputed pixel indices
        data_out.coords = self.coordinates
        data_out.pixel_indices = self.cube_pixel_indices
        data_out.nRows = self.cube_n_row
        data_out.nColumns = self.cube_n_col
        return data_out


    def set_instrument_type(self, instrument_type, resolving_power, at_mz):
        self.instrument = getattr(instrument, instrument_type)(at_mz=at_mz, resolving_power=resolving_power)

    def align_peaks(self):
        if not self.instrument:
            raise IOError("Instrument type not set, call instrument.set_instrument_type()")
        raise NotImplementedError

    def tic(self):
        raise NotImplementedError


class ImzmlDataset(BaseDataset):
    def __init__(self, filename):
        from pyimzml.ImzMLParser import ImzMLParser
        super(ImzmlDataset, self).__init__(filename)
        self.imzml = ImzMLParser(filename)
        self.coordinates = np.asarray(self.imzml.coordinates)
        self.step_size = [1,1,1] #fixme get pixel size from header data

    def get_spectrum(self, ix):
        mzs, counts = self.imzml.getspectrum(ix)
        return [np.asarray(mzs), np.asarray(counts)] #todo return MassSpectrum

    def get_image(self, mz, tol):
        im = self.imzml.getionimage(mz, tol)
        return im

class InMemoryDataset(BaseDataset):
    def __init__(self, filename, min_mz=0, max_mz=np.inf, min_int=0):
        super(InMemoryDataset, self).__init__(filename)
        outOfMemoryDataset = imsDataset(filename)
        self.load_file(outOfMemoryDataset, min_mz, max_mz, min_int)

    def load_file(self, outOfMemoryDataset, min_mz, max_mz, min_int, index_range=[], spectrum_type='centroids'):
        # parse file to get required parameters
        # can use thin hdf5 wrapper for getting data from file
        self.file_dir, self.filename = os.path.split(outOfMemoryDataset.filename)
        self.filename, self.file_type = os.path.splitext(self.filename)
        self.coordinates = outOfMemoryDataset.coordinates
        step_size = outOfMemoryDataset.step_size
        cube = ion_datacube(step_size=step_size)
        cube.add_coords(self.coordinates)
        self.cube_pixel_indices = cube.pixel_indices
        self.cube_n_row, self.cube_n_col = cube.nRows, cube.nColumns
        self.spectrum_type = spectrum_type  # fixme this should be read from the base file during get_spectrum?
        # load data into memory
        self.mz_list = []
        self.count_list = []
        self.idx_list = []
        for ii in range(len(self.coordinates)):
            # load spectrum, keep values gt0 (shouldn't be here anyway)
            mzs, counts = outOfMemoryDataset.get_spectrum(ii)
            if len(mzs) != len(counts):
                raise TypeError('length of mzs ({}) not equal to counts ({})'.format(len(mzs), len(counts)))
            # Enforce data limits
            valid = np.where((mzs > min_mz) & (mzs < max_mz) & (counts > min_int))
            counts = counts[valid]
            mzs = mzs[valid]
            # update min/max
            # append ever-growing lists (should probably be preallocated or piped to disk and re-loaded)
            self.mz_list.append(mzs)
            self.count_list.append(counts)
            self.idx_list.append(np.ones(len(mzs), dtype=int) * ii)
        print('loaded spectra')
        self.mz_list = np.concatenate(self.mz_list)
        self.count_list = np.concatenate(self.count_list)
        self.idx_list = np.concatenate(self.idx_list)
        # sort by mz for fast image formation
        mz_order = np.argsort(self.mz_list)
        self.mz_list = self.mz_list[mz_order]
        self.count_list = self.count_list[mz_order]
        self.idx_list = self.idx_list[mz_order]
        self.mz_min = self.mz_list[0]
        self.mz_max = self.mz_list[-1]
        # split binary searches into two stages for better locality
        self.window_size = 1024
        self.mz_sublist = self.mz_list[::self.window_size].copy()
        print('file loaded')
        self.outOfMemoryDataset = outOfMemoryDataset
        self.tic = np.bincount(self.idx_list, weights=self.count_list, minlength=self.idx_list.max() + 1)


    def get_spectrum(self, index):
        #mzs = []
        #counts = []
        #for ix in self.idx_list:
        #    if ix == index:
        #        mzs.append(self.mz_list[ix])
        #        counts.append(self.count_list[ix])
        #return np.asarray(mzs), np.asarray(counts)
        return self.outOfMemoryDataset.get_spectrum(index)

    def get_ion_image(self, mzs, tols, tol_type='ppm'):
        try:
            len(mzs)
        except TypeError as e:
            mzs = [mzs, ]
        try:
            len(tols)
        except TypeError as e:
            tols = [tols, ]
        mzs = np.asarray(mzs)
        tols = np.asarray(tols)
        data_out = self.empty_datacube()
        if len(tols) == 1:
            tols = tols * np.ones(np.shape(mzs))
        if type(mzs) not in (np.ndarray, list):
            mzs = np.asarray([mzs, ])
        if tol_type == 'ppm':
            tols = tols * mzs / 1e6  # to m/z
        # Fast search for insertion point of mz in self.mz_list
        # First stage is looking for windows using the sublist
        idx_left = np.searchsorted(self.mz_sublist, mzs - tols, 'l')
        idx_right = np.searchsorted(self.mz_sublist, mzs + tols, 'r')
        for mz, tol, il, ir in zip(mzs, tols, idx_left, idx_right):
            l = max(il - 1, 0) * self.window_size
            r = ir * self.window_size
            # Second stage is binary search within the windows
            il = l + np.searchsorted(self.mz_list[l:r], mz - tol, 'l')
            ir = l + np.searchsorted(self.mz_list[l:r], mz + tol, 'r')
            # slice list for code clarity
            mz_vect = self.mz_list[il:ir]
            idx_vect = self.idx_list[il:ir]
            count_vect = self.count_list[il:ir]
            # bin vectors
            ion_vect = np.bincount(idx_vect, weights=count_vect, minlength=self.idx_list.max() + 1)
            data_out.add_xic(ion_vect, [mz], [tol])
        return data_out

    def generate_summary_spectrum(self, summary_type='mean', ppm=1., hist_axis=[]):
        if hist_axis == []:
            hist_axis = self.get_histogram_axis(ppm=ppm)
        # calcualte mean along some m/z axis
        mean_spec = np.zeros(np.shape(hist_axis))
        for ii in range(0, len(hist_axis) - 1):
            mz_upper = hist_axis[ii + 1]
            mz_lower = hist_axis[ii]
            idx_left = bisect.bisect_left(self.mz_list, mz_lower)
            idx_right = bisect.bisect_right(self.mz_list, mz_upper)
            # slice list for code clarity
            count_vect = self.count_list[idx_left:idx_right]
            if summary_type == 'mean':
                count_vect = self.count_list[idx_left:idx_right]
                mean_spec[ii] = np.sum(count_vect)
            elif summary_type == 'freq':
                idx_vect = self.idx_list[idx_left:idx_right]
                mean_spec[ii] = float(len(np.unique(idx_vect)))
            else:
                raise ValueError('Summary type not recognised; {}'.format(summary_type))
        if summary_type == 'mean':
            mean_spec = mean_spec / len(self.coordinates)
        elif summary_type == 'freq':
            mean_spec = mean_spec / len(self.coordinates)
        return hist_axis, mean_spec

    def get_summary_image(self, summary_func='tic'):
        if summary_func not in ['tic', 'mic']: raise KeyError("requested type not in 'tic' mic'")
        data_out = self.empty_datacube()
        data_out.add_xic(np.asarray(getattr(self, summary_func)), [0], [0])
        return data_out

