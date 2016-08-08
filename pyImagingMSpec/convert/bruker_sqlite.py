import sqlite3
import numpy as np
import argparse
from skimage.filters import threshold_otsu

from pyimzml.ImzMLWriter import ImzMLWriter

class Spectrum(object):
    def __init__(self, row):
        self.mzs = np.frombuffer(row['PeakMzValues'], dtype=np.float64)
        self.intensities = np.frombuffer(row['PeakIntensityValues'], dtype=np.float32)
        self.fwhms = np.frombuffer(row['PeakFwhmValues'], dtype=np.float32)
        self.x = row['XIndexPos']
        self.y = row['YIndexPos']


def estimateThreshold(cursor):
    sp = Spectrum(cursor.execute("select * from Spectra").fetchone())
    coeff = sp.fwhms / sp.mzs**2  # fwhm is proportional to mz^2 for FTICR
    thr = threshold_otsu(coeff)  # find a point to split the histogram
    if float((coeff > thr).sum()) / len(coeff) < 0.2:
        thr = 0  # there's nothing to fix, apparently
    return thr


def centroid_imzml(sqlite_fn, imzml_fn):
    with ImzMLWriter(imzml_fn) as w:
        peaks = sqlite3.connect(sqlite_fn)
        peaks.row_factory = sqlite3.Row
        c = peaks.cursor()
        i = 0
        count = int(c.execute("select count(*) from Spectra").fetchone()[0])
        threshold = estimateThreshold(c)
        for sp in map(Spectrum, c.execute("select * from Spectra")):
            real_peaks = sp.fwhms / sp.mzs**2 > threshold
            mzs = sp.mzs[real_peaks]
            intensities = sp.intensities[real_peaks]
            w.addSpectrum(mzs, intensities, (sp.x, sp.y))
            i += 1
            if i % 1000 == 0:
                print "{}% complete".format(float(i) / count * 100.0)
        print "done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert centroids from Bruker sqlite to imzML")
    parser.add_argument('input', type=str, help="peaks.sqlite file")
    parser.add_argument('output', type=str, help="output filename (will guess conversion from extension)")

    args = parser.parse_args()
    centroid_imzml(args.input, args.output)
