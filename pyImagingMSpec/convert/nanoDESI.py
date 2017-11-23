from __future__ import print_function
import argparse
from pyimzml.ImzMLWriter import ImzMLWriter
import pymzml
import json
import os
import numpy as np


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])):
        return idx - 1, array[idx - 1]
    else:
        return idx, array[idx]


def fit_grid_histogram_homogeneity(coords):
    """
    Measure of homogeneity to estimate optimal x spacing for dataset
    Assumes approximately equal number of pixels per row
    coords: dictionary of coordinates: each key is a row containing a list of (x,y) micrometer positions
    """
    # get all x values
    # estimate possible range of solutions for x grid and calculate the bin filling
    # define x grid and y grid coordinaes
    x_values = [cc[0] for c in coords for cc in coords[c]]
    range_guess = list(map(lambda x: int(x * np.max([len(coords[co]) for co in coords])), [0.8, 1.1]))
    m = []
    for ii in np.linspace(range_guess[0], range_guess[1], range_guess[1] - range_guess[0]):
        hist = np.histogram(x_values, np.linspace(0, np.max(x_values), int(ii)))[0]
        m.append((ii, np.max(hist), np.min(hist), np.std(hist)))
    p3 = np.poly1d(np.polyfit([_m[0] for _m in m], [float(_m[1]) / _m[2] for _m in m], 3))([_m[0] for _m in m])
    # assume the best bin filling is when all are approximatley equally full
    min_spacing = int(np.asarray([_m[0] for _m in m])[np.argmin(p3)])

    _, x_bins = np.histogram(x_values, min_spacing)
    x_coords = [np.mean([x_bins[ii], x_bins[ii + 1]]) for ii in range(0, len(x_bins) - 1)]
    y_coords = [coords[c][0][1] for c in coords]
    return x_coords, y_coords


class DESIDataset():
    def __init__(self, json_fn, grid_fit='nearest'):
        self._parse_json_fn(json_fn)
        self._get_coordinates()
        self._generate_grid()

    def _parse_json_fn(self, json_fn):
        """
        loads config file and find mzml files
        param: json_fn
            full path to json config file
        """
        self._parse_file_paths(json_fn)
        self._config = json.load(open(json_fn))
        self._mzml_fns = [os.path.join(self._dr, fn) for fn in os.listdir(self._dr) if
                          all([fn.startswith(self._ds_name), fn.endswith('.mzML')])]
        self._load_json(json_fn)

    def _parse_file_paths(self, json_fn):
        j = json.load(open(json_fn))
        if 'data_folder' in j:
            self._dr = j['data_folder']
        else:
            self._dr = os.path.split(json_fn)[0]
        if 'data_name' in j:
            self._ds_name = j['ds_name']
        else:
            self._ds_name = os.path.splitext(os.path.split(json_fn)[1])[0]

    def _load_json(self, json_fn):
        # scan time will be in seconds
        self._config = json.load(open(json_fn))
        # distance will be in um
        if not self._config["stage_parameters"]["x_velocity_units"] == 'micrometers per second':
            raise ValueError("x velocity should be in 'micrometers per second' ")

        if not self._config["stage_parameters"]["y_spacing_units"] == 'micrometers':
            raise ValueError("y spacing should be in 'micrometers' ")

    def _get_coordinates(self):
        """
        grab the scan time from the mzml files and turn it into x distance
        grab the scan row number from the filename and turn it into y distance
        """
        coords = {}
        for mzml_fn in self._mzml_fns:
            print('parsing coordinates from {}'.format(mzml_fn))
            ii = int(mzml_fn.split("_")[-1][:-5])
            coords[ii] = []
            y_coord = ii * float(self._config["stage_parameters"]["y_spacing"])
            mzml = pymzml.run.Reader(mzml_fn)
            for spectrum in mzml:
                x_coord = spectrum.scan_time * 60. * float(self._config["stage_parameters"]["x_velocity"])
                coords[ii].append((x_coord, y_coord))
        self._coords = coords

    def _generate_grid(self):
        """
        fit the best grid to the x data points
        assumes that all rows start at x=0
        assumes roughly equal number of pixels per row
        """
        x_spacing, y_spacing = fit_grid_histogram_homogeneity(self._coords)
        self._x_coords = x_spacing
        self._y_coords = y_spacing

    def parse_imzml_fn(self, imzml_fn):
        if not imzml_fn:
            imzml_fn = os.path.join(self._dr, self._ds_name + ".imzML")
        return imzml_fn

    def export_imzml(self, imzml_fn=None):
        """
        :param imzml_fn:
        string containing path to write imzml file to, 
        if not supplied the default (same name, same directory as mzml) will be used
        """
        imzml_fn = self.parse_imzml_fn(imzml_fn)

        with ImzMLWriter(imzml_fn) as imzml:
            for mzml_fn in self._mzml_fns:
                print("exporting file: {}".format(mzml_fn))
                ii = int(mzml_fn.split("_")[-1][:-5])
                y_coord = ii * float(self._config["stage_parameters"]["y_spacing"])
                y_ix = ii
                mzml = pymzml.run.Reader(mzml_fn)
                for spectrum in mzml:
                    x_scan = spectrum.scan_time * 60. * float(self._config["stage_parameters"]["x_velocity"])
                    x_ix, x_coord = find_nearest(self._x_coords, x_scan)
                    peaks = np.asarray(spectrum.peaks('centroided'))
                    imzml.addSpectrum(mzs=peaks[:, 0],
                                      intensities=peaks[:, 1],
                                      coords=(x_ix, y_ix, 1),
                                      userParams=[
                                          {'name': 'xCoord', 'value': str(x_coord)},
                                          {'name': 'yCoord', 'value': str(y_coord)}
                                      ]
                                      )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert centroids from nanoDESI/Orbitrap to imzML")
    parser.add_argument('json_fn', type=str, help="xml file")
    parser.add_argument('imzml_fn', type=str, help="output file")

    args = parser.parse_args()
    desi = DESIDataset(args.json_fn)
    desi.export_imzml(args.imzml_fn)
