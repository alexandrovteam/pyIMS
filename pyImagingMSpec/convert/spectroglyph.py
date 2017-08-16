"""
    Datafiles produced by the Spectroglyph MALDI source
    The data consists of an xml file with spatial information and a mzml file of centroided peaks (exported from the .raw file with msconverter)
"""
from __future__ import print_function
import pandas as pd
from xml.etree.ElementTree import ElementTree
import pymzml
from pyimzml.ImzMLWriter import ImzMLWriter
import argparse
import numpy as np

def xml2df(tree, step_size, round=True):
    all_records = []
    image_data = tree.find('{http://www.spectroglyph.com}image')
    info = {_i: float(image_data.attrib[_i]) for _i in image_data.attrib}
    for i, child in enumerate(image_data):
        all_records.append(child.attrib)
    df = pd.DataFrame(all_records)
    df = df.applymap(int)
    #df['x'] = (df['offset'] - df['offset'].min()) * step_size * info['pixelWidth']
    df['x'] = (df['offset'] - df['offset'].min()) / np.median(np.diff(df['offset'].unique()))
    if round == True:
        df['x'] = df['x'].apply(int)
    return df.sort_values(by='microscan'), info

class SpectroglyphFile():
    def __init__(self, mzml_fn, xml_fn, step_size = 1. / 600):
        self.mzml_fn = mzml_fn
        self.xml_fn = xml_fn
        self.step_size = step_size
        self.parseXml()
        df, info = xml2df(self.tree, step_size)
        self.df = df
        self.info =info

    def parseXml(self):
        self.tree = ElementTree()
        self.tree.parse(self.xml_fn)

    def plotpixels(self):
        import matplotlib.pyplot as plt
        self.df.plot(x='x', y='y', kind='scatter')
        plt.axis('equal')
        plt.show()


    def write_imzml(self, imzml_fn):
        mzml = pymzml.run.Reader(self.mzml_fn)
        with ImzMLWriter(imzml_fn, mode="processed") as imzml:
            for ii, spec in enumerate(mzml):
                ix = spec["id"]
                if ix == 'TIC':
                    continue
                row = self.df.loc[self.df['microscan'] == ix]
                if row.empty:
                    continue
                if spec["ms level"] == 1:
                    print('.', end='')
                    imzml.addSpectrum(spec.mz, spec.i, coords=(row['x'].values[0], row['y'].values[0], 1))
                if (ii > 0) & (ii % 1000 == 0):
                    print("{:3.2f}".format(float(ii) / mzml.info['spectrum_count']), end="")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert centroids from Spectroglyph/Orbitrap to imzML")
    parser.add_argument('mzml_fn', type=str, help="mzml file (centroided)")
    parser.add_argument('xml_fn', type=str, help="xml file")
    parser.add_argument('imzml_fn', type=str, help="output file")

    args = parser.parse_args()
    SpectroglyphFile(args.mzml_fn, args.xml_fn).write_imzml(args.imzml_fn)