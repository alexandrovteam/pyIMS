import argparse
from pyMSpec.convert import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert data to \n supported input formats: \n * Bruker peaks.sqlite\n * profile imzML")
    parser.add_argument('input', type=str, help="input data path")
    parser.add_argument('output', type=str, help="output filename (will be a centroided imzml file)")

    args = parser.parse_args()
    args.input
    if args.output.lower().endswith('_centroid.imzml'):
        data_in.centroid_imzml(args.input, args.output)

