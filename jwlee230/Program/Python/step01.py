"""
step01.py: Read data from TIDEpy
"""
import argparse
import pandas
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input PKL file", type=str)
    parser.add_argument("output", help="Output tar.gz file", type=str)

    args = parser.parse_args()

    data = pandas.read_pickle(args.input)
    step00.make_pickle(args.output, data)
