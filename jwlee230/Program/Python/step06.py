"""
step06.py: Impute missing features
"""
import argparse
import pandas
import sklearn.impute
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output TAR.gz file")

    args = parser.parse_args()

    raw_data = step00.read_pickle(args.input)
    print(raw_data)

    object_data = raw_data.select_dtypes(include=["object"])
    raw_data = raw_data.select_dtypes(exclude=["object"])

    imputer = sklearn.impute.KNNImputer(weights="distance")
    imputed_data = pandas.DataFrame(data=imputer.fit_transform(raw_data), index=raw_data.index, columns=raw_data.columns, dtype=float, copy=True)
    print(imputed_data)

    data = pandas.concat([object_data, imputed_data], axis="columns", join="inner", verify_integrity=True)
    print(data)
    step00.make_pickle(args.output, data)
