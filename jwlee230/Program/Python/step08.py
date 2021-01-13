"""
step08.py: get R2 scores
"""
import argparse
import multiprocessing
import pandas
import scipy.stats
import step00


def r2_score(x, y):
    """
    r2_score: get R2 score between x and y
    """
    return scipy.stats.linregress(x, y)[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output TAR.gz file")
    parser.add_argument("--cpus", type=int, default=1, help="CPUS to use")

    args = parser.parse_args()

    if args.cpus < 1:
        raise ValueError("CPUS must be greater than zero!!")

    wanted_column = args.output.split("/")[-1].split(".")[0]

    # read input data
    data: pandas.DataFrame = step00.read_pickle(args.input)
    print(data)
    data = data.select_dtypes(exclude="object")
    data.dropna(axis="index", inplace=True)

    y_data = list(data["Clinical_" + wanted_column])
    data = data[list(filter(lambda x: not x.startswith("Clinical_"), list(data.columns)))]
    data = data[[i for i in data if len(set(data[i])) > 1]]

    with multiprocessing.Pool(args.cpus) as pool:
        r2_score_list = sorted(zip(pool.starmap(r2_score, [(list(data[column]), y_data) for column in list(data.columns)]), list(data.columns)), reverse=True)

    print(r2_score_list[:10])

    data = data[list(map(lambda x: x[1], list(filter(lambda x: x[0] > 0, r2_score_list))))]
    data["Clinical_" + wanted_column] = y_data
    print(data)

    step00.make_pickle(args.output, data)
