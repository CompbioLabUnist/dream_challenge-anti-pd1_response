"""
step05: make t-SNE with Synapse data
"""
import argparse
import matplotlib
import matplotlib.pyplot
import pandas
import seaborn
import sklearn.manifold
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output PNG file")
    parser.add_argument("--cpus", type=int, default=1, help="CPUS to use")

    args = parser.parse_args()

    if not args.output.endswith(".png"):
        raise ValueError("Output must end with .PNG!!")
    elif args.cpus < 1:
        raise ValueError("CPUS must be greater than zero!!")

    data: pandas.DataFrame = step00.read_pickle(args.input)
    data.dropna(axis="index", how="any", inplace=True)
    print(data)

    used_columns = list(filter(lambda x: not x.startswith("Clinical_"), list(data.columns)))
    tsne_data = pandas.DataFrame(sklearn.manifold.TSNE(n_components=2, init="pca", random_state=0, method="exact", n_jobs=args.cpus).fit_transform(data[used_columns]), columns=["TSNE1", "TSNE2"])

    for column in tsne_data.columns:
        tsne_data[column] = sklearn.preprocessing.scale(tsne_data[column])

    tsne_data["TMB"] = list(data["Clinical_TMB"])
    print(tsne_data)

    seaborn.set(context="poster", style="whitegrid")
    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
    seaborn.scatterplot(data=tsne_data, x="TSNE1", y="TSNE2", hue="TMB", size="TMB", ax=ax, legend="brief")
    fig.savefig(args.output)
    matplotlib.pyplot.close(fig)
