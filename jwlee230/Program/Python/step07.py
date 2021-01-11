"""
step07.py: t-SNE with imputed data
"""
import argparse
import tarfile
import typing
import matplotlib
import matplotlib.pyplot
import pandas
import seaborn
import sklearn.manifold
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output TAR file")
    parser.add_argument("--cpus", type=int, default=1, help="CPUS to use")

    args = parser.parse_args()

    if not args.output.endswith(".tar"):
        raise ValueError("Output must end with .TAR!!")
    elif args.cpus < 1:
        raise ValueError("CPUS must be greater than zero!!")

    tar_files: typing.List[str] = list()

    data = step00.read_pickle(args.input)
    float_data = data.select_dtypes(exclude=["object"])
    object_data = data.select_dtypes(include=["object"])

    tsne_data = pandas.DataFrame(sklearn.manifold.TSNE(n_components=2, init="pca", random_state=0, method="exact", n_jobs=args.cpus).fit_transform(float_data), columns=["TSNE1", "TSNE2"])

    for column in tsne_data.columns:
        tsne_data[column] = sklearn.preprocessing.scale(tsne_data[column])

    tsne_data = pandas.concat([object_data, tsne_data], axis="columns", join="inner", verify_integrity=True)
    print(tsne_data)

    seaborn.set(context="poster", style="whitegrid")
    for feature in sorted(object_data.columns):
        print(feature)
        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))
        seaborn.scatterplot(data=tsne_data, x="TSNE1", y="TSNE2", hue=feature, style=feature, ax=ax, legend="full")
        tar_files.append(feature + ".png")
        fig.savefig(tar_files[-1])
        matplotlib.pyplot.close(fig)

    # Save data
    with tarfile.open(args.output, "w") as tar:
        for file_name in tar_files:
            tar.add(file_name, arcname=file_name)
