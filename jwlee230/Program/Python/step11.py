"""
step11.py: Build Random-Forest model
"""
import argparse
import sklearn.ensemble
import pandas
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("clinical", type=str, help="Clinical data CSV file")
    parser.add_argument("expression", type=str, help="Expression CSV file(s)", nargs="+")
    parser.add_argument("output", type=str, help="Output TAR.gz file")
    parser.add_argument("--cpus", type=int, default=1, help="CPUS to use")

    args = parser.parse_args()

    if args.cpus < 1:
        raise ValueError("CPUS must be greater than zero!!")
    elif not args.clinical.endswith(".csv"):
        raise ValueError("Clinical must end with CSV!!")

    our_data = step00.read_pickle(args.input)
    our_data.dropna(axis="index", inplace=True)
    our_data.info()

    # clinical_data
    clinical_data = pandas.read_csv(args.clinical)
    clinical_data.set_index("patientID", inplace=True)
    clinical_data["ECOGPS"] = list(map(lambda x: float(x) if step00.can_convert_to_float(x) else None, list(clinical_data["ECOGPS"])))
    clinical_data["TMB"] = list(map(lambda x: float(x) if step00.can_convert_to_float(x) else None, list(clinical_data["TMB"])))
    clinical_data.columns = list(map(lambda x: "Clinical_" + x, list(clinical_data.columns)))
    clinical_data.sort_index(axis="index", inplace=True)

    data_list = [clinical_data]
    for i, f in enumerate(args.expression):
        tmp_data = pandas.read_csv(f)
        tmp_data.set_index(list(tmp_data.columns)[0], inplace=True)
        tmp_data = tmp_data.T
        tmp_data.columns = list(map(lambda x: str(i) + "_" + x, list(tmp_data.columns)))
        tmp_data.sort_index(axis="index", inplace=True)
        data_list.append(tmp_data)

    given_data = pandas.concat(data_list, axis="columns", join="inner", verify_integrity=True)
    given_data = given_data.select_dtypes(exclude="object")

    selected_columns = sorted(set(map(lambda x: x.split("_")[-1], list(our_data.columns))) & set(map(lambda x: x.split("_")[-1], list(given_data.columns))))
    total = len(selected_columns)

    wanted_column = args.output.split("/")[-1].split(".")[0]
    train_answer = list(our_data["Clinical_" + wanted_column])
    train_data = pandas.DataFrame()
    for i, gene in enumerate(selected_columns):
        print(i, total, gene)
        train_data[gene] = our_data[list(filter(lambda x: x.endswith(gene), list(our_data.columns)))].sum(axis=1)
    train_data.info()

    classifier = sklearn.ensemble.RandomForestClassifier(max_features=None, n_jobs=args.cpus, random_state=0)
    classifier.fit(train_data, train_answer)
    step00.make_pickle(args.output, classifier)
