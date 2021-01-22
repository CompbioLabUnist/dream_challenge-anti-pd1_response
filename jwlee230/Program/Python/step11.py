"""
step11.py: Build Random-Forest model
"""
import argparse
import numpy
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
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

    if not args.clinical.endswith(".csv"):
        raise ValueError("Clinical must end with CSV!!")
    elif args.cpus < 1:
        raise ValueError("CPUS must be greater than zero!!")

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
    given_data.info()

    selected_columns = sorted(set(map(lambda x: x.split("_")[-1], list(our_data.columns))) & set(map(lambda x: x.split("_")[-1], list(given_data.columns))))

    wanted_column = args.output.split("/")[-1].split(".")[0]
    train_data = pandas.DataFrame()
    for i, gene in enumerate(selected_columns):
        train_data[gene] = our_data[list(filter(lambda x: x.endswith(gene), list(our_data.columns)))].sum(axis=1)
    train_data["answer"] = our_data["Clinical_" + wanted_column]
    train_data.info()

    r2_scores = list()
    kfold = sklearn.model_selection.KFold(n_splits=10)
    regressor = sklearn.ensemble.RandomForestRegressor(max_features=None, n_jobs=args.cpus, random_state=0, bootstrap=False, verbose=1)

    for train_index, test_index in kfold.split(train_data):
        x_train, x_test = train_data.iloc[train_index][selected_columns], train_data.iloc[test_index][selected_columns]
        y_train, y_test = train_data.iloc[train_index]["answer"], train_data.iloc[test_index]["answer"]

        regressor.fit(x_train, y_train)
        r2_scores.append(sklearn.metrics.r2_score(y_test, regressor.predict(x_test)))

    print("Mean:", "%.3f" % numpy.mean(r2_scores))
    print("STD:", "%.3f" % numpy.std(r2_scores))

    regressor = sklearn.ensemble.RandomForestRegressor(max_features=None, n_jobs=1, random_state=0, bootstrap=False, verbose=1)
    regressor.fit(train_data[selected_columns], train_data["answer"])
    step00.make_pickle(args.output, {"columns": selected_columns, "regressor": regressor})
