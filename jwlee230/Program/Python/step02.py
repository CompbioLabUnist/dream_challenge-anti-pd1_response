"""
step02.py:
"""
import argparse
import math
import typing
import numpy
import pandas
import sklearn.ensemble
import sklearn.model_selection
import sklearn.neighbors


def can_convert_to_float(value: typing.Any) -> bool:
    """
    can_convert_to_float: determines whether the value cant be converted to a float
    """
    try:
        float(value)
    except ValueError:
        return False

    if math.isnan(float(value)):
        return False
    else:
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("expression", type=str, help="Expression TSV file")
    parser.add_argument("TPM", type=str, help="TPM TSV file")
    parser.add_argument("clinical", type=str, help="Clinical TSV file")
    parser.add_argument("output", type=str, help="Output TAR.gz file")
    parser.add_argument("--cpus", type=int, default=1, help="CPU number to use")

    args = parser.parse_args()

    if not args.expression.endswith(".tsv"):
        raise ValueError("EXPRESSION must end with .tsv!!")
    elif not args.TPM.endswith(".tsv"):
        raise ValueError("TPM must end with .tsv!!")
    elif not args.clinical.endswith(".tsv"):
        raise ValueError("CLINICAL must end with .tsv!!")
    elif args.cpus < 1:
        raise ValueError("CPU must be greater that zero!!")

    # read data
    expression_data = pandas.read_csv(args.expression, sep="\t")
    expression_data.set_index("b.df", inplace=True)
    expression_data = expression_data.T
    expression_data.sort_index(axis="index", inplace=True)
    expression_data.columns = list(map(lambda x: "Expression_" + x, list(expression_data.columns)))

    TPM_data = pandas.read_csv(args.TPM, sep="\t")
    TPM_data.set_index("V1", inplace=True)
    TPM_data = TPM_data.T
    TPM_data.sort_index(axis="index", inplace=True)
    TPM_data.columns = list(map(lambda x: "TPM_" + x, list(TPM_data.columns)))

    clinical_data = pandas.read_csv(args.clinical, sep="\t")
    clinical_data.set_index("WTS_ID", inplace=True)
    clinical_data.sort_index(axis="index", inplace=True)

    # intersect
    intersect_index = set(expression_data.index) & set(TPM_data.index) & set(clinical_data.index)
    expression_data = expression_data.loc[list(map(lambda x: x in intersect_index, list(expression_data.index)))]
    TPM_data = TPM_data.loc[list(map(lambda x: x in intersect_index, list(TPM_data.index)))]
    clinical_data = clinical_data.loc[list(map(lambda x: x in intersect_index, list(clinical_data.index)))]

    print(expression_data)
    print(TPM_data)
    print(clinical_data)

    # make train data
    known_index = list(map(can_convert_to_float, clinical_data["TMB"]))
    known_data = pandas.concat([expression_data, TPM_data], axis="columns", verify_integrity=True)
    known_data = known_data.loc[(known_index)]
    known_answer = clinical_data.loc[(known_index)]["TMB"]

    k_fold = sklearn.model_selection.KFold(n_splits=5)

    # Random Forest
    randomforest_regressor = sklearn.ensemble.RandomForestRegressor(max_features=None, random_state=0, n_jobs=args.cpus, bootstrap=False)
    randomforest_scores = list()
    for train_index, test_index in k_fold.split(known_data):
        x_train, x_test = known_data.iloc[train_index], known_data.iloc[test_index]
        y_train, y_test = list(map(float, known_answer.iloc[train_index])), list(map(float, known_answer.iloc[test_index]))

        randomforest_regressor.fit(x_train, y_train)
        randomforest_scores.append(randomforest_regressor.score(x_test, y_test))
    print("RandomForest:", numpy.mean(randomforest_scores), randomforest_scores)

    # K-neighbors
    kneighbors_regressor = sklearn.neighbors.KNeighborsRegressor(algorithm="brute", n_jobs=args.cpus)
    kneighbors_scores = list()
    for train_index, test_index in k_fold.split(known_data):
        x_train, x_test = known_data.iloc[train_index], known_data.iloc[test_index]
        y_train, y_test = list(map(float, known_answer.iloc[train_index])), list(map(float, known_answer.iloc[test_index]))

        kneighbors_regressor.fit(x_train, y_train)
        kneighbors_scores.append(kneighbors_regressor.score(x_test, y_test))
    print("K-Neighbors:", numpy.mean(kneighbors_scores), kneighbors_scores)
