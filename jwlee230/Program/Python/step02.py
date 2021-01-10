"""
step02.py:
"""
import argparse
import math
import typing
import numpy
import pandas
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.svm
import sklearn.tree


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
    clinical_data["TMB"] = list(map

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

    # Mean
    mean = numpy.mean(list(map(float, known_answer)))
    print("Mean:", sklearn.metrics.r2_score(list(map(float, known_answer)), [mean for _ in known_answer]))

    # Random Forest
    RandomForest_regressor = sklearn.ensemble.RandomForestRegressor(max_features=None, random_state=0, n_jobs=args.cpus, bootstrap=False)
    RandomForest_scores = list()
    for train_index, test_index in k_fold.split(known_data):
        x_train, x_test = known_data.iloc[train_index], known_data.iloc[test_index]
        y_train, y_test = list(map(float, known_answer.iloc[train_index])), list(map(float, known_answer.iloc[test_index]))
        RandomForest_regressor.fit(x_train, y_train)
        RandomForest_scores.append(RandomForest_regressor.score(x_test, y_test))
    print("RandomForest:", numpy.mean(RandomForest_scores), RandomForest_scores)

    # K-neighbors
    KNeighbors_regressor = sklearn.neighbors.KNeighborsRegressor(algorithm="brute", weights="distance", n_jobs=args.cpus)
    KNeighbors_scores = list()
    for train_index, test_index in k_fold.split(known_data):
        x_train, x_test = known_data.iloc[train_index], known_data.iloc[test_index]
        y_train, y_test = list(map(float, known_answer.iloc[train_index])), list(map(float, known_answer.iloc[test_index]))
        KNeighbors_regressor.fit(x_train, y_train)
        KNeighbors_scores.append(KNeighbors_regressor.score(x_test, y_test))
    print("K-Neighbors:", numpy.mean(KNeighbors_scores), KNeighbors_scores)

    # Linear SVR
    linearSVR_regressor = sklearn.svm.SVR(kernel="linear", cache_size=400 * 1000)
    linearSVR_scores = list()
    for train_index, test_index in k_fold.split(known_data):
        x_train, x_test = known_data.iloc[train_index], known_data.iloc[test_index]
        y_train, y_test = list(map(float, known_answer.iloc[train_index])), list(map(float, known_answer.iloc[test_index]))
        linearSVR_regressor.fit(x_train, y_train)
        linearSVR_scores.append(linearSVR_regressor.score(x_test, y_test))
    print("Linear SVR:", numpy.mean(linearSVR_scores), linearSVR_scores)

    # SGD
    SGD_regressor = sklearn.linear_model.SGDRegressor(max_iter=10 ** 8, random_state=0, learning_rate="optimal", early_stopping=True)
    SGD_scores = list()
    for train_index, test_index in k_fold.split(known_data):
        x_train, x_test = known_data.iloc[train_index], known_data.iloc[test_index]
        y_train, y_test = list(map(float, known_answer.iloc[train_index])), list(map(float, known_answer.iloc[test_index]))
        SGD_regressor.fit(x_train, y_train)
        SGD_scores.append(SGD_regressor.score(x_test, y_test))
    print("SGD:", numpy.mean(SGD_scores), SGD_scores)

    # Decision Tree
    DecisionTree_regressor = sklearn.tree.DecisionTreeRegressor(max_features=None, random_state=0)
    DecisionTree_scores = list()
    for train_index, test_index in k_fold.split(known_data):
        x_train, x_test = known_data.iloc[train_index], known_data.iloc[test_index]
        y_train, y_test = list(map(float, known_answer.iloc[train_index])), list(map(float, known_answer.iloc[test_index]))
        DecisionTree_regressor.fit(x_train, y_train)
        DecisionTree_scores.append(DecisionTree_regressor.score(x_test, y_test))
    print("Decision Tree:", numpy.mean(DecisionTree_scores), DecisionTree_scores)
