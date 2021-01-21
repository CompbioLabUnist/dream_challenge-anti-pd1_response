"""
q1-final.py: for sub-challenge 1
"""
import sklearn.ensemble
import pandas
import step00

if __name__ == "__main__":
    # clinical_data
    clinical_data = pandas.read_csv("/data/clinical_data.csv")
    clinical_data.set_index("patientID", inplace=True)
    clinical_data["ECOGPS"] = list(map(lambda x: float(x) if step00.can_convert_to_float(x) else None, list(clinical_data["ECOGPS"])))
    clinical_data["TMB"] = list(map(lambda x: float(x) if step00.can_convert_to_float(x) else None, list(clinical_data["TMB"])))
    clinical_data.columns = list(map(lambda x: "Clinical_" + x, list(clinical_data.columns)))
    clinical_data.sort_index(axis="index", inplace=True)

    data_list = [clinical_data]
    for i, f in enumerate(["GRCh37ERCC_ensembl75_isoforms_tpm.csv"]):
        tmp_data = pandas.read_csv(f)
        tmp_data.set_index(list(tmp_data.columns)[0], inplace=True)
        tmp_data = tmp_data.T
        tmp_data.columns = list(map(lambda x: str(i) + "_" + x, list(tmp_data.columns)))
        tmp_data.sort_index(axis="index", inplace=True)
        data_list.append(tmp_data)

    given_data = pandas.concat(data_list, axis="columns", join="inner", verify_integrity=True)
    given_data = given_data.select_dtypes(exclude="object")

    selected_columns = sorted(set(map(lambda x: x.split("_")[-1], list(given_data.columns))))
    total = len(selected_columns)

    test_data = pandas.DataFrame()
    for i, gene in enumerate(selected_columns):
        print(i, total, gene)
        test_data[gene] = given_data[list(filter(lambda x: x.endswith(gene), list(given_data.columns)))].sum(axis=1)
    test_data.info()

    regressor: sklearn.ensemble.RandomForestRegressor = step00.read_pickle("/Output/Step11/PFS.RF.tar.gz")

    answer_data = pandas.DataFrame()
    answer_data["patientID"] = list(test_data.index)
    answer_data["prediction"] = regressor.predict(test_data)
    answer_data.to_csv("/output/predictions.csv", index=False)
