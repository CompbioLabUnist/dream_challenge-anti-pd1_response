"""
q1-final.py: for sub-challenge 1
"""
import sklearn.ensemble
import pandas
import step00

if __name__ == "__main__":
    train_data = step00.read_pickle("/Output/Step02/merged.tar.gz")
    train_data.dropna(axis="index", inplace=True)
    print(train_data)

    given_files = step00.file_list("/data")

    # clinical_data
    clinical_data = pandas.read_csv(given_files[-1])
    clinical_data.set_index("patientID", inplace=True)
    clinical_data["ECOGPS"] = list(map(lambda x: float(x) if step00.can_convert_to_float(x) else None, list(clinical_data["ECOGPS"])))
    clinical_data["TMB"] = list(map(lambda x: float(x) if step00.can_convert_to_float(x) else None, list(clinical_data["TMB"])))
    clinical_data.columns = list(map(lambda x: "Clinical_" + x, list(clinical_data.columns)))
    clinical_data.sort_index(axis="index", inplace=True)

    data_list = [clinical_data]
    for i, f in enumerate(given_files[:-2]):
        tmp_data = pandas.read_csv(f)
        tmp_data.set_index(list(tmp_data.columns)[0], inplace=True)
        tmp_data = tmp_data.T
        tmp_data.columns = list(map(lambda x: str(i) + "_" + x, list(tmp_data.columns)))
        tmp_data.sort_index(axis="index", inplace=True)
        data_list.append(tmp_data)

    given_data = pandas.concat(data_list, axis="columns", join="inner", verify_integrity=True)
    given_data = given_data.select_dtypes(exclude="object")

    selected_columns = sorted(set(train_data.columns) & set(map(lambda x: x.split("_")[-1], list(given_data.columns))))
    total = len(selected_columns)

    train_data = train_data[selected_columns + ["Clinical_PFS"]]
    train_answer = list(train_data["Clinical_PFS"])
    del train_data["Clinical_PFS"]

    test_data = pandas.DataFrame()
    for i, gene in enumerate(selected_columns):
        print(i, total, gene)
        test_data[gene] = given_data[list(filter(lambda x: x.endswith(gene), list(given_data.columns)))].sum(axis=1)
    print(test_data)

    classifier = sklearn.ensemble.RandomForestClassifier(max_features=None, n_jobs=1, random_state=0)
    classifier.fit(train_data, train_answer)
    answer = classifier.predict(test_data)
    print(answer)
