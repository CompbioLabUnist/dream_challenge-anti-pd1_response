"""
step02.py: Clearify and Merge data
"""
import argparse
import pandas
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("expression", type=str, help="Expression TSV file")
    parser.add_argument("TPM", type=str, help="TPM TSV file")
    parser.add_argument("clinical", type=str, help="Clinical TSV file")
    parser.add_argument("output", type=str, help="Output TAR.gz file")

    args = parser.parse_args()

    if not args.expression.endswith(".tsv"):
        raise ValueError("EXPRESSION must end with .tsv!!")
    elif not args.TPM.endswith(".tsv"):
        raise ValueError("TPM must end with .tsv!!")
    elif not args.clinical.endswith(".tsv"):
        raise ValueError("CLINICAL must end with .tsv!!")

    # read data
    expression_data = pandas.read_csv(args.expression, sep="\t")
    expression_data.set_index("b.df", inplace=True)
    expression_data = expression_data.T
    expression_data.sort_index(axis="index", inplace=True)
    expression_data.columns = list(map(lambda x: "Expression_" + x, list(expression_data.columns)))
    expression_data.dropna(axis="columns", how="all", inplace=True)
    print(expression_data)

    TPM_data = pandas.read_csv(args.TPM, sep="\t")
    TPM_data.set_index("V1", inplace=True)
    TPM_data = TPM_data.T
    TPM_data.sort_index(axis="index", inplace=True)
    TPM_data.columns = list(map(lambda x: "TPM_" + x, list(TPM_data.columns)))
    TPM_data.dropna(axis="columns", how="all", inplace=True)
    print(TPM_data)

    clinical_data = pandas.read_csv(args.clinical, sep="\t")
    clinical_data.set_index("WTS_ID", inplace=True)
    clinical_data.sort_index(axis="index", inplace=True)
    clinical_data.dropna(axis="columns", how="all", inplace=True)
    print(clinical_data)

    # intersect
    intersect_index = set(expression_data.index) & set(TPM_data.index) & set(clinical_data.index)
    expression_data = expression_data.loc[list(map(lambda x: x in intersect_index, list(expression_data.index)))]
    TPM_data = TPM_data.loc[list(map(lambda x: x in intersect_index, list(TPM_data.index)))]
    clinical_data = clinical_data.loc[list(map(lambda x: x in intersect_index, list(clinical_data.index)))]

    # clear clinical data
    clinical_data["TMB"] = list(map(lambda x: float(x) if step00.can_convert_to_float(x) else None, list(clinical_data["TMB"])))
    clinical_data["IHC"] = list(map(lambda x: float(x) if (step00.can_convert_to_float(x) or (x != x)) else float(x.split("/")[-1]), list(clinical_data["IHC"])))
    clinical_data["sex"] = list(map(lambda x: {"1": "M", "2": "F", "male": "M", "female": "F"}[x], list(clinical_data["sex"])))
    clinical_data["Tobacco"] = list(map(lambda x: {"0": "NEVER", "Never": "NEVER", "1": "FORMER", "Ex": "FORMER", "2": "CURRENT", "Current": "CURRENT", "Unknown": "UNKNOWN"}[x], list(clinical_data["Tobacco"])))
    clinical_data.columns = list(map(lambda x: "Clinical_" + x, list(clinical_data.columns)))

    print(expression_data)
    print(TPM_data)
    print(clinical_data)

    # merge
    output_data = pandas.concat([expression_data, TPM_data, clinical_data], axis="columns", join="inner", verify_integrity=True)

    print(output_data)
    output_data.info()
    step00.make_pickle(args.output, output_data)
