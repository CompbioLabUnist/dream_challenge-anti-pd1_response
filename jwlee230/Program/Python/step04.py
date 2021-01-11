"""
step04.py: Clearify and Merge Synapse Data
"""
import argparse
import pandas
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("clinical", type=str, help="Clinical CSV file")
    parser.add_argument("expression", type=str, help="Expression CSV file(s)", nargs="+")
    parser.add_argument("output", type=str, help="Output TAR.gz file")

    args = parser.parse_args()

    if not args.clinical.endswith(".csv"):
        raise ValueError("CLINICAL must end with .csv!!")
    elif list(filter(lambda x: not x.endswith(".csv"), args.expression)):
        raise ValueError("CLINICAL must end with .csv!!")

    # read data
    clinical_data = pandas.read_csv(args.clinical)
    clinical_data.set_index("patientID", inplace=True)
    clinical_data["ECOGPS"] = list(map(lambda x: float(x) if step00.can_convert_to_float(x) else None, list(clinical_data["ECOGPS"])))
    clinical_data["TMB"] = list(map(lambda x: float(x) if step00.can_convert_to_float(x) else None, list(clinical_data["TMB"])))
    clinical_data.sort_index(axis="index", inplace=True)

    data_list = [clinical_data]
    for i, expression_file in enumerate(sorted(args.expression)):
        tmp_data = pandas.read_csv(expression_file)
        tmp_data.set_index(list(tmp_data.columns)[0], inplace=True)
        tmp_data = tmp_data.T
        tmp_data.columns = list(map(lambda x: str(i) + "_" + x, list(tmp_data.columns)))
        tmp_data.sort_index(axis="index", inplace=True)
        data_list.append(tmp_data)

    output_data = pandas.concat(data_list, axis="columns", join="inner", verify_integrity=True)

    print(output_data)
    print(output_data.info())
    step00.make_pickle(args.output, output_data)
