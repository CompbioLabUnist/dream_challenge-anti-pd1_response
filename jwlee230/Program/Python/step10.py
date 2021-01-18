"""
step10.py: Select & Merge columns
"""
import argparse
import pandas
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ours", type=str, help="Input TAR.gz file")
    parser.add_argument("synapse", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output TAR.gz file")

    args = parser.parse_args()

    wanted_column = args.output.split("/")[-1].split(".")[0]

    our_data = step00.read_pickle(args.ours)
    our_data.dropna(axis="index", inplace=True)
    our_answer_data = our_data[list(filter(lambda x: x.startswith("Clinical_"), list(our_data.columns)))[0]]
    del our_data[list(filter(lambda x: x.startswith("Clinical_"), list(our_data.columns)))[0]]
    print(our_data)

    synapse_data = step00.read_pickle(args.synapse)
    synapse_data.dropna(axis="index", inplace=True)
    synapse_answer_data = synapse_data[list(filter(lambda x: x.startswith("Clinical_"), list(synapse_data.columns)))[0]]
    del synapse_data[list(filter(lambda x: x.startswith("Clinical_"), list(synapse_data.columns)))[0]]
    print(synapse_data)

    selected_columns = sorted(set(map(lambda x: x.split("_")[-1], list(our_data.columns))) & set(map(lambda x: x.split("_")[-1], list(synapse_data.columns))))
    total_length = len(selected_columns)
    print("Selected:", total_length)

    our_merge_data = pandas.DataFrame()
    for i, gene in enumerate(selected_columns):
        print(i, total_length, gene)
        our_merge_data[gene] = our_data[list(filter(lambda x: x.endswith(gene), list(our_data.columns)))].sum(axis=1)
    our_merge_data[wanted_column] = our_answer_data
    print(our_merge_data)

    synapse_merge_data = pandas.DataFrame()
    for i, gene in enumerate(selected_columns):
        print(i, total_length, gene)
        synapse_merge_data[gene] = synapse_data[list(filter(lambda x: x.endswith(gene), list(synapse_data.columns)))].sum(axis=1)
    synapse_merge_data[wanted_column] = synapse_answer_data
    print(synapse_merge_data)

    output_data = pandas.concat([our_merge_data, our_merge_data], axis="index", join="inner", ignore_index=True, verify_integrity=True)
    print(output_data)
    step00.make_pickle(args.output, output_data)
