import json
import pickle
from pathlib import Path

from rdflib import Graph, Literal
from argparse import ArgumentParser, Namespace

from tqdm import tqdm

from kbgen.util_models import RealWorldOracle
from kbgen.rules import RealWorldRuleSet


def cli_args() -> Namespace:
    parser = ArgumentParser(description="Mangle a yago graph")
    parser.add_argument("input", type=str, default=None, help="path to the binary graph file")
    parser.add_argument("-r", "--rules_path", type=str, help="the rule file containing the amie rules")
    parser.add_argument("--gold_standard", type=str, help="path to the gold standard file")
    return parser.parse_args()


def save_graph(graph: Graph,
               oracle: RealWorldOracle,
               input_file: Path,
               filter_literals: bool = True):
    oracle_output = f"{input_file.parent}/{input_file.stem}_oracle.json"
    print(f"Saving oracle to {oracle_output}")
    oracle.to_json(open(oracle_output, "w"))

    print(f"Not saving graph binary since no changes were made.")

    tsv_output = f"{input_file.parent}/{input_file.stem}.tsv"
    print(f"Saving graph to {tsv_output}")
    with open(tsv_output, "w") as file:
        for subject, predicate, object in tqdm(graph):
            if not (isinstance(object, Literal) and filter_literals):
                line = str(subject) + "\t" + str(predicate) + "\t" + str(object)
                file.write(f"{line}\n")


def main():
    args = cli_args()
    print(args)

    input_path = Path(args.input)

    print(f"Loading graph from {input_path}")
    with open(input_path, "rb") as graph_file:
        graph: Graph = pickle.load(graph_file)

    print(f"Loading rules from {args.rules_path}")
    ruleset = RealWorldRuleSet.parse_amie(args.rules_path)
    assert len(ruleset.rules) == 1, "Currently only a ruleset with a single rule can be handled"

    rule = ruleset.rules[0]
    gold_standard = json.load(open(args.gold_standard))

    oracle = RealWorldOracle.from_gold_standard(rule=rule, graph=graph, gold_standard_dict=gold_standard)

    save_graph(graph=graph, oracle=oracle, input_file=input_path)


if __name__ == '__main__':
    main()
