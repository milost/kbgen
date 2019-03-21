import pickle
from pathlib import Path
from typing import List

from rdflib import Graph
from argparse import ArgumentParser, Namespace

from tqdm import tqdm

from kbgen.rules import RealWorldRuleSet, RealWorldRule


def cli_args() -> Namespace:
    parser = ArgumentParser(description="Mangle a yago graph")
    parser.add_argument("input", type=str, default=None, help="path to the binary graph file")
    parser.add_argument("-r", "--rules_path", type=str, help="the rule file containing the amie rules")
    return parser.parse_args()


def main():
    args = cli_args()
    print(args)

    input_path = Path(args.input)

    print(f"Loading graph from {input_path}")
    with open(input_path, "rb") as graph_file:
        graph: Graph = pickle.load(graph_file)

    print(f"Loading rules from {args.rules_path}")
    rules: List[RealWorldRule] = RealWorldRuleSet.parse_amie(args.rules_path).rules

    for rule in tqdm(rules):
        rule.plot_frequency_distribution(graph)


if __name__ == '__main__':
    main()
