import json
import pickle
from pathlib import Path
from typing import Tuple

from rdflib import Graph, URIRef
from argparse import ArgumentParser, Namespace

from tqdm import tqdm

from kbgen.rules import RealWorldRuleSet


def cli_args() -> Namespace:
    parser = ArgumentParser(description="Mangle a yago graph")
    parser.add_argument("input", type=str, default=None, help="path to the binary graph file")
    parser.add_argument("-r", "--rules_path", type=str, help="the rule file containing the amie rules")
    return parser.parse_args()


def create_example(query_tuple: Tuple[URIRef, URIRef], correct: bool, id: int):
    subject, object = query_tuple
    return {
        "subject": str(subject),
        "object": str(object),
        "correct": correct,
        "id": id,
        "gold_standard": None
    }


def main():
    args = cli_args()
    print(args)

    input_path = Path(args.input)

    print(f"Loading graph from {input_path}")
    with open(input_path, "rb") as graph_file:
        graph: Graph = pickle.load(graph_file)

    print(f"Loading rules from {args.rules_path}")
    ruleset = RealWorldRuleSet.parse_amie(args.rules_path)

    for rule in tqdm(ruleset.rules):
        template = {"rule": rule.to_dict()}
        positive_examples = set(graph.query(rule.full_query_pattern(include_conclusion=True)))
        examples = list(graph.query(rule.full_query_pattern()))
        examples = [create_example(example, example in positive_examples, id) for id, example in enumerate(examples)]
        template["examples"] = examples
        filename = f"rule_{rule.hashcode}_gold_standard.json"
        with open(filename, "w") as file:
            json.dump(template, file, indent=4)


if __name__ == '__main__':
    main()
