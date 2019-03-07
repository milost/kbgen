import datetime
import pickle
from pathlib import Path

from rdflib import Graph, URIRef, Literal
from argparse import ArgumentParser, Namespace

from tqdm import tqdm

from kbgen.util_models import RealWorldOracle
from kbgen.rules import RealWorldRuleSet


def cli_args() -> Namespace:
    parser = ArgumentParser(description="Mangle a yago graph")
    parser.add_argument("input", type=str, default=None, help="path to the binary graph file")
    parser.add_argument("-r", "--rules_path", type=str, help="the rule file containing the amie rules")
    return parser.parse_args()


def save_mangled_graph(graph: Graph, oracle: RealWorldOracle, input_file: Path):
    oracle_output = f"{input_file.parent}/{input_file.stem}_oracle.json"
    print(f"Saving graph to {oracle_output}")
    oracle.to_json(open(oracle_output, "w"))

    pickle_output = f"{input_file.parent}/{input_file.stem}_mangled{input_file.suffix}"
    print(f"Saving graph to {pickle_output}")
    with open(pickle_output, "wb") as graph_file:
        pickle.dump(graph, graph_file, protocol=pickle.HIGHEST_PROTOCOL)

    tsv_output = f"{input_file.parent}/{input_file.stem}_mangled.tsv"
    print(f"Saving graph to {tsv_output}")
    with open(tsv_output, "w") as file:
        for subject, predicate, object in tqdm(graph):
            if not isinstance(object, Literal):
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
    rules = RealWorldRuleSet.parse_amie(args.rules_path)

    oracle_facts_correctness = {}
    oracle_facts_correctness_ratio = {}
    oracle_facts_brokenness_ratio = {}
    for rule in rules.rules:
        oracle_facts_correctness[rule] = {}
        threshold = datetime.date(year=1950, month=1, day=1)
        graph, oracle_data = rule.break_by_birth_date(graph,
                                                      birth_date_relation=URIRef("wasBornOnDate"),
                                                      break_chance=0.7,
                                                      threshold=threshold)
        facts_correctness, correctness_ratio, brokenness_ratio = oracle_data
        oracle_facts_correctness[rule] = facts_correctness
        oracle_facts_correctness_ratio[rule] = correctness_ratio
        oracle_facts_brokenness_ratio[rule] = brokenness_ratio

    oracle = RealWorldOracle(facts_to_correctness=oracle_facts_correctness,
                             rules_to_correctness_ratio=oracle_facts_correctness_ratio,
                             rules_to_brokenness_ratio=oracle_facts_brokenness_ratio)

    save_mangled_graph(graph, oracle, input_path)


if __name__ == '__main__':
    main()
