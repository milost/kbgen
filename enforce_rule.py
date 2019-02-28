import pickle

from rdflib import Graph
from argparse import ArgumentParser, Namespace

from kbgen.rules import RealWorldRuleSet


def cli_args() -> Namespace:
    parser = ArgumentParser(description="Load tensor from rdf data")
    parser.add_argument("input", type=str, default=None, help="path to the binary graph file")
    parser.add_argument("-r", "--rules_path", type=str, help="the rule file containing the rudik rules")
    return parser.parse_args()


def main():
    args = cli_args()
    print(args)

    print(f"Loading graph from {args.input}")
    with open(args.input, "rb") as graph_file:
        graph: Graph = pickle.load(graph_file)

    print(f"Loading rules from {args.rules_path}")
    rules = RealWorldRuleSet.parse_rudik(args.rules_path)

    for rule in rules.rules:
        print(f"Enforcing rule {rule}")
        graph = rule.enforce(graph)
        distribution = rule.get_distribution(graph)
        print(f"Got distribution {distribution}")

    # file_ending = args.input[args.input.rindex(".") + 1:]
    # output = f"{args.input[:args.input.rindex('.')]}_enforced.{file_ending}"
    # print(f"Saving graph to {output}")
    # with open(output, "wb") as graph_file:
    #     pickle.dump(graph, graph_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
