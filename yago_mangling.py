import datetime
import operator
import pickle
from pathlib import Path
from typing import Tuple, List

from rdflib import Graph, URIRef, Literal
from argparse import ArgumentParser, Namespace

from tqdm import tqdm

from kbgen.rules.systematic_rule_breaking import break_by_birth_date
from kbgen.util_models import RealWorldOracle
from kbgen.rules import RealWorldRuleSet, RealWorldRule


def cli_args() -> Namespace:
    parser = ArgumentParser(description="Mangle a yago graph")
    parser.add_argument("input", type=str, default=None, help="path to the binary graph file")
    parser.add_argument("-r", "--rules_path", type=str, help="the rule file containing the amie rules")
    parser.add_argument("--random",  dest='random', action='store_true', help="set if the rules should be broken at "
                                                                              "random")
    return parser.parse_args()


def save_mangled_graph(graph: Graph,
                       oracle: RealWorldOracle,
                       input_file: Path,
                       random: bool,
                       filter_literals: bool = True):
    suffix = "mangled_random" if random else "mangled_systematic"

    oracle_output = f"{input_file.parent}/{input_file.stem}_{suffix}_oracle.json"
    print(f"Saving graph to {oracle_output}")
    oracle.to_json(open(oracle_output, "w"))

    pickle_output = f"{input_file.parent}/{input_file.stem}_{suffix}{input_file.suffix}"
    print(f"Saving graph to {pickle_output}")
    with open(pickle_output, "wb") as graph_file:
        pickle.dump(graph, graph_file, protocol=pickle.HIGHEST_PROTOCOL)

    tsv_output = f"{input_file.parent}/{input_file.stem}_{suffix}.tsv"
    print(f"Saving graph to {tsv_output}")
    with open(tsv_output, "w") as file:
        for subject, predicate, object in tqdm(graph):
            if not (isinstance(object, Literal) and filter_literals):
                line = str(subject) + "\t" + str(predicate) + "\t" + str(object)
                file.write(f"{line}\n")


def method_for_rule(rule: RealWorldRule):
    return {
        "created(subject, object) & produced(subject, object) => directed(subject, object)": {
            "method": break_by_birth_date,
            "arguments": {
                "rule": rule,
                "birth_date_relation": URIRef("wasBornOnDate"),
                "comparison_date": datetime.date(year=1950, month=1, day=1),
                "comparison": operator.lt,
                "break_chance": 0.7
            }
        }
    }[rule._to_rudik_str()]


def break_systematic(graph: Graph, rules: RealWorldRuleSet) -> Tuple[Graph, RealWorldOracle]:
    def break_function(rule: RealWorldRule, graph: Graph):
        systematic_method_data = method_for_rule(rule)
        method = systematic_method_data["method"]
        arguments = systematic_method_data["arguments"]
        return method(graph=graph, **arguments)

    return break_rules(rules=rules.rules, break_function=break_function, graph=graph)


def break_randomly(graph: Graph, rules: RealWorldRuleSet, break_chance: float) -> Tuple[Graph, RealWorldOracle]:
    def break_function(rule: RealWorldRule, **kwargs):
        return rule.break_randomly(**kwargs)

    return break_rules(rules=rules.rules, break_function=break_function, graph=graph, break_chance=break_chance)


def break_rules(rules: List[RealWorldRule], break_function: callable, **kwargs) -> Tuple[Graph, RealWorldOracle]:
    oracle_facts_correctness = {}
    oracle_facts_correctness_ratio = {}
    oracle_facts_brokenness_ratio = {}
    for rule in rules:
        oracle_facts_correctness[rule] = {}
        graph, oracle_data = break_function(rule, **kwargs)
        facts_correctness, correctness_ratio, brokenness_ratio = oracle_data
        oracle_facts_correctness[rule] = facts_correctness
        oracle_facts_correctness_ratio[rule] = correctness_ratio
        oracle_facts_brokenness_ratio[rule] = brokenness_ratio

    oracle = RealWorldOracle(facts_to_correctness=oracle_facts_correctness,
                             rules_to_correctness_ratio=oracle_facts_correctness_ratio,
                             rules_to_brokenness_ratio=oracle_facts_brokenness_ratio)
    return graph, oracle


def main():
    args = cli_args()
    print(args)

    input_path = Path(args.input)

    print(f"Loading graph from {input_path}")
    with open(input_path, "rb") as graph_file:
        graph: Graph = pickle.load(graph_file)

    print(f"Loading rules from {args.rules_path}")
    rules = RealWorldRuleSet.parse_amie(args.rules_path)

    if args.random:
        graph, oracle = break_randomly(graph, rules, break_chance=0.3)
    else:
        graph, oracle = break_systematic(graph, rules)

    save_mangled_graph(graph, oracle, input_path, random=args.random)


if __name__ == '__main__':
    main()
