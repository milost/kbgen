import json
import pickle
import random
from typing import List

from rdflib import Graph
from argparse import ArgumentParser, Namespace

from tqdm import tqdm

from kbgen.rules import RealWorldRuleSet, RealWorldRule


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
        # print(f"Enforcing rule {rule}")
        # graph = rule.enforce(graph)
        distribution = rule.get_distribution(graph)
        print(f"Got distribution for rule {rule}: {distribution}")

    # file_ending = args.input[args.input.rindex(".") + 1:]
    # output = f"{args.input[:args.input.rindex('.')]}_enforced.{file_ending}"
    # print(f"Saving graph to {output}")
    # with open(output, "wb") as graph_file:
    #     pickle.dump(graph, graph_file, protocol=pickle.HIGHEST_PROTOCOL)


def get_dist(rule: RealWorldRule, graph: Graph):
    premise_predicate = rule.premise[0].relation
    conclusion_predicate = rule.conclusion.relation

    distribution = {}
    for sub, _, ob in tqdm(graph.triples((None, premise_predicate, None)), total=335407):
        assert (ob, conclusion_predicate, sub) in graph, f"{ob} {conclusion_predicate} {sub} not in graph"
        if ob not in distribution:
            distribution[ob] = 0
        distribution[ob] += 1

    return distribution


def get_freq(rule: RealWorldRule, graph: Graph):
    premise_predicate = rule.premise[0].relation
    conclusion_predicate = rule.conclusion.relation

    distribution = {}
    for sub, _, ob in tqdm(graph.triples((None, premise_predicate, None)), total=335407):
        assert (ob, conclusion_predicate, sub) in graph, f"{ob} {conclusion_predicate} {sub} not in graph"
        if ob not in distribution:
            distribution[ob] = 0
        distribution[ob] += 1

    frequencies = {}
    for _, occurrences in tqdm(distribution.items()):
        if occurrences not in frequencies:
            frequencies[occurrences] = 0
        frequencies[occurrences] += 1

    return frequencies


def break_to_ground_truth(rule: RealWorldRule, graph: Graph):
    premise = rule.premise[0]
    query = f"SELECT {premise.literal_subject()} {premise.literal_object()} "
    query += f"WHERE {{"
    query += f"{premise.sparql_patterns()}"
    query += f"FILTER NOT EXISTS {{ {premise.literal_object()} rdf:type <http://dbpedia.org/ontology/Person> . }}"
    query += f"}}"

    print(f"Querying: {query}")

    result = graph.query(query)
    return result


def get_confidence(min: float, max: float):
    confidence_range = max - min
    scale = 1.0 / confidence_range
    confidence = random.random() / scale
    return confidence + min


def break_randomly(rule: RealWorldRule, graph: Graph, confidences: List[float] = None):
    if confidences:
        ground_truth_confidence, noise_confidence = confidences
        noise_confidence += ground_truth_confidence
    else:
        ground_truth_confidence = get_confidence(0.05, 0.2)
        noise_confidence = get_confidence(ground_truth_confidence + 0.05, ground_truth_confidence + 0.2)

    premise_predicate = rule.premise[0].relation
    conclusion_predicate = rule.conclusion.relation

    num_ground_truth_facts = 0
    num_noise_facts = 0
    total_count = 0
    oracle = []
    for subject, _, object in tqdm(graph.triples((None, premise_predicate, None)), total=335407):
        total_count += 1
        number = random.random()
        fact = (object, conclusion_predicate, subject)
        oracle_fact = {"subject": str(object), "predicate": str(conclusion_predicate), "object": str(subject)}
        # ground truth
        if number < ground_truth_confidence:
            oracle_fact["correctness"] = False
            num_ground_truth_facts += 1
            graph.remove(fact)
        # step 2 (breaking the fact without recording it)
        elif number < noise_confidence:
            oracle_fact["correctness"] = True
            oracle_fact["noise_removal"] = True
            num_noise_facts += 1
            graph.remove(fact)
        # don't touch the fact
        else:
            oracle_fact["correctness"] = True
        oracle.append(oracle_fact)

    final_oracle = {
        "rule": rule.to_dict(),
        "num_ground_truth_facts_removed": num_ground_truth_facts,
        "num_noise_facts_removed": num_noise_facts,
        "oracle": oracle,
        "ground_truth_ratio": num_ground_truth_facts / total_count,
        "noise_ratio": num_noise_facts / total_count
    }
    with open("dbpedia_oracle.json", "w") as file:
        json.dump(final_oracle, file)

    print("Saving graph")
    iri = "http://dbpedia.org/"
    with open("dbpedia_noisy_filtered.tsv", "w") as file:
        for subject, predicate, object in tqdm(graph):
            if iri in str(subject) and iri in str(predicate) and iri in str(object):
                line = str(subject) + "\t" + str(predicate) + "\t" + str(object)
                file.write(f"{line}\\n")

    return graph


if __name__ == '__main__':
    main()
