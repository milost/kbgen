import pickle
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

from rdflib import URIRef

from kbgen.kb_models import KBModelM1
from kbgen.rules import RuleSet
from synthesize import URINameReplacer


def cli_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("input", type=str, default=None, help="path to the directory containing npy and npz files")
    parser.add_argument("-r", "--rules-path", type=str, default=None, help="path to txt file with Amie horn rules")
    return parser.parse_args()


def main():
    args = cli_args()
    input_dir = args.input

    m1_model_path = f"{input_dir}/{input_dir}-M1.pkl"
    m1_model = pickle.load(open(m1_model_path, "rb"))
    assert isinstance(m1_model, KBModelM1)

    name_replacer = URINameReplacer(m1_model)

    rule_set = RuleSet.parse_amie(args.rules_path, m1_model.relation_to_id)
    rule_set.to_rudik()
    rule_dump = []
    for rule in rule_set.rules:
        rule_dict = rule.to_dict()
        for triple in rule_dict["premise_triples"]:
            triple["predicate"] = str(name_replacer.replace_name(URIRef(triple["predicate"])))
        conclusion_predicate = rule_dict["conclusion_triple"]["predicate"]
        conclusion_predicate = str(name_replacer.replace_name(URIRef(conclusion_predicate)))
        rule_dict["conclusion_triple"]["predicate"] = conclusion_predicate
        rule_dump.append(rule_dict)

    with open(f"{Path(args.rules_path).stem}_rudik.json", "w") as rule_file:
        json.dump(rule_dump, rule_file)


if __name__ == '__main__':
    main()
