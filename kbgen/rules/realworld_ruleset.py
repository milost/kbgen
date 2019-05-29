import json
from typing import Dict, List

from rdflib import URIRef

from kbgen.util import read_csv
from .realworld_rule import RealWorldRule


class RealWorldRuleSet(object):
    def __init__(self, rules: List[RealWorldRule] = None):
        """
        Creates a set of rules given a list of rules. This object knows which relation types are affected by which
        rules.
        :param rules: the parsed rudik rules
        """
        self.rules: List[RealWorldRule] = rules or []

        # dictionary that contains the rules grouped by the relation types that they have in their premise
        # a rule that has multiple literals in its premise will be in the both lists for the different relation types
        # in it a relation id points to a list of rules that all affect this relation type (i.e., a new fact with this
        # relation type needs to be checked against all the rules in that list)
        self.rules_per_relation: Dict[URIRef, List[RealWorldRule]] = {}

        self.fill_rules_per_relation()

    def fill_rules_per_relation(self):
        for rule in self.rules:
            for literal in rule.premise:

                if literal.relation not in self.rules_per_relation:
                    self.rules_per_relation[literal.relation] = []

                self.rules_per_relation[literal.relation].append(rule)

    def contains_negative_rules(self):
        for rule in self.rules:
            if rule.is_negative():
                return True
        return False

    @classmethod
    def parse_rudik(cls, rule_file: str) -> 'RealWorldRuleSet':
        parsed_file = json.load(open(rule_file, "r", encoding="utf-8"))
        rules = []
        for rule_dict in parsed_file:
            try:
                rules.append(RealWorldRule.parse_rudik(rule_dict))
            except RuntimeError as e:
                print(e)
        print(f"Rules successfully parsed: {len(rules)}...")
        return cls(rules)

    @classmethod
    def parse_amie(cls, rule_file: str, rule_type: bool = True) -> 'RealWorldRuleSet':
        csv_lines = read_csv(rule_file)
        rules = []
        for line in csv_lines:
            try:
                rules.append(RealWorldRule.parse_amie(line, rule_type))
            except RuntimeError as e:
                print(e)
        print(f"Rules successfully parsed: {len(rules)}...")
        return cls(rules)
