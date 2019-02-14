import json
from typing import Dict, List

from rdflib import URIRef

from .rudik_rule import RudikRule
from .rule import Rule


class RuleSet(object):
    def __init__(self, rules: List[Rule] = None):
        """
        Creates a set of rules given a list of rules. This object knows which relation types are affected by which
        rules.
        :param rules: the AMIE rules that were parsed from a file
        """
        self.rules: List[Rule] = rules or []

        # dictionary that contains the rules grouped by the relation types that they have in their premise
        # a rule that has multiple literals in its premise will be in the both lists for the different relation types
        # in it a relation id points to a list of rules that all affect this relation type (i.e., a new fact with this
        # relation type needs to be checked against all the rules in that list)
        self.rules_per_relation: Dict[int, List[Rule]] = {}

        self.fill_rules_per_relation()

    def fill_rules_per_relation(self):
        for rule in self.rules:
            for literal in rule.antecedents:

                if literal.relation.id not in self.rules_per_relation:
                    self.rules_per_relation[literal.relation.id] = []

                self.rules_per_relation[literal.relation.id].append(rule)

    def contains_negative_rules(self):
        for rule in self.rules:
            if rule.is_negative():
                return True
        return False

    def to_rudik(self):
        self.rules = [rule.to_rudik() for rule in self.rules]
        self.fill_rules_per_relation()

    @classmethod
    def parse_amie(cls, rules_path: str, relation_to_id: Dict[URIRef, int]) -> 'RuleSet':
        """
        Parses AMIE rules in the given file, translates relation URIs to relation ids and creates a rule set object
        from them.
        :param rules_path: path to the file containing AMIE rules
        :param relation_to_id: dictionary pointing from relation URIs to the ids used in the models
        :return: rule set object containing the parsed AMIE rules
        """
        rules = []
        with open(rules_path, "r") as file:
            for line in file:
                # the line contains an AMIE rule that needs to be parsed
                if line.startswith("?"):
                    rule = Rule.parse_amie(line, relation_to_id)
                    if rule is not None:
                        rules.append(rule)
        print(f"Rules successfully parsed: {len(rules)}...")
        return cls(rules)

    @classmethod
    def parse_rudik(cls, rules_path: str, relation_to_id: Dict[URIRef, int]) -> 'RuleSet':
        parsed_file = json.load(open(rules_path, "r", encoding="utf-8"))
        rules = [RudikRule.parse_rudik(rule_dict, relation_to_id) for rule_dict in parsed_file]
        print(f"Rules successfully parsed: {len(rules)}...")
        return cls(rules)
