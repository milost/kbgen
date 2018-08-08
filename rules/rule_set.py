from typing import Dict

from rdflib import URIRef

from rules import Rule


class RuleSet(object):
    def __init__(self, rules=None):
        self.rules = rules or []
        self.rules_per_relation = {}
        for rule in rules:
            for literal in rule.antecedents:
                if literal.relation.id not in self.rules_per_relation:
                    self.rules_per_relation[literal.relation.id] = []
                self.rules_per_relation[literal.relation.id].append(rule)

    @classmethod
    def parse_amie(cls, rules_path: str, relation_to_id: Dict[URIRef, int]):
        rules = []
        with open(rules_path, "r") as file:
            for line in file:
                if line.startswith("?"):
                    rule = Rule.parse_amie(line, relation_to_id)
                    if rule is not None:
                        rules.append(rule)
        print(f"Rules successfully parsed: {len(rules)}...")
        return cls(rules)
