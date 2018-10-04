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
