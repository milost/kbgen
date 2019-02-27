import json
from typing import Dict, TextIO, Optional

from ..rules import Rule, RudikRule


class Oracle(object):
    def __init__(self,
                 facts_to_correctness: Dict[Rule, Dict[tuple, bool]],
                 rules_to_correctness_ratio: Dict[Rule, float],
                 rules_to_brokenness_ratio: Dict[Rule, float]):
        """
        Creates an oracle object from the dictionary that contains the correctness information of the facts for each
        rule as well as from the dictionary that contains the ratio of broken facts for each rule.
        :param facts_to_correctness: contains the correctness of each fact for each rule
        :param rules_to_correctness_ratio: contains the percentage of correctness (i.e., 1.0 means every fact is correct
                                           for each rule)
        :param rules_to_brokenness_ratio: how many edges were removed (relative number) without recording them in the
                                          oracle
        """
        self.facts_to_correctness: Dict[Rule, Dict[tuple, bool]] = facts_to_correctness
        self.rules_to_correctness_ratio: Dict[Rule, float] = rules_to_correctness_ratio
        self.rules_to_brokenness_ratio: Dict[Rule, float] = rules_to_brokenness_ratio

    def serialize_rule(self, rule: Rule) -> dict:
        def fact_to_json(fact: tuple, correctness: bool):
            subject_uri, relation_uri, object_uri = fact
            return {
                "subject": str(subject_uri),
                "relation": str(relation_uri),
                "object": str(object_uri),
                "correct": correctness
            }
        facts_to_correctness = self.facts_to_correctness[rule]
        return {
            "rule": rule.to_dict(),
            "facts": [fact_to_json(fact, correctness) for fact, correctness in facts_to_correctness.items()],
            "ratio": self.rules_to_correctness_ratio[rule],
            "removed_edges": self.rules_to_brokenness_ratio[rule]
        }

    def to_json(self, file: TextIO=None) -> Optional[str]:
        data_dict = {}
        for rule in self.facts_to_correctness:
            if isinstance(rule, RudikRule):
                data_dict[rule.hashcode] = self.serialize_rule(rule)
            else:
                data_dict[str(rule)] = self.serialize_rule(rule)

        if file:
            json.dump(data_dict, file)
        else:
            return json.dumps(data_dict)
