import json
from typing import Dict, TextIO, Optional

from rdflib import Graph, URIRef

from ..rules import RealWorldRule


class RealWorldOracle(object):
    def __init__(self,
                 facts_to_correctness: Dict[RealWorldRule, Dict[tuple, bool]],
                 rules_to_correctness_ratio: Dict[RealWorldRule, float],
                 rules_to_brokenness_ratio: Dict[RealWorldRule, float]):
        """
        Creates an oracle object from the dictionary that contains the correctness information of the facts for each
        rule as well as from the dictionary that contains the ratio of broken facts for each rule.
        :param facts_to_correctness: contains the correctness of each fact for each rule
        :param rules_to_correctness_ratio: contains the percentage of correctness (i.e., 1.0 means every fact is correct
                                           for each rule)
        :param rules_to_brokenness_ratio: how many edges were removed (relative number) without recording them in the
                                          oracle
        """
        self.facts_to_correctness: Dict[RealWorldRule, Dict[tuple, bool]] = facts_to_correctness
        self.rules_to_correctness_ratio: Dict[RealWorldRule, float] = rules_to_correctness_ratio
        self.rules_to_brokenness_ratio: Dict[RealWorldRule, float] = rules_to_brokenness_ratio

    def serialize_rule(self, rule: RealWorldRule) -> dict:
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
            "ground_truth_ratio": self.rules_to_correctness_ratio[rule],
            "noise_ratio": self.rules_to_brokenness_ratio[rule]
        }

    def to_json(self, file: TextIO = None) -> Optional[str]:
        data_dict = {}
        for rule in self.facts_to_correctness:
            data_dict[rule.hashcode] = self.serialize_rule(rule)

        if file:
            json.dump(data_dict, file)
        else:
            return json.dumps(data_dict)

    @classmethod
    def from_gold_standard(cls, rule: RealWorldRule, graph: Graph, gold_standard_dict: dict):
        gold_standard = {}
        for example in gold_standard_dict["examples"]:
            key = URIRef(example["subject"]), URIRef(example["object"])
            gold_standard[key] = example["gold_standard"]

        predicate = rule.conclusion.relation
        facts_to_correctness = {}
        num_correct_facts = 0
        num_broken_facts = 0
        for subject, object in graph.query(rule.full_query_pattern()):
            fact = subject, predicate, object

            is_correct = gold_standard[(subject, object)]
            facts_to_correctness[fact] = is_correct
            num_correct_facts += is_correct

            is_broken = is_correct and fact not in graph
            num_broken_facts += is_broken

        correctness_ratio = num_correct_facts / len(gold_standard)
        brokenness_ratio = num_broken_facts / len(gold_standard)
        return cls(facts_to_correctness={rule: facts_to_correctness},
                   rules_to_correctness_ratio={rule: correctness_ratio},
                   rules_to_brokenness_ratio={rule: brokenness_ratio})
