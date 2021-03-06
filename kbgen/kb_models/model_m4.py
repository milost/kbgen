import random
from typing import Dict

from rdflib import Graph, URIRef

from .model_m3 import KBModelM3
from ..rules import Rule
from ..util_models import URIRelation, Oracle, URIType


class KBModelM4(KBModelM3):
    """
    This model does not add any features to the M3 model. It only provides a different synthesize method that breaks
    rules after the synthesization step and remembers which facts were removed via an oracle.
    """
    def __init__(self, m3_model: KBModelM3):
        """
        Initialize lower model variables and the variables used for rule breaking in the synthesization step.
        """
        assert isinstance(m3_model, KBModelM3), f"Model is of type {type(m3_model)} but needs to be of type KBModelM3"
        super(KBModelM4, self).__init__(
            m2_model=m3_model,
            rules=m3_model.rules,
            enforce_rules=m3_model.enforce_rules
        )
        self.facts_to_correctness: Dict[Rule, Dict[tuple, bool]] = {}
        self.rules_to_correctness_ratio: Dict[Rule, float] = {}
        self.rules_to_brokenness_ratio: Dict[Rule, float] = {}
        self.oracle: Oracle = None

    def get_confidence(self, min: float, max: float):
        """
        The confidence should be between the min and max values.
        :return: a confidence in the defined range
        """
        confidence_range = max - min
        scale = 1.0 / confidence_range
        confidence = random.random() / scale
        return confidence + min

    def break_rule(self,
                   graph: Graph,
                   rule: Rule,
                   oracle_confidence: float = None,
                   breaking_confidence: float = None) -> None:
        """
        A rule is broken in a two step process. Initially it is enforced in 100% of the cases. First we break the rule
        according to a target confidence. These changes are recorded in the oracle and represent the ground truth.
        Then we break the rule again according to a second confidence. This represents actually breaking the graph
        and should be repaired by the model we train later on.
        :param graph: the graph from which triples will be removed
        :param rule: the current rule
        :param oracle_confidence: float between 0.0 and 1.0 which represents in how many cases the rule is broken and
                                  recorded in the oracle (e.g., 0.2 means that 20% of the facts are broken (removed)
                                  and set to False in the oracle).
        :param breaking_confidence: float between 0.0 and 1.0 which represents in how many cases the rule is broken
                                    in the second step. The sum of this confidence and the oracle confidence represent
                                    how many facts in total are removed (e.g. sum of 1.0 means all facts are removed).
        """
        oracle_confidence = oracle_confidence or self.get_confidence(min=0.08, max=0.3)
        breaking_confidence = breaking_confidence or self.get_confidence(min=0.1, max=0.2)
        self.logger.info(f"Breaking rule {rule} with oracle confidence {oracle_confidence} and breaking facts "
                         f"confidence {breaking_confidence}")

        if rule not in self.facts_to_correctness:
            self.facts_to_correctness[rule] = {}

        # get the uri of the relation type of the resulting triple (conclusion)
        relation_uri = URIRelation.get_uri(rule.consequents[0].relation)

        # query the graph for all subjects and objects that fulfill this rule
        rule_query = rule.full_query_pattern()
        query_result = graph.query(rule_query)

        num_total_facts = len(query_result)
        num_oracle_facts = 0
        num_broken_facts = 0
        # add the oracle confidence so that we have the intervals (for the random number)
        # [0, oracle_confidence) => fact broken in step 1
        # [oracle_confidence, breaking_confidence) => fact broken in step 2
        # [breaking_confidence, 1.0] => fact not broken
        breaking_confidence += oracle_confidence
        # break the rule for a certain amount of facts (i.e., remove a number of facts)
        for subject_uri, object_uri in query_result:
            fact_to_break = (subject_uri, relation_uri, object_uri)
            number = random.random()
            # step 1 (oracle)
            if number < oracle_confidence:
                self.facts_to_correctness[rule][fact_to_break] = False
                num_oracle_facts += 1
                graph.remove(fact_to_break)
            # step 2 (breaking the fact without recording it)
            elif number < breaking_confidence:
                num_broken_facts += 1
                graph.remove(fact_to_break)
            # don't touch the fact
            else:
                self.facts_to_correctness[rule][fact_to_break] = True

        if num_total_facts:
            # how many edges were removed in step 2
            self.rules_to_brokenness_ratio[rule] = num_broken_facts / num_total_facts
            # how many edges were kept in step 1
            self.rules_to_correctness_ratio[rule] = 1.0 - num_oracle_facts / num_total_facts
        else:
            self.rules_to_brokenness_ratio[rule] = 0.0
            self.rules_to_correctness_ratio[rule] = 1.0

    def synthesize(self,
                   size: float = 1.0,
                   number_of_entities: int = None,
                   number_of_edges: int = None,
                   debug: bool = False,
                   pca: bool = True):
        """
        This extends the M3 synthesization by breaking the rules afterwards and tracking what was broken and what not.
        """
        graph = super(KBModelM4, self).synthesize(size, number_of_entities, number_of_edges, debug, pca)
        self.logger.info("Breaking rules...")
        for rule in self.rules.rules:
            self.break_rule(graph, rule)
        self.create_oracle()
        return graph

    def create_oracle(self):
        """
        Create the oracle after renaming the URIs in the changed facts to the proper names.
        """
        relation_id_to_uri: Dict[int, URIRef] = {relation_id: relation_uri
                                                 for relation_uri, relation_id in self.relation_to_id.items()}
        entity_type_id_to_uri: Dict[int, URIRef] = {type_id: type_uri
                                                    for type_uri, type_id in self.entity_type_to_id.items()}

        def replace_name(rdf_entity: URIRef) -> URIRef:
            """
            Replace the name of URIRelations and URITypes while keeping the names of URIEntities.
            :param rdf_entity: the synthesized URI that is resolved to its original name
            :return: the resolved URI if the original URI was an URIRelation or an URIType, otherwise the synthesized URI
            """
            name = rdf_entity

            if str(rdf_entity).startswith(URIRelation.prefix):
                name = relation_id_to_uri[URIRelation.extract_id(rdf_entity).id]
            elif str(rdf_entity).startswith(URIType.prefix):
                name = entity_type_id_to_uri[URIType.extract_id(rdf_entity).id]

            return URIRef(name)

        named_facts_to_correctness: Dict[Rule, Dict[tuple, bool]] = {}
        for rule, facts in self.facts_to_correctness.items():
            correctness_dict = {}
            for fact, correctness in facts.items():
                subject_uri, predicate_uri, object_uri = fact
                subject_with_name = replace_name(subject_uri)
                relation_with_name = replace_name(predicate_uri)
                object_with_name = replace_name(object_uri)
                named_fact = (subject_with_name, relation_with_name, object_with_name)
                correctness_dict[named_fact] = correctness
            named_facts_to_correctness[rule] = correctness_dict

        self.oracle = Oracle(named_facts_to_correctness,
                             self.rules_to_correctness_ratio,
                             self.rules_to_brokenness_ratio)
