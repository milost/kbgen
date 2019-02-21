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
        self.oracle: Oracle = None

    def break_rule(self, graph: Graph, rule: Rule, target_confidence: float = None) -> None:
        target_confidence = target_confidence or random.random() / 2.0 + 0.5
        self.logger.info(f"Breaking rule {rule} with factor {target_confidence}")

        if rule not in self.facts_to_correctness:
            self.facts_to_correctness[rule] = {}

        # get the uri of the relation type of the resulting triple
        rule_relation = rule.consequents[0].relation
        if isinstance(rule_relation, URIRelation):
            rule_relation_uri = rule.consequents[0].relation.uri
        else:
            rule_relation_uri = URIRelation(rule.consequents[0].relation).uri

        # query the graph for all subjects and objects that fulfill this rule
        rule_query = rule.full_query_pattern()
        query_result = graph.query(rule_query)

        num_total_facts = len(query_result)
        num_broken_facts = 0.0
        # break the rule for a certain amount of facts (i.e., remove a number of facts)
        for subject_uri, object_uri in query_result:
            fact_to_break = (subject_uri, rule_relation_uri, object_uri)
            rand_number = random.random()
            destroy_fact = rand_number > target_confidence
            self.facts_to_correctness[rule][fact_to_break] = not destroy_fact
            if destroy_fact:
                num_broken_facts += 1.0
                graph.remove(fact_to_break)

        if num_total_facts:
            ratio = 1.0 - num_broken_facts / num_total_facts
        else:
            ratio = 1.0
        self.rules_to_correctness_ratio[rule] = ratio

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

        self.oracle = Oracle(named_facts_to_correctness, self.rules_to_correctness_ratio)
