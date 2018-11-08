import logging
import random
from copy import deepcopy
from typing import Dict, Set, Tuple

import numpy as np
import tqdm
from matplotlib import pyplot as plt
from kb_models.model_m2 import KBModelM2
from numpy.random import choice
from rdflib import Graph, URIRef
from rdflib.namespace import RDF
import datetime

from rules import RuleSet
from util_models import URIEntity, URIRelation, MultiType
from util import normalize, create_logger


class KBModelM3(KBModelM2):
    """
    Model based on the KBModelM2 and containing horn rules for synthesizing facts
    - Since the original distribution can be affected by the facts produced by rules, we keep
     the counts of the distributions and decrease them for every added fact, in order to make
      sure the original distribution is not disturbed.
    - Horn rules are assumed to be produced by AMIE
    """
    def __init__(self, m2_model: KBModelM2, rules: RuleSet, enforce_rules: bool = True):
        """
        Create an M3 model based on an M2 model and a rule set of horn rules (produces by AMIE).
        :param m2_model: the previously built M2 model
        :param rules: Horn rules that will be used when synthesizing a knowledge base. Currently it is assumed that they
                      are produced by AMIE TODO: why this assumption
        :param enforce_rules: if True rules will be enforced even if the resulting tuples will "destroy" distributions
        """
        assert isinstance(m2_model, KBModelM2), f"Model is of type {type(m2_model)} but needs to be of type KBModelM2"
        super(KBModelM3, self).__init__(
            m1_model=m2_model,
            functionalities=m2_model.functionalities,
            inverse_functionalities=m2_model.inverse_functionalities,
            relation_id_to_density=m2_model.relation_id_to_density,
            relation_id_to_distinct_subjects=m2_model.relation_id_to_distinct_subjects,
            relation_id_to_distinct_objects=m2_model.relation_id_to_distinct_objects,
            relation_id_to_reflexiveness=m2_model.relation_id_to_reflexiveness
        )
        self.rules = rules
        # the maximum number of recursions allowed when checking Horn rules
        self.max_recursion_count = 10
        # whether the "partial completeness assumption:" is used
        self.pca = True
        # if True rules will be always enforced, otherwise the results will be filtered according to the distributions
        self.enforce_rules = enforce_rules

        # initialized in other methods
        #
        # counters initialized in the method start_counts()
        self.fact_count = 0
        self.count_rule_facts = 0
        self.number_of_facts_with_empty_distributions = 0
        self.duplicate_fact_count = 0
        self.number_of_empty_pool_occurrences = 0
        self.num_facts_violating_functionality = 0
        self.num_facts_violating_inverse_functionality = 0
        self.num_facts_violating_non_reflexiveness = 0
        self.number_of_key_errors = 0

        # size of the synthetic knowledge base
        self.num_synthetic_entities = 0
        self.num_synthetic_facts = 0

        # inverse relation_to_id dictionary
        self.relation_id_to_relation: Dict[int, URIRef] = None

        # copies of distributions used for pruning
        self.adjusted_relations_distribution_copy: Dict[int, float] = None
        self.relation_domain_distribution_copy: Dict[int, Dict[MultiType, int]] = None
        self.relation_range_distribution_copy: Dict[int, Dict[MultiType, Dict[MultiType, int]]] = None

        # sets of entity ids that have specific relation as incoming or outgoing edges exactly once
        self.relation_id_to_subject_pool: Dict[int, Set[int]] = None
        self.relation_id_to_object_pool: Dict[int, Set[int]] = None

        # contain invalid combinations of subject types with object ids and vice versa
        self.saturated_subject_ids: Dict[int, Dict[MultiType, Set[int]]] = None
        self.saturated_object_ids: Dict[int, Dict[MultiType, Set[int]]] = None

        # logger used for the query times
        self.query_time = None

    def print_synthesis_details(self):
        """
        Print the statistics of the synthesization of a knowledge base with this M3 model.
        """
        super(KBModelM3, self).print_synthesis_details()
        self.logger.debug(f"exhausted: {self.number_of_facts_with_empty_distributions}")
        self.logger.debug(f"no entities of type: {self.number_of_empty_pool_occurrences}")
        self.logger.debug(f"key error {self.number_of_key_errors}")
        self.logger.debug(f"added by rules: {self.count_rule_facts}")
        self.logger.debug(f"step {self.step}")
        self.logger.debug(self.adjusted_relations_distribution_copy)

    def plot_histogram(self, vals, weights):
        """
        Plots a histrogram of precalculated counts (display distribution)
        :param vals: values of the bins (x axis)
        :param weights: counts of each bin (y axis)
        """
        x = np.array(vals)
        y = np.array(weights)
        y /= np.sum(y)

        plt.figure(0)
        n, bins, patches = plt.hist(x, weights=y, bins=len(vals), normed=1, facecolor='green', alpha=0.75)
        plt.show()

    def start_counts(self):
        """
        Initializes various counts.
        """
        # counters from M1 model
        self.fact_count = 0
        self.duplicate_fact_count = 0

        # counters from M2 model
        self.num_facts_violating_functionality = 0
        self.num_facts_violating_inverse_functionality = 0
        self.num_facts_violating_non_reflexiveness = 0

        # counters from this model
        self.count_rule_facts = 0
        self.number_of_facts_with_empty_distributions = 0
        self.number_of_empty_pool_occurrences = 0
        self.number_of_key_errors = 0

    def validate_owl(self, relation_id: int, subject_id: int, object_id: int):
        """
        Validates functionality, inverse functionality and non-reflexiveness for a given fact.

        :param relation_id: id of the relation in the fact
        :param subject_id: id of the subject in the fact
        :param object_id: id of the object in the fact
        :return: true if consistent, false otherwise
        """
        # assert that the subject is a viable subject for the relation type
        if relation_id in self.relation_id_to_subject_pool:
            if subject_id not in self.relation_id_to_subject_pool[relation_id]:
                self.num_facts_violating_functionality += 1
                return False

        # assert that the object is a viable object for the relation type
        if relation_id in self.relation_id_to_object_pool:
            if object_id not in self.relation_id_to_object_pool[relation_id]:
                self.num_facts_violating_inverse_functionality += 1
                return False

        # assert that the new fact is not reflexive if reflexiveness is not allowed for the relation type
        if not self.relation_id_to_reflexiveness[relation_id]:
            if subject_id == object_id:
                self.num_facts_violating_non_reflexiveness += 1
                return False

        return True

    def produce_rules(self,
                      graph: Graph,
                      relation_id: int,
                      fact: Tuple[URIRef, URIRef, URIRef],
                      recursion_count: int = 0):
        """
        Checks if a new fact added, triggers any horn rules to generate new facts.
        :param graph: synthesized graph
        :param relation_id: id of the added relation
        :param fact: added fact
        :param recursion_count: recursion count (tracks the recursion depth)
        """
        subject_uri, relation_uri, object_uri = fact

        # only continue if the maximum recursion level has not been reached yet and there are rules that affect the
        # relation type of the new fact
        if recursion_count < self.max_recursion_count and relation_id in self.rules.rules_per_relation:

            # the rules that affect the current relation type
            rules = self.rules.rules_per_relation[relation_id]

            # iterate over the rules
            for rule in rules:
                ##### TODO: this as well?
                # # only adhere to the rule according to its confidence level (i.e. a confidence of 0.5 means that only
                # # for half of the facts for which the rule would apply it is applied)
                # rand_number = random.random()
                # if (
                #     (self.pca and rand_number < rule.pca_confidence)
                #     or (not self.pca and rand_number < rule.standard_confidence)
                # ):
                #####
                # measure the time it takes for the rule to produce new facts
                start_t = datetime.datetime.now()

                # produce new facts that are generated by the rule
                new_facts = rule.produce(graph, subject_uri, relation_uri, object_uri)
                delta = datetime.datetime.now() - start_t

                rule_added_facts = 0
                for new_fact in new_facts:
                    # extract subject, object and relation ids from the URIs in the new fact
                    subject_uri, relation_uri, object_uri = new_fact
                    subject_id = URIEntity.extract_id(subject_uri).id
                    object_id = URIEntity.extract_id(object_uri).id
                    relation_id = URIRelation.extract_id(relation_uri).id

                    # get the multi types for the subject and object of the new fact
                    subject_type = self.synthetic_id_to_type[subject_id]
                    object_type = self.synthetic_id_to_type[object_id]

                    # only continue if the new fact does not violate functionality, inverse functionality and
                    # reflexiveness of the relation type and if there is a non-zero number of occurrences of this
                    # subject and object combination in the range distribution of this relation type
                    if (
                        self.validate_owl(relation_id, subject_id, object_id)
                        and (self.enforce_rules or
                             self.relation_range_distribution_copy[relation_id][subject_type][object_type] > 0)
                    ):
                        # returns True if the fact did not already exist
                        if self.add_fact(graph, new_fact):
                            rule_added_facts += 1

                            # update the distributions and pools after the new fact was added
                            # (i.e., reduce occurrences in distributions and remove subject and object from their
                            # respective pools)
                            self.update_distributions(relation_id, subject_type, object_type)
                            self.update_pools(relation_id, subject_id, object_id)
                            self.count_rule_facts += 1

                            # check if any rules apply to the new fact that was also generated by a rule
                            self.produce_rules(graph, relation_id, new_fact, recursion_count + 1)
                    else:
                        self.number_of_facts_with_empty_distributions += 1

                # log the information about the new facts that were added by this rule
                self.query_time.debug(f"rule_size={len(rule.antecedents)}, "
                                      f"number_of_produced_facts={len(list(new_facts))}, "
                                      f"newly_added_facts={rule_added_facts}, "
                                      f"query_time_micros={delta.microseconds}, "
                                      f"knowledge_base_size={len(graph)}")

    def update_distributions(self, relation_id: int, subject_type: MultiType, object_type: MultiType) -> None:
        """
        Updates the distributions after a given fact has been added
        (decreases the step from the original distribution counts)
        :param relation_id: id of the relation added
        :param subject_type: multitype of the subject
        :param object_type: multitype of the object
        """
        step = float(self.step)

        # if the relation is still in the pruned distributions
        if relation_id in self.adjusted_relations_distribution_copy:

            # reduce the occurrences of the relation by step and remove it from all three
            # distributions, if it falls to or below zero
            self.adjusted_relations_distribution_copy[relation_id] -= step
            if self.adjusted_relations_distribution_copy[relation_id] <= 0:
                self.delete_relation_entries(relation_id)

            # continue if the subject type is still in the pruned domain distributions
            if (
                relation_id in self.relation_domain_distribution_copy
                and subject_type in self.relation_domain_distribution_copy[relation_id]
            ):
                # reduce the occurrences of the domain type by step, and delete the subject type from the domain
                # and range distributions of that relation if it falls to or below zero
                self.relation_domain_distribution_copy[relation_id][subject_type] -= step
                if self.relation_domain_distribution_copy[relation_id][subject_type] <= 0:
                    self.delete_relation_domain_entries(relation_id, subject_type)

                # continue if the object type is still in the range distribution of the domain (subject type)
                if (
                    subject_type in self.relation_range_distribution_copy[relation_id]
                    and object_type in self.relation_range_distribution_copy[relation_id][subject_type]
                ):
                    # reduce the occurrences of the range type by step, and delete the object type from the range
                    # distribution of that relation and domain if it falls to or below zero
                    self.relation_range_distribution_copy[relation_id][subject_type][object_type] -= step
                    if self.relation_range_distribution_copy[relation_id][subject_type][object_type] <= 0:
                        self.delete_relation_domain_range_entries(relation_id, subject_type, object_type)

        # remove any resulting empty entries in the relation or domain distributions
        self.delete_empty_entries(relation_id, subject_type)

    def prune_distributions(self):
        """
        Prunes the original distributions based on the entities generated.
        If a given multitype does not have any instances in the synthesized data, all the entries
        in the distributions containing the given multitype are deleted
        """
        # aggregate of the entries removed from the distributions
        # the distributions contain occurrences of edges so this sum is a number of removed edges
        number_removed_edges = 0

        # local copies of the distributions that are pruned
        adjusted_relations_distribution = deepcopy(self.adjusted_relations_distribution_copy)
        relation_domain_distribution = deepcopy(self.relation_domain_distribution_copy)
        relation_range_distribution = deepcopy(self.relation_range_distribution_copy)

        for relation_id in adjusted_relations_distribution.keys():
            # delete relation from distributions if it never occurs (i.e., distribution value equals 0
            if adjusted_relations_distribution[relation_id] == 0:
                # delete the relation from all three distributions
                self.delete_relation_entries(relation_id)

            # iterate over distribution copies that still contain the deleted relation
            for domain in relation_domain_distribution[relation_id].keys():

                # if the domain multitype does not have any synthetic entities of its type
                # i.e., no entities of this domains multitype were generated
                if domain not in self.entity_types_to_entity_ids.keys() or not self.entity_types_to_entity_ids[domain]:
                    number_removed_edges += relation_domain_distribution[relation_id][domain]

                    # delete the domain from the original domain and range distributions
                    self.delete_relation_domain_entries(relation_id, domain)
                else:
                    # otherwise check each range of the domain
                    for range_of_domain in set(relation_range_distribution[relation_id][domain].keys()):
                        # if no entities of the type of this range were synthesized
                        if (
                            range_of_domain not in self.entity_types_to_entity_ids.keys()
                            or not self.entity_types_to_entity_ids[range_of_domain]
                        ):
                            number_removed_edges += relation_range_distribution[relation_id][domain][range_of_domain]

                            # delete the range from the original range distribution that was copied
                            self.delete_relation_domain_range_entries(relation_id, domain, range_of_domain)

                    # if the distribution contains no ranges for this domain delete the domain
                    if not relation_range_distribution[relation_id][domain]:
                        number_removed_edges += relation_domain_distribution[relation_id][domain]

                        # delete the domain from the original domain and range distributions
                        self.delete_relation_domain_entries(relation_id, domain)

            # if the distribution contains no domains for this relation delete the relation from all three distributions
            if not relation_domain_distribution[relation_id]:
                number_removed_edges += adjusted_relations_distribution[relation_id]

                # delete the relation from all three distributions
                self.delete_relation_entries(relation_id)

        # TODO: why is this done?
        # if any entries were removed from the distributions scale the step parameter (inverse size) down by a value
        # in [0, 1) in relation to the number of removed entries and the total edge count in the original knowledge base
        if number_removed_edges > 0:
            self.step *= (self.edge_count - number_removed_edges) / self.edge_count

    def delete_relation_entries(self, relation_id: int) -> None:
        """
        Delete the entry of the given relation in the three distributions of relations, relation domains and relation
        ranges.
        :param relation_id: the id of the relation to delete
        """
        if relation_id in self.adjusted_relations_distribution_copy:
            del self.adjusted_relations_distribution_copy[relation_id]

        if relation_id in self.relation_domain_distribution_copy:
            del self.relation_domain_distribution_copy[relation_id]

        if relation_id in self.relation_range_distribution_copy:
            del self.relation_range_distribution_copy[relation_id]

    def delete_relation_domain_entries(self, relation_id: int, domain: MultiType):
        """
        Delete the given domain from the distributions of the given relation in the distributions of domains and ranges
        :param relation_id: the relation whose domain type is deleted
        :param domain: the domain that will be removed from the original distributions
        """
        if (
            relation_id in self.relation_domain_distribution_copy
            and domain in self.relation_domain_distribution_copy[relation_id]
        ):
            del self.relation_domain_distribution_copy[relation_id][domain]

        if (
            relation_id in self.relation_range_distribution_copy
            and domain in self.relation_range_distribution_copy[relation_id]
        ):
            del self.relation_range_distribution_copy[relation_id][domain]

    def delete_relation_domain_range_entries(self,
                                             relation_id: id,
                                             domain: MultiType,
                                             range_of_domain: MultiType) -> None:
        """
        Check that the given range exists in the range distribution with the given relation and domain and delete it if
        it does.
        :param relation_id: id of the relation whose range is deleted
        :param domain: domain whose range is deleted
        :param range_of_domain: the range of the domain to delete
        """
        if (
            relation_id in self.relation_range_distribution_copy
            and domain in self.relation_range_distribution_copy[relation_id]
            and range_of_domain in self.relation_range_distribution_copy[relation_id][domain]
        ):
            del self.relation_range_distribution_copy[relation_id][domain][range_of_domain]

    def delete_empty_entries(self, relation_id: id, subject_type: MultiType) -> None:
        """
        Deletes empty relations, with empty domain distributions and subject types with empty range distributions from
        the distributions.
        :param relation_id: the id of the relation
        :param subject_type: the multitype of the subject
        """
        # the number of edge occurrences of a relation and edge occurrences of a relation with a specific domain that
        # are removed from distributions due to empty domain or empty range distributions
        number_of_removed_occurrences = 0

        # delete the relation from all three distributions if there are no domain distributions of the relation
        if (
            relation_id in self.relation_domain_distribution_copy
            and not self.relation_domain_distribution_copy[relation_id]
        ):
            number_of_removed_occurrences += self.adjusted_relations_distribution_copy[relation_id]
            self.delete_relation_entries(relation_id)

        # delete the subject type from the relations domain and and range distributions if there are no range
        # distributions for that subject type left
        if (
            relation_id in self.relation_range_distribution_copy
            and subject_type in self.relation_range_distribution_copy[relation_id]
            and not self.relation_range_distribution_copy[relation_id][subject_type]
        ):
            number_of_removed_occurrences += self.relation_domain_distribution_copy[relation_id][subject_type]
            self.delete_relation_domain_entries(relation_id, subject_type)

        # TODO: why is this scaling needed
        # scale the step parameter according the number of removed occurrences
        if number_of_removed_occurrences > 0:
            remaining_facts = float(self.entity_count - self.fact_count) * self.step
            self.step *= (remaining_facts - number_of_removed_occurrences) / remaining_facts

    def update_pools(self, relation_id: int, subject_id: int, object_id: int) -> None:
        """
        Remove the subject and object from their respective pools after the fact was successfully added. Also remove
        the relation from the distributions if either pool ends up empty and scale the step paramater according to the
        number of removed occurrences from the distributions.
        :param relation_id: the id of the relation
        :param subject_id: the id of the subject
        :param object_id: the id of the object
        """
        # the number of edge occurrences of a relation and that are removed from distributions due to an empty subject
        # or empty object pool
        number_of_removed_occurrences = 0

        # only continue if the relation still remains in the relation distribution
        if relation_id in self.adjusted_relations_distribution_copy:
            # remove the subject from the subject pool of the relation
            if (
                relation_id in self.relation_id_to_subject_pool
                and subject_id in self.relation_id_to_subject_pool[relation_id]
            ):
                self.relation_id_to_subject_pool[relation_id].remove(subject_id)

            # remove the object from the object pool of the relation
            if (
                relation_id in self.relation_id_to_object_pool
                and object_id in self.relation_id_to_object_pool[relation_id]
            ):
                self.relation_id_to_object_pool[relation_id].remove(object_id)

            # remove the relation from all three distributions if either the subject or the object pool of the relation
            # is empty
            if (
                (relation_id in self.relation_id_to_subject_pool and not self.relation_id_to_subject_pool[relation_id])
                or (relation_id in self.relation_id_to_object_pool and not self.relation_id_to_object_pool[relation_id])
            ):
                number_of_removed_occurrences += self.adjusted_relations_distribution_copy.get(relation_id, 0)
                self.delete_relation_entries(relation_id)

        # TODO: why is this scaling needed
        # scale the step parameter according the number of removed occurrences
        if number_of_removed_occurrences > 0:
            remaining_facts = float(self.num_synthetic_facts - self.fact_count) * self.step
            self.step *= (remaining_facts - number_of_removed_occurrences) / remaining_facts

    def adjust_relation_distribution_with_pools(self) -> Dict[int, float]:
        """
        Adjust the number of occurrences of relations with a functionality score of 1.0, i.e. an entity can only be the
        subject on one relation of that type. If only 5 of the synthesized entities are possible subjects, but the
        relation has 10 occurrences in the distribution, the number of occurrences is set to 5 instead.
        :return: the adjusted relation distribution
        """
        # the number of removed occurrences from the relation distribution
        number_removed_occurrences = 0

        # iterate over all functionalities
        for relation_id in self.functionalities.keys():

            # if the relation has a functionality of 1.0 (it has a subject pool) and still exists in the pruned
            # relation distribution, adjust the number of occurrences of the relation in the pruned distribution
            if (
                relation_id in self.adjusted_relations_distribution_copy
                and relation_id in self.relation_id_to_subject_pool
            ):
                num_subjects_in_pool = len(self.relation_id_to_subject_pool[relation_id])
                # the number of occurrences of the relation subtracted by the number of possible subject entities
                diff = self.adjusted_relations_distribution_copy[relation_id] - num_subjects_in_pool

                # if the number of occurrences is higher than the number of possible subjects, set the number of
                # occurrences to the (lower) number of possible subjects
                if diff > 0:
                    self.adjusted_relations_distribution_copy[relation_id] = num_subjects_in_pool
                    number_removed_occurrences += diff

        # iterate over all relations with an inverse functionality of 1.0 and for which valid object entities were
        # synthesized
        for relation_id in self.relation_id_to_object_pool.keys():

            # if the relation still exists in the pruned relations distribution, adjust the number of occurrences of the
            # relation in the pruned distribution
            if relation_id in self.adjusted_relations_distribution_copy:
                num_objects_in_pool = len(self.relation_id_to_object_pool[relation_id])
                # the number of occurrences of the relation subtracted by the number of possible object entities
                diff = self.adjusted_relations_distribution_copy[relation_id] - num_objects_in_pool

                # if the number of occurrences is higher than the number of possible objects, set the number of
                # occurrences to the (lower) number of possible objects
                if diff > 0:
                    self.adjusted_relations_distribution_copy[relation_id] = num_objects_in_pool
                    number_removed_occurrences += diff

            # TODO: is the indentation correct? (it was copied correctly)
            if number_removed_occurrences > 0:
                # the fact count is incremented in KBModelM1::add_fact
                # TODO: why this scaling
                remaining_facts = float(self.num_synthetic_facts - self.fact_count) * self.step
                self.step *= (remaining_facts - number_removed_occurrences) / remaining_facts

        return self.adjusted_relations_distribution_copy

    def functionality_entity_id_pools(self) -> Dict[int, Set[int]]:
        """
        Create a dictionary pointing from a relation id to a set of entities that have this relation as outgoing edge.
        These sets (pools) are only created for relations with a functionality score of 1, meaning that every entity
        that has an outgoing edge of this relation, has only one outgoing edge of this relation (not more).
        :return: the dictionary pointing from relation ids to sets of entity ids
        """
        # dictionary pointing from relation id to a set of entity ids that have this relation as outgoing edge
        relation_id_to_domain_entity_id_pool = {}

        # iterate over the functionalities of every relation
        for relation_id, functionality in self.functionalities.items():
            # if every entity that has an outgoing edge of this relation has only one of those edges and the relation
            # is still in the pruned domain distribution add the entities to the entity pool
            if functionality == 1 and relation_id in self.relation_domain_distribution_copy:
                relation_id_to_domain_entity_id_pool[relation_id] = set()

                # add all entities that are of a multitype that is a domain of this relation to the entity pool
                for domain in self.relation_domain_distribution_copy[relation_id]:
                    if domain in self.entity_types_to_entity_ids.keys():
                        # set of the entity ids that have the multitype of this domain
                        new_entities = set(self.entity_types_to_entity_ids[domain])
                        # add the ids of these new entities to the existing pool of entity ids
                        new_entities = relation_id_to_domain_entity_id_pool[relation_id].union(new_entities)
                        relation_id_to_domain_entity_id_pool[relation_id] = new_entities

        return relation_id_to_domain_entity_id_pool

    def inverse_functionality_entity_id_pools(self) -> Dict[int, Set[int]]:
        """
        Create a dictionary pointing from a relation id to a set of entities that have this relation as incoming edge.
        These sets (pools) are only created for relations with an inverse functionality score of 1, meaning that every
        entity that has an incoming edge of this relation, has only one incoming edge of this relation (not more).
        :return: the dictionary pointing from relation ids to sets of entity ids
        """
        # dictionary pointing from relation id to a set of entity ids that have this relation as incoming edge
        relation_id_to_range_entity_id_pool = {}

        # iterate over the inverse functionalities of every relation
        for relation_id, inverse_functionality in self.inverse_functionalities.items():
            # if every entity that has an incoming edge of this relation has only one of those edges and the relation
            # is still in the pruned range distribution add the entities to the entity pool
            if inverse_functionality == 1 and relation_id in self.relation_range_distribution_copy:
                relation_id_to_range_entity_id_pool[relation_id] = set()

                # add all entities that are of a multitype that is a range of this relation to the entity pool
                for domain in self.relation_domain_distribution_copy[relation_id]:
                    for range_of_domain in self.relation_range_distribution_copy[relation_id][domain]:
                        if range_of_domain in self.entity_types_to_entity_ids.keys():
                            # set of the entity ids that have the multitype of this range
                            new_entities = set(self.entity_types_to_entity_ids[range_of_domain])
                            # add the ids of these new entities to the existing pool of entity ids
                            new_entities = relation_id_to_range_entity_id_pool[relation_id].union(new_entities)
                            relation_id_to_range_entity_id_pool[relation_id] = new_entities

        return relation_id_to_range_entity_id_pool

    def synthesize(self,
                   size: float = 1.0,
                   number_of_entities: int = None,
                   number_of_edges: int = None,
                   debug: bool = False,
                   pca: bool = True):
        """
        Synthesizes a knowledge base of a given size either determined by a scaling factor or a static number of
        entities and edges.
        :param size: scale of the synthesized knowledge base (e.g., 1.0 means it should have the same size as the KB
                     the model was trained on, 2.0 means it should have twice the size)
        :param number_of_entities: the number of entities the synthesized graph should have. If not set, this number
                                   will be determined by the number of entities on which the model was trained and the
                                   size parameter
        :param number_of_edges: the number of edges (facts) the synthesized graph should have. If not set this number
                                will be determined by the number of edges on which the model was trained and the size
                                parameter
        :param debug: boolean if logging should be on debug level
        :param pca: boolean if PCA (partial completeness assumption) should be used. This parameter is not used
        :return: the synthesized graph as rdf graph object
        """
        print("Synthesizing HORN model")

        level = logging.DEBUG if debug else logging.INFO
        self.logger = create_logger(level, name="kbgen")
        self.synth_time_logger = create_logger(level, name="synth_time")
        self.query_time = create_logger(level, name="query_logger")

        self.pca = pca

        self.start_counts()

        self.logger.info(f"Original dataset contains {self.entity_count} entities and {self.edge_count} facts")

        # scale the entity and edge count by the given size
        self.step = 1.0 / float(size)
        self.num_synthetic_entities = int(self.entity_count / self.step)
        self.num_synthetic_facts = int(self.edge_count / self.step)

        # overwrite dynamic sizes with static sizes if they were set
        if number_of_entities is not None:
            self.num_synthetic_entities = number_of_entities
        if number_of_edges is not None:
            self.num_synthetic_facts = number_of_edges

        # TODO: why is step scaled here?
        self.step /= 1.1

        self.logger.info(f"Synthetic dataset contains {self.num_synthetic_entities} entities and "
                         f"{self.num_synthetic_facts} facts")

        # TODO: what is this for?
        # a list of relation ids of the quadratic relations
        quadratic_relations = self.check_for_quadratic_relations()
        adjusted_relations_distribution = self.adjust_quadratic_relation_distributions(
            deepcopy(self.relation_distribution),
            quadratic_relations)

        # inverse relation to id dictionary (integer relation id points to relation URI)
        self.relation_id_to_relation = {relation_id: relation for relation, relation_id in self.relation_to_id.items()}

        # print URIs of the quadratic relations
        quadratic_relation_uris = [self.relation_id_to_relation[relation_id] for relation_id in quadratic_relations]
        self.logger.debug(f"Quadratic relations: {quadratic_relation_uris}")

        # initialize graph with the triples defining the entity types, relations and type hierarchies
        graph = self.initialize_graph_with_metadata()

        # insert synthetic entities whose entity types mirror the distribution of entity types
        graph = self.insert_synthetic_entities(graph, self.num_synthetic_entities)

        self.logger.debug("Copying distributions...")
        self.adjusted_relations_distribution_copy = deepcopy(adjusted_relations_distribution)
        self.relation_domain_distribution_copy = deepcopy(self.relation_domain_distribution)
        self.relation_range_distribution_copy = deepcopy(self.relation_range_distribution)

        self.logger.debug("Pruning distributions for domains and ranges of types without instances...")
        self.prune_distributions()

        # create entity id pools for domains and ranges of every relation with a score of 1 (i.e., no node has 2 or more
        # incoming or 2 or more outgoing edges of this relation)
        # outgoing edge => stored in domain; incoming edge => stored in range
        # these pools contain the entity ids of entities that have the relation as outgoing or incoming edge
        self.relation_id_to_subject_pool = self.functionality_entity_id_pools()
        self.relation_id_to_object_pool = self.inverse_functionality_entity_id_pools()

        # adjust the relation distribution with the subject and object pools
        self.adjusted_relations_distribution_copy = self.adjust_relation_distribution_with_pools()

        # dictionary pointing from a relation id to a dictionary pointing from object types (multitypes) to a set of
        # subject ids that are impossible to add as a fact with that object type (i.e., facts from that subject with
        # that relation id can not exist with the given object type
        self.saturated_subject_ids = {relation_id: {} for relation_id in range(self.relation_count)}

        # dictionary pointing from a relation id to a dictionary pointing from subject types (multitypes) to a set of
        # objects ids that are impossible to add as a fact with that subject type (i.e., facts from that object with
        # that relation id can not exist with the given subject type
        self.saturated_object_ids = {relation_id: {} for relation_id in range(self.relation_count)}

        # debug output of the pool sizes
        num_subjects_for_relations = {relation_id: len(entity_ids)
                                      for relation_id, entity_ids in self.relation_id_to_subject_pool.items()}
        num_objects_for_relations = {relation_id: len(entity_ids)
                                     for relation_id, entity_ids in self.relation_id_to_object_pool.items()}
        self.logger.debug(f"subject pool = {num_subjects_for_relations}")
        self.logger.debug(f"object pool = {num_objects_for_relations}")

        self.logger.info("Synthesizing facts...")
        # progress bar
        self.progress_bar = tqdm.tqdm(total=self.num_synthetic_facts)
        # start delta used for time logging
        self.start_t = datetime.datetime.now()

        # initialize variables for pool sizes
        number_of_possible_subjects = 0
        number_of_possible_objects = 0

        # add facts until the desired fact count is reached and the adjusted relation distribution is not empty
        while self.fact_count < self.num_synthetic_facts and self.adjusted_relations_distribution_copy:
            # select the relation type of the new fact
            relation_id = choice(
                list(self.adjusted_relations_distribution_copy.keys()),
                replace=True,
                p=normalize(self.adjusted_relations_distribution_copy.values()))
            selected_subject_type = None
            selected_object_type = None
            subject_id = -1
            object_id = -1
            self.logger.debug(f"Selected relation {relation_id} = {self.relation_id_to_relation[relation_id]}")

            # only continue if the relation is still in the pruned relation and the domain distribution is not empty
            if (
                relation_id in self.relation_domain_distribution_copy
                and self.relation_domain_distribution_copy[relation_id]
            ):
                # select the multi type of the subject
                selected_subject_type = choice(
                    list(self.relation_domain_distribution_copy[relation_id].keys()),
                    replace=True,
                    p=normalize(self.relation_domain_distribution_copy[relation_id].values()))

                # entity ids of all synthetic entities that have the selected multi type
                possible_subject_entities = set(self.entity_types_to_entity_ids[selected_subject_type])

                # if the relation has a functionality score of 1.0 (i.e., it has a subject pool) choose only
                # the possible entities that are also in the subject pool
                if relation_id in self.relation_id_to_subject_pool:
                    pool_subjects = self.relation_id_to_subject_pool[relation_id]
                    possible_subject_entities = possible_subject_entities.intersection(pool_subjects)

                number_of_possible_subjects = len(possible_subject_entities)
                self.logger.debug(f"Selected subject type {selected_subject_type} with {number_of_possible_subjects}"
                                  f"entities in subject pool")

                # only continue if the relation has a non-empty pool of possible subjects and there is a range
                # distribution for the selected subject type
                if (
                    number_of_possible_subjects > 0
                    and selected_subject_type in self.relation_range_distribution_copy[relation_id]
                    and self.relation_range_distribution_copy[relation_id][selected_subject_type]
                ):
                    # select the multi type of the object
                    selected_object_type = choice(
                        list(self.relation_range_distribution_copy[relation_id][selected_subject_type].keys()),
                        replace=True,
                        p=normalize(self.relation_range_distribution_copy[relation_id][selected_subject_type].values()))

                    # entity ids of all synthetic entities that have the selected multi type
                    possible_object_entities = set(self.entity_types_to_entity_ids[selected_object_type])

                    # if the relation has an inverse functionality score of 1.0 (i.e., it has a subject pool) choose
                    # only the possible entities that are also in the object pool
                    if relation_id in self.relation_id_to_object_pool:
                        pool_objects = self.relation_id_to_object_pool[relation_id]
                        possible_object_entities = possible_object_entities.intersection(pool_objects)

                    # if the selected subject type was already added to the list of saturated object ids, remove all
                    # objects of that list from the list of possible object entities
                    if selected_subject_type in self.saturated_object_ids[relation_id]:
                        # these objects were not able to be added with the current relation and the selected subject
                        # type
                        saturated_objects = self.saturated_object_ids[relation_id][selected_subject_type]
                        possible_object_entities = possible_object_entities - saturated_objects

                    # if the selected object type was already added to the list of saturated subject ids, remove all
                    # subjects of that list from the list of possible subject entities
                    if selected_object_type in self.saturated_subject_ids[relation_id]:
                        # these subjects were not able to be added with the current relation and the selected object
                        # type
                        saturated_subjects = self.saturated_subject_ids[relation_id][selected_object_type]
                        possible_subject_entities = possible_subject_entities - saturated_subjects

                    # ensures non-reflexiveness by removing subject id from objects pool
                    if not self.relation_id_to_reflexiveness[relation_id] and subject_id in possible_object_entities:
                        possible_object_entities.remove(subject_id)

                    number_of_possible_subjects = len(possible_subject_entities)
                    number_of_possible_objects = len(possible_object_entities)
                    self.logger.debug(
                        f"Selected object type {selected_object_type} with {number_of_possible_objects}"
                        f"entities in object pool")

                    # only continue if the relation has non-empty pools of possible subjects and objects
                    if number_of_possible_objects > 0 and number_of_possible_subjects > 0:

                        # choose a subject from the pool of possible subjects
                        # TODO: what does this model do (implemented in emi model)
                        subject_model = self.select_subject_model(relation_id, selected_subject_type)

                        selected_subject_index = self.select_instance(number_of_possible_subjects, subject_model)
                        subject_id = list(possible_subject_entities)[selected_subject_index]

                        # choose an object from the pool of possible objects
                        # TODO: what does this model do (implemented in emi model)
                        object_model = self.select_object_model(relation_id,
                                                                selected_subject_type,
                                                                selected_object_type)
                        selected_object_index = self.select_instance(number_of_possible_objects, object_model)
                        object_id = list(possible_object_entities)[selected_object_index]

                        # create the actual entities
                        subject_uri = URIEntity(subject_id).uri
                        object_uri = URIEntity(object_id).uri
                        relation_uri = URIRelation(relation_id).uri
                        self.logger.debug(f"Trying to add triple ({subject_id}, {relation_id}, {object_id})")
                        try:
                            subject_offset = 0
                            object_offset = 0

                            # choose randomly with a 50:50 chance for if or else
                            if random.random() < 0.5:

                                # continue as long as the fact was not added and the object_offset stays in its bounds
                                while (
                                    not self.add_fact(graph, (subject_uri, relation_uri, object_uri))
                                    and object_offset < len(possible_object_entities)
                                ):
                                    # try adding the fact with the next possible object
                                    object_offset += 1
                                    new_object_index = selected_object_index + object_offset
                                    new_object_index = new_object_index % len(possible_object_entities)
                                    object_id = list(possible_object_entities)[new_object_index]
                                    object_uri = URIEntity(object_id).uri

                                # the fact could not be added
                                if object_offset >= len(possible_object_entities):
                                    # it's impossible to add facts for relation_id with given subject "subject_id"
                                    # add the impossible object type to the collection of saturated subject ids
                                    if selected_object_type not in self.saturated_subject_ids[relation_id]:
                                        self.saturated_subject_ids[relation_id][selected_object_type] = set()

                                    # the current subject can't be added with the current relation and the selected
                                    # object type
                                    self.saturated_subject_ids[relation_id][selected_object_type].add(subject_id)
                            else:
                                # this line should be unnecessary
                                object_id = list(possible_object_entities)[selected_object_index]
                                object_uri = URIEntity(object_id).uri

                                # continue as long as the fact was not added and the subject_offset stays in its bounds
                                while (
                                    not self.add_fact(graph, (subject_uri, relation_uri, object_uri))
                                    and subject_offset < len(possible_subject_entities)
                                ):
                                    # try adding fact with the next possible subjects
                                    subject_offset += 1
                                    new_subect_index = selected_subject_index + subject_offset
                                    new_subect_index = new_subect_index % len(possible_subject_entities)
                                    subject_id = list(possible_subject_entities)[new_subect_index]
                                    subject_uri = URIEntity(subject_id).uri

                                # the fact could not be added
                                if subject_offset >= len(possible_subject_entities):
                                    # it's impossible to add facts for relation_id with given object "object_id"
                                    # add the impossible subject type to the collection of saturated object ids
                                    if selected_subject_type not in self.saturated_object_ids[relation_id]:
                                        self.saturated_object_ids[relation_id][selected_subject_type] = set()

                                    # the current object can't be added with the current relation and the selected
                                    # subject type
                                    self.saturated_object_ids[relation_id][selected_subject_type].add(object_id)

                            # if both object and subject were successfully selected (i.e., the offsets stayed in
                            # their bounds
                            if (
                                object_offset < len(possible_object_entities)
                                and subject_offset < len(possible_subject_entities)
                            ):
                                # reduce the distributions of the selected types since the fact was added successfully
                                self.update_distributions(relation_id, selected_subject_type, selected_object_type)

                                # test if any rules cause new facts to be added, after the current fact was added, and
                                # add them
                                self.produce_rules(graph, relation_id, (subject_uri, relation_uri, object_uri))

                                # remove the subject and object from the respective pools of the relation after the fact
                                # was added successfully
                                # also remove the relation from the distributions if one of the pools ends up empty
                                self.update_pools(relation_id, subject_id, object_id)

                                # skip the final cleanup of the distributions and pools and logging
                                continue

                        except KeyError:
                            self.number_of_key_errors += 1
                    else:
                        self.number_of_empty_pool_occurrences += 1
                else:
                    self.number_of_facts_with_empty_distributions += 1
            else:
                self.number_of_facts_with_empty_distributions += 1

            # delete the chosen domain from the domain distribution because its subject pool is empty
            if number_of_possible_subjects == 0:
                self.logger.debug(f"Deleting relation domain entries for relation {relation_id} and "
                                  f"subject type {selected_subject_type}")
                self.delete_relation_domain_entries(relation_id, selected_subject_type)

            # delete the chosen range from the range distribution because its object pool is empty
            if number_of_possible_objects == 0:
                self.logger.debug(
                    f"Deleting relation range entries for relation {relation_id}, subject type {selected_subject_type} "
                    f"and object type {selected_object_type}")
                self.delete_relation_domain_range_entries(relation_id, selected_subject_type, selected_object_type)

            # TODO: why here
            self.delete_empty_entries(relation_id, selected_subject_type)

            # TODO: why here
            self.update_pools(relation_id, subject_id, object_id)

            self.print_synthesis_details()

        self.logger.debug(f"Synthesized {self.fact_count} facts of {self.num_synthetic_facts} synthesized facts")
        return graph
