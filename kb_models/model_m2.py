from typing import Dict

from load_tensor_tools import loadGraphNpz
from kb_models.model_m1 import KBModelM1
from rdflib import Graph
from numpy.random import choice
from util_models import URIEntity, URIRelation
from util import normalize, create_logger
import logging
from scipy.sparse import csr_matrix
import tqdm
import datetime


class KBModelM2(KBModelM1):
    """
    Model based on the KBModelM1 and containing nonreflexiveness, functionality and inverse funcitonality of relations
    - To avoid violations of the three relation characteristics we keep pools of subjects and entities available for
     each relation. Whenever a fact is generated the subject and the object are removed from their repective pools.
    """

    def __init__(self,
                 naive_model: KBModelM1,
                 functionalities: Dict[int, float],
                 inv_functionalities: Dict[int, float],
                 rel_densities: Dict[int, float],
                 rel_distinct_subjs: Dict[int, int],
                 rel_distinct_objs: Dict[int, int],
                 reflexiveness: Dict[int, bool]):
        """
        Creates an M2 model with the passed data.
        :param naive_model: the previously built M1 model (on the same data)
        :param functionalities:
        :param inv_functionalities:
        :param rel_densities:
        :param rel_distinct_subjs:
        :param rel_distinct_objs:
        :param reflexiveness:
        """
        assert type(naive_model) == KBModelM1
        for k, v in naive_model.__dict__.items():
            self.__dict__[k] = v
        self.functionalities = functionalities
        self.inv_functionalities = inv_functionalities
        self.rel_densities = rel_densities
        self.rel_distinct_subjs = rel_distinct_subjs
        self.rel_distinct_objs = rel_distinct_objs
        self.reflexiveness = reflexiveness

    def print_synthesis_details(self):
        super(KBModelM2, self).print_synthesis_details()
        self.logger.debug("violate func: %d" % self.count_violate_functionality_facts)
        self.logger.debug("violate invfunc: %d" % self.count_violate_inv_functionality_facts)
        self.logger.debug("violate nonreflex: %d" % self.count_violate_non_reflexiveness_facts)

    def valid_functionality(self, g, fact):
        try:
            g.triples((fact[0], fact[1], None)).next()
            self.count_violate_functionality_facts += 1
            return False
        except StopIteration:
            return True

    def valid_inv_functionality(self, g, fact):
        try:
            g.triples((None, fact[1], fact[2])).next()
            self.count_violate_inv_functionality_facts += 1
            return False
        except StopIteration:
            return True

    def valid_reflexiveness(self, g, fact):
        if fact[0] != fact[2]:
            return True
        else:
            self.count_violate_non_reflexiveness_facts += 1
            return False

    def functional_rels_subj_pool(self):
        func_rels_subj_pool = {}
        for r, func in self.functionalities.items():
            if func == 1 and r in self.d_dr:
                func_rels_subj_pool[r] = set()
                for domain in self.d_dr[r]:
                    if domain in self.entity_types_to_entity_ids.keys():
                        func_rels_subj_pool[r] = func_rels_subj_pool[r].union(set(self.entity_types_to_entity_ids[domain]))
        return func_rels_subj_pool

    def invfunctional_rels_subj_pool(self):
        invfunc_rels_subj_pool = {}
        for r, inv_func in self.inv_functionalities.items():
            if inv_func == 1 and r in self.d_rdr:
                invfunc_rels_subj_pool[r] = set()
                for domain in self.d_dr[r]:
                    for range in self.d_rdr[r][domain]:
                        if range in self.entity_types_to_entity_ids.keys():
                            invfunc_rels_subj_pool[r] = invfunc_rels_subj_pool[r].union(set(self.entity_types_to_entity_ids[range]))
        return invfunc_rels_subj_pool

    def synthesize(self, size=1, number_of_entities=None, number_of_edges=None, debug=False, pca=True):
        print("Synthesizing OWL model")

        level = logging.DEBUG if debug else logging.INFO
        self.logger = create_logger(level, name="kbgen")
        self.synth_time_logger = create_logger(level, name="synth_time")

        self.step = 1.0 / float(size)
        synthetic_entities = int(self.entity_count / self.step)
        synthetic_facts = int(self.edge_count / self.step)
        if number_of_entities is not None:
            synthetic_entities = number_of_entities
        if number_of_edges is not None:
            synthetic_facts = number_of_edges

        g = Graph()

        quadratic_relations = self.check_for_quadratic_relations()
        adjusted_dist_relations = self.adjust_quadratic_relation_distributions(self.relation_distribution, quadratic_relations)

        types = range(self.entity_type_count)
        relations = range(self.relation_count)

        g = self.synthesize_entity_types(g, self.entity_type_count)
        g = self.synthesize_relations(g, self.relation_count)
        g = self.synthesize_schema(g)
        g, entities_types = self.synthesize_entities(g, synthetic_entities)
        self.synthetic_id_to_type = {k: v for v in entities_types.keys() for k in entities_types[v]}
        self.entity_types_to_entity_ids = entities_types

        self.logger.info("synthesizing facts")
        dist_relations = normalize(adjusted_dist_relations.values())

        dist_domains_relation = {}
        for rel in relations:
            dist_domains_relation[rel] = normalize(self.relation_domain_distribution[rel].values())

        dist_ranges_domain_relation = {}
        for rel in relations:
            dist_ranges_domain_relation[rel] = {}
            for domain_i in self.relation_range_distribution[rel].keys():
                dist_ranges_domain_relation[rel][domain_i] = normalize(
                    self.relation_range_distribution[rel][domain_i].values())

        self.count_facts = 0
        self.count_already_existent_facts = 0
        self.count_violate_functionality_facts = 0
        self.count_violate_inv_functionality_facts = 0
        self.count_violate_non_reflexiveness_facts = 0

        self.logger.info(str(synthetic_facts) + " facts to be synthesized")
        self.progress_bar = tqdm.tqdm(total=synthetic_facts)
        self.start_t = datetime.datetime.now()
        while self.count_facts < synthetic_facts:
            rel_i = choice(self.relation_distribution.keys(), 1, True, dist_relations)[0]
            if rel_i in self.relation_distribution.keys():
                # rel_i = self.dist_relations.keys().index(rel_uri)
                # rel_i = i
                domain_i = choice(self.relation_domain_distribution[rel_i].keys(), 1, p=dist_domains_relation[rel_i])
                domain_i = domain_i[0]
                n_entities_domain = len(entities_types[domain_i])

                range_i = choice(self.relation_range_distribution[rel_i][domain_i].keys(),
                                 1, p=dist_ranges_domain_relation[rel_i][domain_i])
                range_i = range_i[0]
                n_entities_range = len(entities_types[range_i])

                if n_entities_domain > 0 and n_entities_range > 0:
                    subject_model = self.select_subject_model(rel_i, domain_i)
                    object_model = self.select_object_model(rel_i, domain_i, range_i)

                    object_i = entities_types[range_i][self.select_instance(n_entities_range, object_model)]
                    subject_i = entities_types[domain_i][self.select_instance(n_entities_domain, subject_model)]

                    p_i = URIRelation(rel_i).uri
                    s_i = URIEntity(subject_i).uri
                    o_i = URIEntity(object_i).uri

                    fact = (s_i, p_i, o_i)
                    if (self.functionalities[rel_i] > 1 or self.valid_functionality(g, fact)) and \
                            (self.inv_functionalities[rel_i] > 1 or self.valid_inv_functionality(g, fact)) and \
                            (self.reflexiveness[rel_i] or self.valid_reflexiveness(g, fact)):
                        self.add_fact(g, fact)

        self.print_synthesis_details()
        self.logger.info("synthesized facts = %d from %d" % (self.count_facts, synthetic_facts))
        return g

    @staticmethod
    def generate_entities_stats(g):
        pass

    @staticmethod
    def generate_from_tensor(naive_model: KBModelM1, input_path: str, debug=False) -> 'KBModelM2':
        """
        Generates an M2 model from the specified tensor file and M1 model.
        :param naive_model: the previously generated M1 model
        :param input_path: path to the numpy tensor file
        :param debug: boolean indicating if the logging level is on debug
        :return: an M2 model generated from the tensor file and M1 model
        """
        # the list of adjacency matrices of the object property relations created in load_tensor
        relation_adjaceny_matrices = loadGraphNpz(input_path)

        # dictionary pointing from a relation id to the functionality score
        # this functionality says how often an entity, that appears as subject with this relation, has this relation
        # on average (as subject)
        # Therefore the lowest score would be 1.0 and the highest score would be the number of objects that
        # have this relation
        functionalities = {}

        # dictionary pointing from a relation id to the inverse functionality score
        # this inverse functionality says how often an entity, that appears as object with this relation, has this
        # relation on average (as object)
        # Therefore the lowest score would be 1.0 and the highest score would be the number of objects that have this
        # relation
        inverse_functionalities = {}

        # dictionary pointing from a relation id to a boolean indicating if this relation has any reflexive edges
        relation_id_to_reflexiveness = {}

        # dictionary pointing from a relation id to its density
        # the density says how clustered the edges are around specific nodes
        # the lowest possible density is sqrt(num_edges_of_relation_type) and it means that every edge is between
        # a different subject and object than the other edges, i.e. an entity can appear only once as subject and once
        # as object for this relation type
        # the highest possible density is 1.0 and it means that the edges of this relation type have the minimum amount
        # of entities as subjects and objects (e.g., we have 1000 relations they start at 1 entity (subject) and go to
        # 1000 other entities (objects)
        relation_id_to_density = {}

        # dictionary pointing from relation id to a count of how many different subjects appear with this relation
        relation_id_to_distinct_subjects = {}

        # dictionary pointing from relation id to a count of how many different objects appear with this relation
        relation_id_to_distinct_objects = {}

        # iterate over the adjacency matrix of each relation type
        # the index of each matrix is the id of the relation type
        # the rows of each matrix contain the ids of the subject of the relation
        # the columns of each matrix contain the ids of the object of the relation
        for relation_id, adjacency_matrix in enumerate(relation_adjaceny_matrices):
            # how often an entity id appears as subject in a relation
            # axis = 1 sums the row values
            subject_frequencies = csr_matrix(adjacency_matrix.sum(axis=1))

            # how often an entity id appears as object in a relation
            # axis = 0 sums the column values
            object_frequencies = csr_matrix(adjacency_matrix.sum(axis=0))

            # the number of different (distinct) entities that appear as subject/object
            num_distinct_subjects = subject_frequencies.nnz
            num_distinct_objects = object_frequencies.nnz
            relation_id_to_distinct_subjects[relation_id] = num_distinct_subjects
            relation_id_to_distinct_objects[relation_id] = num_distinct_objects

            # the number of edges of this relation type divided by the
            relation_id_to_density[relation_id] = float(adjacency_matrix.nnz) / (num_distinct_subjects * num_distinct_objects)

            # the total number of relations divided by the number of different entities that appear as subject
            # the result is how often an entity that appears as subject actually appears as subject
            # a score of 2 for the relation "has car" would mean that an entity that has a relation "has car" has on
            # average two cars
            # a score of 1 means that every subject that appears in this relation has this relation exactly once
            functionalities[relation_id] = float(subject_frequencies.sum()) / num_distinct_subjects

            # the total number of relations divided by the number of different entities that appear as object
            inverse_functionalities[relation_id] = float(object_frequencies.sum()) / num_distinct_objects

            # True if any reflexive edge exists in the adjacency matrix
            relation_id_to_reflexiveness[relation_id] = adjacency_matrix.diagonal().any()

        owl_model = KBModelM2(
            naive_model=naive_model,
            functionalities=functionalities,
            inv_functionalities=inverse_functionalities,
            rel_densities=relation_id_to_density,
            rel_distinct_subjs=relation_id_to_distinct_subjects,
            rel_distinct_objs=relation_id_to_distinct_objects,
            reflexiveness=relation_id_to_reflexiveness)

        return owl_model
