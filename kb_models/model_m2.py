from typing import Dict, Tuple, List

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
                 inverse_functionalities: Dict[int, float],
                 relation_id_to_density: Dict[int, float],
                 relation_id_to_distinct_subjects: Dict[int, int],
                 relation_id_to_distinct_objects: Dict[int, int],
                 relation_id_to_reflexiveness: Dict[int, bool]):
        """
        Creates an M2 model with the passed data.
        :param naive_model: the previously built M1 model (on the same data)
        :param functionalities: points from relation id to the functionality score of this relation type. This score
                                is the average number of outgoing edges an entity has, given that it has any outgoing
                                edges of this relation type
        :param inverse_functionalities: points from relation id to the inverse functionality score of this relation
                                        type. This score is the average number of incoming edges an entity has, given
                                        that it has any incoming edges of this relation type
        :param relation_id_to_density: points from relation id to the density of this relation type, which is a metric
                                       that TODO
        :param relation_id_to_distinct_subjects: points from relation id to the number of different entities that appear
                                                 as subject for this relation type (i.e., how many different entities
                                                 have this relation type as outgoing edge)
        :param relation_id_to_distinct_objects: points from relation id to the number of different entities that appear
                                                as object for this relation type (i.e., how many different entities
                                                have this relation type as incoming edge)
        :param relation_id_to_reflexiveness: points from relation id to a boolean indicating if any reflexive edge of
                                             that relation type exists
        """
        assert type(naive_model) == KBModelM1
        # initialize the M1 model data with the values of the passed M1 model
        # TODO does the init call have side-effects that the __dict__ way does not have
        # for k, v in naive_model.__dict__.items():
        #     self.__dict__[k] = v
        super(KBModelM2, self).__init__(
            entity_type_hierarchy=naive_model.entity_type_hierarchy,
            object_property_hierarchy=naive_model.object_property_hierarchy,
            domains=naive_model.domains,
            ranges=naive_model.ranges,
            entity_count=naive_model.entity_count,
            relation_count=naive_model.relation_count,
            edge_count=naive_model.edge_count,
            entity_type_count=naive_model.entity_type_count,
            entity_type_distribution=naive_model.entity_type_distribution,
            relation_distribution=naive_model.relation_distribution,
            relation_domain_distribution=naive_model.relation_domain_distribution,
            relation_range_distribution=naive_model.relation_range_distribution,
            relation_to_id=naive_model.relation_to_id,
            entity_type_to_id=naive_model.entity_type_to_id
        )

        self.functionalities = functionalities
        self.inverse_functionalities = inverse_functionalities
        self.relation_id_to_density = relation_id_to_density
        self.relation_id_to_distinct_subjects = relation_id_to_distinct_subjects
        self.relation_id_to_distinct_objects = relation_id_to_distinct_objects
        self.relation_id_to_reflexiveness = relation_id_to_reflexiveness

        # initialized in other methods
        #
        # the number of facts that violated the functionality of 1 of a relation type
        self.num_facts_violating_functionality = 0
        # the number of facts that violated the inverse functionality of 1 of a relation type
        self.num_facts_violating_inverse_functionality = 0
        # the number of facts that violated the non reflexiveness by being reflexive
        self.num_facts_violating_non_reflexiveness = 0

    def print_synthesis_details(self):
        """
        Print the statistics of the synthesization of a knowledge base with this M2 model.
        """
        super(KBModelM2, self).print_synthesis_details()
        self.logger.debug(f"{self.num_facts_violating_functionality} facts violated functionality")
        self.logger.debug(f"{self.num_facts_violating_inverse_functionality} facts violated inverse functionality")
        self.logger.debug(f"{self.num_facts_violating_non_reflexiveness} facts violated non-reflexiveness")

    def check_for_quadratic_relations(self) -> List[int]:
        """
        A relation type is quadratic when both its functionality score (average number of outgoing edges) and inverse
        functionality score (average number of ingoing edges) are larger than 10 and its density is larger than 0.1.
        TODO: what is the meaning of this selection
        :return: list of relation ids of which each is a quadratic relation
        """
        quadratic_relations = []
        for relation_id in self.relation_id_to_density.keys():
            functionality = self.functionalities[relation_id]
            inverse_functionality = self.inverse_functionalities[relation_id]
            if functionality > 10 and inverse_functionality > 10:
                density = self.relation_id_to_density[relation_id]
                num_distinct_objects = self.relation_id_to_distinct_objects[relation_id]
                num_distinct_subjects = self.relation_id_to_distinct_subjects[relation_id]
                is_reflexive = self.relation_id_to_reflexiveness[relation_id]

                if density > 0.1:
                    self.logger.debug(f"relation {relation_id}: functionality={functionality}, "
                                      f"inverse_functionality={inverse_functionality}, "
                                      f"density={density}, "
                                      f"num_distinct_objects={num_distinct_objects}, "
                                      f"num_distinct_subjects={num_distinct_subjects}, "
                                      f"is_reflexive={is_reflexive}")
                    quadratic_relations.append(relation_id)
        return quadratic_relations

    def valid_functionality(self, graph: Graph, fact: Tuple[str, str, str]) -> bool:
        """
        Test the validity of the new fact in regards to the functionality of the fact's relation type. This method is
        only called if the functionality of the relation type equals 1, which means that an entity can only have one
        outgoing edge of the relation type (i.e., be the subject in only one fact of the relation type).
        It is tested if the subject of the new fact already has a relation of the relation type to a different entity.
        Returns True if the fact is valid, i.e. no such relation to a different object already exists, and False if it
        is not valid.
        :param graph: the synthesized graph object
        :param fact: triple of subject, relation, object
        :return: boolean indicating the validity of the new fact, i.e. False if another fact with the same subject and
                 predicate already exists
        """
        similar_relation_exists = (fact[0], fact[1], None) in graph
        # increment the counter if a similar fact already exists (True -> +1, False -> +0)
        self.num_facts_violating_functionality += similar_relation_exists
        return not similar_relation_exists

    def valid_inverse_functionality(self, graph: Graph, fact: Tuple[str, str, str]) -> bool:
        """
        Test the validity of the new fact in regards to the inverse functionality of the fact's relation type. This
        method is only called if the inverse functionality of the relation type equals 1, which means that an entity can
        only have one incoming edge of the relation type (i.e., be the object in only one fact of the relation type).
        It is tested if the object of the new fact already has a relation of the relation type to a different entity.
        Returns True if the fact is valid, i.e. no such relation to a different subject already exists, and False if it
        is not valid.
        :param graph: the synthesized graph object
        :param fact: triple of subject, relation, object
        :return: boolean indicating the validity of the new fact, i.e. False if another fact with the same predicate and
                 object already exists
        """
        similar_relation_exists = (None, fact[1], fact[2]) in graph
        # increment the counter if a similar fact already exists (True -> +1, False -> +0)
        self.num_facts_violating_inverse_functionality += similar_relation_exists
        return not similar_relation_exists

    def valid_reflexiveness(self, fact: Tuple[str, str, str]) -> bool:
        """
        Tests if the passed fact is a reflexive edge and returns True if it is not reflexive. If it is reflexive, the
        counter for facts that violate the non reflexiveness of facts is incremented and False is returned.
        :param fact: triple of subject, relation, object
        :return: True if the passed fact is not reflexive, False if it is reflexive
        """
        is_not_reflexive = fact[0] != fact[2]
        # increment the counter by the inverse boolean value (+1 if it is reflexive, +0 if is not reflexive)
        self.num_facts_violating_non_reflexiveness += not is_not_reflexive
        return is_not_reflexive

    def functional_rels_subj_pool(self):
        """
        TODO: used in M3 model
        :return:
        """
        func_rels_subj_pool = {}
        for r, func in self.functionalities.items():
            if func == 1 and r in self.d_dr:
                func_rels_subj_pool[r] = set()
                for domain in self.d_dr[r]:
                    if domain in self.entity_types_to_entity_ids.keys():
                        func_rels_subj_pool[r] = func_rels_subj_pool[r].union(set(self.entity_types_to_entity_ids[domain]))
        return func_rels_subj_pool

    def invfunctional_rels_subj_pool(self):
        """
        TODO: used in M3 model
        :return:
        """
        invfunc_rels_subj_pool = {}
        for r, inv_func in self.inverse_functionalities.items():
            if inv_func == 1 and r in self.d_rdr:
                invfunc_rels_subj_pool[r] = set()
                for domain in self.d_dr[r]:
                    for range in self.d_rdr[r][domain]:
                        if range in self.entity_types_to_entity_ids.keys():
                            invfunc_rels_subj_pool[r] = invfunc_rels_subj_pool[r].union(set(self.entity_types_to_entity_ids[range]))
        return invfunc_rels_subj_pool

    def synthesize(self,
                   size: int = 1,
                   number_of_entities: int = None,
                   number_of_edges: int = None,
                   debug: bool = False,
                   pca: bool = True) -> Graph:
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
        :param pca: boolean if PCA should be used. This parameter is not used
        :return: the synthesized graph as rdf graph object
        """
        print("Synthesizing OWL model...")

        level = logging.DEBUG if debug else logging.INFO
        self.logger = create_logger(level, name="kbgen")
        self.synth_time_logger = create_logger(level, name="synth_time")

        # scale the entity and edge count by the given size
        self.step = 1.0 / float(size)
        num_synthetic_entities = int(self.entity_count / self.step)
        num_synthetic_facts = int(self.edge_count / self.step)

        # overwrite dynamic sizes with static sizes if they were set
        if number_of_entities is not None:
            num_synthetic_entities = number_of_entities
        if number_of_edges is not None:
            num_synthetic_facts = number_of_edges

        # the synthesized graph (initialised emtpy)
        graph = Graph()

        # TODO
        quadratic_relations = self.check_for_quadratic_relations()
        adjusted_relations_distribution = self.adjust_quadratic_relation_distributions(self.relation_distribution,
                                                                                       quadratic_relations)
        # list of the ids of the synthesized entities and relations
        # entity_type_ids = range(self.entity_type_count)
        relation_ids = range(self.relation_count)

        # adds the triples that define the entity types as such (classes)
        graph = self.synthesize_entity_types(graph, self.entity_type_count)

        # adds the triples that define the relations as such (object properties)
        graph = self.synthesize_relations(graph, self.relation_count)

        # adds the triples that define the entity type and property type hierarchies and the property domains and ranges
        graph = self.synthesize_schema(graph)

        # add synthetic entities to the graph by adding the triples that define the type of relation for every entity
        # synthetic entities are assigned multi types at random (via distribution) and the type of relation is added
        # for every entity type in that multi type
        # returns the new graph and a dictionary containing the used multi types pointing to the entity ids of the
        # synthetic entities of that type
        graph, entity_types_to_entity_ids = self.synthesize_entities(graph, num_synthetic_entities)

        # reverse the dictionary so that every synthetic entity id points to its multi type
        self.synthetic_id_to_type = {synthetic_entity_id: multi_type for multi_type in entity_types_to_entity_ids.keys()
                                     for synthetic_entity_id in entity_types_to_entity_ids[multi_type]}
        self.entity_types_to_entity_ids = entity_types_to_entity_ids

        self.logger.info("Synthesizing edges/facts..")

        # normalize relation distribution
        # first sort the dictionary by the relation id (which are in [0, num_relations)) then normalize them
        sorted_values = [occurrences for relation_id, occurrences in sorted(adjusted_relations_distribution.items())]
        normalized_values = list(normalize(sorted_values))
        # dictionary of relation ids pointing to their frequency (in [0, 1])
        relation_distribution: Dict[int, float] = {}
        # add normalized relation occurrences to the distribution dictionary
        for relation_id in adjusted_relations_distribution.keys():
            relation_distribution[relation_id] = normalized_values[relation_id]

        # normalize relation domain distribution
        # this is basically just copies relation_domain_distribution and replaces the occurrences with normalized values
        relation_domain_distribution = {}
        for relation_id in relation_ids:
            normalized_distribution = normalize(self.relation_domain_distribution[relation_id].values())
            relation_domain_distribution[relation_id] = normalized_distribution

        # normalize relation range distribution
        # this is basically just copies relation_range_distribution and replaces the occurrences with normalized values
        relation_range_distribution = {}
        for relation_id in relation_ids:
            relation_range_distribution[relation_id] = {}
            for relation_domain in self.relation_range_distribution[relation_id].keys():
                relation_range_values = self.relation_range_distribution[relation_id][relation_domain].values()
                relation_range_distribution[relation_id][relation_domain] = normalize(relation_range_values)

        # counter for the number of facts in the graph
        self.fact_count = 0
        # counter of duplicate facts that were generated
        self.duplicate_fact_count = 0

        # the number of facts that violated the functionality of 1 of a relation type
        self.num_facts_violating_functionality = 0
        # the number of facts that violated the inverse functionality of 1 of a relation type
        self.num_facts_violating_inverse_functionality = 0
        # the number of facts that violated the non reflexiveness by being reflexive
        self.num_facts_violating_non_reflexiveness = 0

        self.logger.info(f"{num_synthetic_facts} facts to be synthesized")
        # progress bar
        self.progress_bar = tqdm.tqdm(total=num_synthetic_facts)
        # start delta used for time logging
        self.start_t = datetime.datetime.now()

        # repeat until enough facts are generated
        while self.fact_count < num_synthetic_facts:
            # choose a random relation type according to the relation distribution
            relation_id = choice(list(self.relation_distribution.keys()),
                                 replace=True,
                                 p=list(relation_distribution.values()))
            if relation_id in self.relation_distribution.keys():
                # rel_i = self.dist_relations.keys().index(rel_uri)
                # rel_i = i

                # select random domain of the valid domains for this relation type according to the domain distribution
                # a domain is a multi type
                relation_domain = choice(list(self.relation_domain_distribution[relation_id].keys()),
                                         p=relation_domain_distribution[relation_id])

                # the number of synthetic entities with the multi type of the selected domain
                entity_domain_count = len(entity_types_to_entity_ids[relation_domain])

                # select random range of the valid ranges for the selected domain according to the range distribution
                # a range is a multi type
                relation_range = choice(list(self.relation_range_distribution[relation_id][relation_domain].keys()),
                                        p=relation_range_distribution[relation_id][relation_domain])

                # the number of synthetic entities with the multi type of the selected range
                entity_range_count = len(entity_types_to_entity_ids[relation_range])

                # only continue if there exist synthetic entities with the correct multi type ofr the domain as well as
                # the range
                if entity_domain_count > 0 and entity_range_count > 0:
                    # TODO
                    subject_model = self.select_subject_model(relation_id, relation_domain)
                    # TODO
                    object_model = self.select_object_model(relation_id, relation_domain, relation_range)

                    # select one of the possible entities as a subject according to the subject model
                    possible_subject_entities = entity_types_to_entity_ids[relation_domain]
                    subject_entity = possible_subject_entities[self.select_instance(entity_domain_count, subject_model)]

                    # select one of the possible entities as an object according to the object model
                    possible_object_entities = entity_types_to_entity_ids[relation_range]
                    object_entity = possible_object_entities[self.select_instance(entity_range_count, object_model)]

                    # create fact with the ids of the entities and add it to the graph
                    relation_uri = URIRelation(relation_id).uri
                    subject_uri = URIEntity(subject_entity).uri
                    object_uri = URIEntity(object_entity).uri

                    fact = (subject_uri, relation_uri, object_uri)

                    # if the functionality score is larger than one, an entity can have more than one outgoing edge
                    # of this relation type. If the score equals one, test if the subject entity already has an
                    # outgoing edge of this relation type
                    # True if the relation type allows more than one outgoing edge for an entity or if the entity has no
                    # outgoing edges of this relation type yet
                    valid_functionality = self.functionalities[relation_id] > 1 or self.valid_functionality(graph, fact)

                    # if the inverse functionality score is larger than one, an entity can have more than one incoming
                    # edge of this relation type. If the score equals one, test if the object entity already has an
                    # incoming edge of this relation type
                    # True if the relation type allows more than one incoming edge for an entity or if the entity has no
                    # incoming edges of this relation type yet
                    valid_inverse_functionality = (self.inverse_functionalities[relation_id] > 1 or
                                                   self.valid_inverse_functionality(graph, fact))

                    # if the relation does not allow reflexive edges, the method "valid_reflexiveness()" is called,
                    # which tests if the edge is reflexive
                    # True if the relation allows reflexiveness or if it doesn't and the edge is not reflexive
                    valid_reflexiveness = (self.relation_id_to_reflexiveness[relation_id] or
                                           self.valid_reflexiveness(fact))

                    # add the new fact to the graph, if it does not violate the constraints of functionality, inverse
                    # functionality and reflexiveness
                    if valid_functionality and valid_inverse_functionality and valid_reflexiveness:
                        self.add_fact(graph, fact)

        self.print_synthesis_details()
        self.logger.info(f"Synthesized facts = {self.fact_count} from {num_synthetic_facts}")
        return graph

    @staticmethod
    def generate_entities_stats(graph: Graph):
        pass

    @staticmethod
    def generate_from_tensor(input_path: str, debug: bool = False) -> 'KBModelM2':
        m1_model = KBModelM1.generate_from_tensor(input_path, debug)
        return KBModelM2.generate_from_tensor_and_model(m1_model, input_path, debug)

    @staticmethod
    def generate_from_tensor_and_model(naive_model: KBModelM1, input_path: str, debug=False) -> 'KBModelM2':
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
        # this functionality score is the average number of outgoing edges an entity has of this relation type given
        # that it has any outgoing edges of this relation type
        functionalities = {}

        # dictionary pointing from a relation id to the inverse functionality score
        # this inverse functionality score is the average number of incoming edges an entity has of this relation type
        # given that it has any incoming edges of this relation type
        inverse_functionalities = {}

        # dictionary pointing from a relation id to a boolean indicating if this relation has any reflexive edges
        relation_id_to_reflexiveness = {}

        # dictionary pointing from a relation id to its density
        # the density says how clustered the edges are around specific nodes
        # the lowest possible density is sqrt(num_edges_of_relation_type) and it means that every edge is between
        # a different subject and object than the other edges, i.e. an entity can appear only once as subject and once
        # as object for this relation type
        # the highest possible density is 1.0 and it means that the edges of this relation type have the minimum amount
        # of entities as subjects and objects needed to have that many edges (e.g., we have 1000 relations they start
        # at 1 entity (subject) and go to 1000 other entities (objects)
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

            # the number of edges of this relation type divided by the product of the number of distinct entities
            # that are subjects and the number of distinct entities that are objects of this relation
            density_score = float(adjacency_matrix.nnz) / (num_distinct_subjects * num_distinct_objects)
            relation_id_to_density[relation_id] = density_score

            # the average number of outgoing edges an entity has of this relation type given that it has any outgoing
            # edges of this relation type
            functionalities[relation_id] = float(subject_frequencies.sum()) / num_distinct_subjects

            # the average number of incoming edges an entity has of this relation type given that it has any incoming
            # edges of this relation type
            inverse_functionalities[relation_id] = float(object_frequencies.sum()) / num_distinct_objects

            # True if any reflexive edge exists in the adjacency matrix
            relation_id_to_reflexiveness[relation_id] = adjacency_matrix.diagonal().any()

        owl_model = KBModelM2(
            naive_model=naive_model,
            functionalities=functionalities,
            inverse_functionalities=inverse_functionalities,
            relation_id_to_density=relation_id_to_density,
            relation_id_to_distinct_subjects=relation_id_to_distinct_subjects,
            relation_id_to_distinct_objects=relation_id_to_distinct_objects,
            relation_id_to_reflexiveness=relation_id_to_reflexiveness)

        return owl_model
