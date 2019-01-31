from typing import Dict, Tuple, List

from load_tensor_tools import load_graph_npz
from kbgen.kb_models import KBModelM1
from rdflib import Graph
from scipy.sparse import csr_matrix


class KBModelM2(KBModelM1):
    """
    Model based on the KBModelM1 and containing nonreflexiveness, functionality and inverse funcitonality of relations
    - To avoid violations of the three relation characteristics we keep pools of subjects and entities available for
     each relation. Whenever a fact is generated the subject and the object are removed from their repective pools.
    """

    def __init__(self,
                 m1_model: KBModelM1,
                 functionalities: Dict[int, float],
                 inverse_functionalities: Dict[int, float],
                 relation_id_to_density: Dict[int, float],
                 relation_id_to_distinct_subjects: Dict[int, int],
                 relation_id_to_distinct_objects: Dict[int, int],
                 relation_id_to_reflexiveness: Dict[int, bool]):
        """
        Creates an M2 model with the passed data.
        :param m1_model: the previously built M1 model (on the same data)
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
        assert isinstance(m1_model, KBModelM1), f"Model is of type {type(m1_model)} but needs to be of type KBModelM1"
        # initialize the M1 model data with the values of the passed M1 model
        super(KBModelM2, self).__init__(
            entity_type_hierarchy=m1_model.entity_type_hierarchy,
            object_property_hierarchy=m1_model.object_property_hierarchy,
            domains=m1_model.domains,
            ranges=m1_model.ranges,
            entity_count=m1_model.entity_count,
            relation_count=m1_model.relation_count,
            edge_count=m1_model.edge_count,
            entity_type_count=m1_model.entity_type_count,
            entity_type_distribution=m1_model.entity_type_distribution,
            relation_distribution=m1_model.relation_distribution,
            relation_domain_distribution=m1_model.relation_domain_distribution,
            relation_range_distribution=m1_model.relation_range_distribution,
            relation_to_id=m1_model.relation_to_id,
            entity_type_to_id=m1_model.entity_type_to_id
        )

        self.functionalities = functionalities
        self.inverse_functionalities = inverse_functionalities
        self.relation_id_to_density = relation_id_to_density
        self.relation_id_to_distinct_subjects = relation_id_to_distinct_subjects
        self.relation_id_to_distinct_objects = relation_id_to_distinct_objects
        self.relation_id_to_reflexiveness = relation_id_to_reflexiveness

        self.name = "M2 (OWL)"

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

    def is_fact_valid(self, graph: Graph, relation_id: int, fact: Tuple[str, str, str]):
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
        return valid_functionality and valid_inverse_functionality and valid_reflexiveness

    @staticmethod
    def generate_entities_stats(graph: Graph):
        pass

    @staticmethod
    def generate_from_tensor(input_path: str, debug: bool = False) -> 'KBModelM2':
        m1_model = KBModelM1.generate_from_tensor(input_path, debug)
        return KBModelM2.generate_from_tensor_and_model(m1_model, input_path, debug)

    @staticmethod
    def generate_from_tensor_and_model(naive_model: KBModelM1, input_path: str, debug: bool = False) -> 'KBModelM2':
        """
        Generates an M2 model from the specified tensor file and M1 model.
        :param naive_model: the previously generated M1 model
        :param input_path: path to the numpy tensor file
        :param debug: boolean indicating if the logging level is on debug
        :return: an M2 model generated from the tensor file and M1 model
        """
        # the list of adjacency matrices of the object property relations created in load_tensor
        relation_adjaceny_matrices = load_graph_npz(input_path)

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
            m1_model=naive_model,
            functionalities=functionalities,
            inverse_functionalities=inverse_functionalities,
            relation_id_to_density=relation_id_to_density,
            relation_id_to_distinct_subjects=relation_id_to_distinct_subjects,
            relation_id_to_distinct_objects=relation_id_to_distinct_objects,
            relation_id_to_reflexiveness=relation_id_to_reflexiveness)

        return owl_model
