from typing import Dict

import numpy as np
from scipy.sparse import csr_matrix

from kbgen import MultiType
from ..model_m1 import KBModelM1
from kbgen.load_tensor_tools import load_single_adjacency_matrix, load_types_npz, load_domains, load_ranges, \
    load_type_hierarchy, load_prop_hierarchy, load_types_dict, load_relations_dict, num_adjacency_matrices
from .interfaces import LearnProcess, ResultCollector


class M1LearnProcess(LearnProcess):
    def __init__(self, dense_entity_types: np.ndarray, multitype_index, **kwargs):
        super(M1LearnProcess, self).__init__(**kwargs)
        self.dense_entity_types = dense_entity_types
        self.multitype_index = multitype_index

    def learn_distributions(self, relation_id: int):
        """
        For an in-depth explanation take a loot at the single core implementation in the M1-Model itself.
        :param relation_id: the relation id for which the features are learned
        """
        adjacency_matrix = load_single_adjacency_matrix(self.input_dir, relation_id)
        result = M1Result(relation_id=relation_id)

        # number of edges for this relation
        # used for fact count as well as relation distribution
        result.num_edges = adjacency_matrix.nnz

        domain_distribution = {}
        range_distribution = {}

        subject_ids_row = adjacency_matrix.row
        object_ids_row = adjacency_matrix.col

        # iterate over all non-zero fields of the adjacency matrix
        # iterate over all edges that exist for the current relation
        for index in range(result.num_edges):
            # get subject and object id from the adjacency matrix
            subject_id = subject_ids_row[index]
            object_id = object_ids_row[index]
            # create multi types from the two sets of entity types for each the subject and the object

            multi_type, = self.dense_entity_types[subject_id].nonzero()
            subject_multi_type = self.multitype_index[frozenset(multi_type)]

            multi_type, = self.dense_entity_types[object_id].nonzero()
            object_multi_type = self.multitype_index[frozenset(multi_type)]

            # if the subject's multi type is not known add it to the relation domain distribution and create an
            # empty relation range distribution for that multi type
            if subject_multi_type not in domain_distribution:
                domain_distribution[subject_multi_type] = 0
                range_distribution[subject_multi_type] = {}

            # if the object's multi type is not known add it to the relation range distribution of the subject's
            # multi type
            if object_multi_type not in range_distribution[subject_multi_type]:
                range_distribution[subject_multi_type][object_multi_type] = 0

                # increment the number of occurrences of the subject's multi type in the relation domain distribution
                domain_distribution[subject_multi_type] += 1
                # increment the number of occurrences of the object's multi type in the relation range distribution
                # of the subject's multi type
                range_distribution[subject_multi_type][object_multi_type] += 1

        result.domain_distribution = domain_distribution
        result.range_distribution = range_distribution

        self.result_queue.put(result)


class M1ResultCollector(ResultCollector):
    """
    For an in-depth explanation take a loot at the single core implementation in the M1-Model itself.
    """
    def __init__(self, input_dir: str, multi_type_index: Dict[frozenset, int]):
        self.input_dir = input_dir
        self.multi_type_index = multi_type_index

        # features that will be learned before the multiprocessing
        self.entity_types = None
        self.dense_entity_types = None
        self.domains = None
        self.ranges = None
        self.entity_type_hierarchy = None
        self.object_property_hierarchy = None
        self.entity_type_to_id = None
        self.relation_to_id = None
        self.count_entities = None
        self.count_types = None
        self.count_facts = None
        self.count_relations = None
        self.entity_type_distribution = None

        self.load_data()

        # distributions and features that will be learned in parallel
        self.relation_distribution: Dict[int, int] = {}
        self.relation_domain_distribution: Dict[int, Dict[int, int]] = {}
        self.relation_range_distribution: Dict[int, Dict[int, Dict[int, int]]] = {}
        self.count_facts = 0

    def load_data(self):
        self.entity_types = load_types_npz(self.input_dir)
        self.dense_entity_types = self.entity_types.toarray()

        self.domains = load_domains(self.input_dir)
        self.ranges = load_ranges(self.input_dir)

        self.entity_type_hierarchy = load_type_hierarchy(self.input_dir)
        self.object_property_hierarchy = load_prop_hierarchy(self.input_dir)

        self.entity_type_to_id = load_types_dict(self.input_dir)
        self.relation_to_id = load_relations_dict(self.input_dir)

        # compress entity type adjacency matrix if it is not already compressed
        if not isinstance(self.entity_types, csr_matrix):
            self.entity_types = self.entity_types.tocsr()

        print("Learning entity type distributions...")
        # the entity type adjacency matrix has the dimensions num_entities x num_entity_types
        self.count_entities = self.entity_types.shape[0]
        self.count_types = self.entity_types.shape[1]

        # number of different relations (object properties)
        self.count_relations = num_adjacency_matrices(self.input_dir)

        self.entity_type_distribution = {}
        for entity_type in self.entity_types:
            entity_type_set = MultiType(entity_type.indices)

            # add the multi type to the distribution if it is new
            if entity_type_set not in self.entity_type_distribution:
                self.entity_type_distribution[entity_type_set] = 0.0

            self.entity_type_distribution[entity_type_set] += 1

    def handle_result(self, result: 'M1Result'):
        relation_id = result.relation_id
        self.count_facts += result.num_edges
        self.relation_distribution[relation_id] = result.num_edges
        self.relation_domain_distribution[relation_id] = result.domain_distribution
        self.relation_range_distribution[relation_id] = result.range_distribution

    def build_model(self) -> KBModelM1:
        return KBModelM1(entity_type_hierarchy=self.entity_type_hierarchy,
                         object_property_hierarchy=self.object_property_hierarchy,
                         domains=self.domains,
                         ranges=self.ranges,
                         entity_count=self.count_entities,
                         relation_count=self.count_relations,
                         edge_count=self.count_facts,
                         entity_type_count=self.count_types,
                         entity_type_distribution=self.entity_type_distribution,
                         relation_distribution=self.relation_distribution,
                         relation_domain_distribution=self.relation_domain_distribution,
                         relation_range_distribution=self.relation_range_distribution,
                         relation_to_id=self.relation_to_id,
                         entity_type_to_id=self.entity_type_to_id,
                         multitype_index=self.multi_type_index)


class M1Result(object):
    def __init__(self,
                 relation_id: int = None,
                 num_edges: int = None,
                 domain_distribution: Dict[MultiType, int] = None,
                 range_distribution: Dict[MultiType, Dict[MultiType, int]] = None):
        self.relation_id = relation_id
        self.num_edges = num_edges
        self.domain_distribution = domain_distribution
        self.range_distribution = range_distribution
