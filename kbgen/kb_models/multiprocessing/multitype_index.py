from typing import Set, List

import numpy as np
from tqdm import tqdm

from kbgen.load_tensor_tools import load_single_adjacency_matrix, load_types_npz
from .interfaces import LearnProcess, ResultCollector


class MultiTypeLearnProcess(LearnProcess):
    def __init__(self, dense_entity_types: np.ndarray, **kwargs):
        super(MultiTypeLearnProcess, self).__init__(**kwargs)
        self.dense_entity_types = dense_entity_types
        self.position: int = None

    # def learn_distributions(self, relation_id: int):
    #     """
    #     For an in-depth explanation take a loot at the single core implementation in the M1-Model itself.
    #     :param relation_id: the relation id for which the features are learned
    #     """
    #     # set the tqdm position to the first input index (which will be in 1..n with n = num_processes)
    #     if self.position is None:
    #         self.position = relation_id + 1
    #
    #     adjacency_matrix = load_single_adjacency_matrix(self.input_dir, relation_id)
    #
    #     num_edges = adjacency_matrix.nnz
    #     subject_ids_row = adjacency_matrix.row
    #     object_ids_row = adjacency_matrix.col
    #
    #     distinct_multi_types = set()
    #
    #     for index in tqdm(range(num_edges), position=self.position):
    #         subject_id = subject_ids_row[index]
    #         object_id = object_ids_row[index]
    #
    #         multi_type, = self.dense_entity_types[subject_id].nonzero()
    #         distinct_multi_types.add(frozenset(multi_type))
    #
    #         multi_type, = self.dense_entity_types[object_id].nonzero()
    #         distinct_multi_types.add(frozenset(multi_type))
    #
    #     self.result_queue.put(distinct_multi_types)

    def learn_distributions(self, relation_id: int):
        """
        For an in-depth explanation take a loot at the single core implementation in the M1-Model itself.
        :param relation_id: the relation id for which the features are learned
        """
        # set the tqdm position to the first input index (which will be in 1..n with n = num_processes)
        if self.position is None:
            self.position = relation_id + 1

        adjacency_matrix = load_single_adjacency_matrix(self.input_dir, relation_id)
        print(f"max subject id: {np.max(adjacency_matrix.row)}")
        print(f"max object id: {np.max(adjacency_matrix.col)}")
        coordinates: List[tuple] = list(zip(*adjacency_matrix.toarray().nonzero()))

        distinct_multi_types = set()

        for subject_id, object_id in tqdm(coordinates, position=self.position):
            multi_type, = self.dense_entity_types[subject_id].nonzero()
            distinct_multi_types.add(frozenset(multi_type))

            multi_type, = self.dense_entity_types[object_id].nonzero()
            distinct_multi_types.add(frozenset(multi_type))

        self.result_queue.put(distinct_multi_types)


class MultiTypeResultCollector(ResultCollector):
    """
    For an in-depth explanation take a loot at the single core implementation in the M1-Model itself.
    """
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.distinct_multi_types: Set[frozenset] = set()

        entity_types = load_types_npz(self.input_dir)
        self.dense_entity_types = entity_types.toarray()

    def handle_result(self, result: Set[frozenset]):
        self.distinct_multi_types = self.distinct_multi_types.union(result)

    def build_model(self):
        multi_type_index = {}
        for multi_type in self.distinct_multi_types:
            multi_type_index[multi_type] = len(multi_type_index)

        return multi_type_index

