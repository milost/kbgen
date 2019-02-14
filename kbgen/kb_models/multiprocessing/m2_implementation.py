from scipy.sparse import csr_matrix

from ..model_m1 import KBModelM1
from ..model_m2 import KBModelM2
from kbgen.load_tensor_tools import load_single_adjacency_matrix
from .interfaces import LearnProcess, ResultCollector


class M2LearnProcess(LearnProcess):
    def learn_distributions(self, relation_id: int):
        """
        Learn the funtionality, inverse functionality, reflexiveness, density and number of distinct subjects and
        objects for the current relation from the relation's adjacency matrix.
        The index of each matrix is the id of the relation type.
        The rows of each matrix contain the ids of the subject of the relation.
        The columns of each matrix contain the ids of the object of the relation.

        The result is added to the result queue.
        :param relation_id: the relation id for which the features are learned
        """
        adjacency_matrix = load_single_adjacency_matrix(self.input_dir, relation_id)
        result = M2Result(relation_id=relation_id)

        # how often an entity id appears as subject in a relation
        # axis = 1 sums the row values
        subject_frequencies = csr_matrix(adjacency_matrix.sum(axis=1))

        # how often an entity id appears as object in a relation
        # axis = 0 sums the column values
        object_frequencies = csr_matrix(adjacency_matrix.sum(axis=0))

        # the number of different (distinct) entities that appear as subject/object
        result.num_distinct_subjects = subject_frequencies.nnz
        result.num_distinct_objects = object_frequencies.nnz

        # the number of edges of this relation type divided by the product of the number of distinct entities
        # that are subjects and the number of distinct entities that are objects of this relation
        result.density = float(adjacency_matrix.nnz) / (result.num_distinct_subjects * result.num_distinct_objects)

        # the average number of outgoing edges an entity has of this relation type given that it has any outgoing
        # edges of this relation type
        result.functionality = float(subject_frequencies.sum()) / result.num_distinct_subjects

        # the average number of incoming edges an entity has of this relation type given that it has any incoming
        # edges of this relation type
        result.inverse_functionality = float(object_frequencies.sum()) / result.num_distinct_objects

        # True if any reflexive edge exists in the adjacency matrix
        result.reflexiveness = adjacency_matrix.diagonal().any()

        self.result_queue.put(result)


class M2ResultCollector(ResultCollector):
    """
    For an in-depth explanation take a loot at the single core implementation in the M2-Model itself.
    """
    def __init__(self, m1_model: KBModelM1):
        self.m1_model = m1_model

        # distributions and features that will be learned
        self.functionalities = {}
        self.inverse_functionalities = {}
        self.relation_id_to_reflexiveness = {}
        self.relation_id_to_density = {}
        self.relation_id_to_distinct_subjects = {}
        self.relation_id_to_distinct_objects = {}

    def handle_result(self, result: 'M2Result'):
        relation_id = result.relation_id
        self.relation_id_to_distinct_subjects[relation_id] = result.num_distinct_subjects
        self.relation_id_to_distinct_objects[relation_id] = result.num_distinct_objects
        self.relation_id_to_density[relation_id] = result.density
        self.functionalities[relation_id] = result.functionality
        self.inverse_functionalities[relation_id] = result.inverse_functionality
        self.relation_id_to_reflexiveness[relation_id] = result.reflexiveness

    def build_model(self) -> KBModelM2:
        return KBModelM2(m1_model=self.m1_model,
                         functionalities=self.functionalities,
                         inverse_functionalities=self.inverse_functionalities,
                         relation_id_to_density=self.relation_id_to_density,
                         relation_id_to_distinct_subjects=self.relation_id_to_distinct_subjects,
                         relation_id_to_distinct_objects=self.relation_id_to_distinct_objects,
                         relation_id_to_reflexiveness=self.relation_id_to_reflexiveness)


class M2Result(object):
    def __init__(self,
                 relation_id: int,
                 density: float = None,
                 num_distinct_subjects: int = None,
                 num_distinct_objects: int = None,
                 functionality: float = None,
                 inverse_functionality: float = None,
                 reflexiveness: bool = None):
        self.relation_id = relation_id
        self.num_distinct_subjects = num_distinct_subjects
        self.num_distinct_objects = num_distinct_objects
        self.density = density
        self.functionality = functionality
        self.inverse_functionality = inverse_functionality
        self.reflexiveness = reflexiveness
