import warnings
from math import floor
from typing import Dict, List

import numpy as np
from scipy.stats import pareto, zipf, powerlaw, uniform, expon, foldnorm, truncexpon, truncnorm

from load_tensor_tools import load_graph_npz, load_types_npz
from kbgen.kb_models import KBModelM1
from util_models import MultiType

models_dict = {"pareto": pareto,
               "zipf": zipf,
               "powerlaw": powerlaw,
               "uniform": uniform,
               "expon": expon,
               "foldnorm": foldnorm,
               "truncexpon": truncexpon,
               "truncnorm": truncnorm}


class KBModelEMi(KBModelM1):
    def __init__(self, model, dist_subjects, dist_objects):
        assert isinstance(model, KBModelM1)
        self.base_model = model
        for k, v in model.__dict__.items():
            self.__dict__[k] = v
        self.dist_subjects = dist_subjects
        self.dist_objects = dist_objects

    def select_instance(self, n, model=None):
        if model is not None:
            d, params = model
            model = models_dict[d]
            arg = params[:-2]
            scale = params[-1]
            idx = model.rvs(size=1, loc=0, scale=min([n, scale / self.step]), *arg)[0]
            return int(floor(min([idx, n - 1])))
        else:
            return super(KBModelEMi, self).select_instance(n)

    def select_subject_model(self, relation_id, relation_domain):
        if hasattr(self, "inv_functionalities") and self.inv_functionalities[relation_id] == 1:
            return None
        if relation_domain in self.dist_subjects[relation_id]:
            return self.dist_subjects[relation_id][relation_domain]
        else:
            return None

    def select_object_model(self, relation_id, relation_domain, relation_range):
        if hasattr(self, "functionalities") and self.functionalities[relation_id] == 1:
            return None
        if relation_range in self.dist_objects[relation_id]:
            return self.dist_objects[relation_id][relation_range]
        else:
            return None

    @classmethod
    def learn_best_dist_model(cls, type_distribution: List[int], distributions: list = None):
        """
        TODO
        :param type_distribution: the number of occurrences of entities of this type in a specific relation either as
                                  subject or as object
        :param distributions: TODO
        :return: TODO
        """
        distributions = distributions or [truncexpon]

        # TODO: why
        type_distribution.sort(reverse=True)

        x = range(len(type_distribution))
        y = np.array(type_distribution)
        y = (y.astype(float) / np.sum(y)).tolist()

        uniform_y = np.full(len(type_distribution), 1.0 / len(type_distribution), dtype=float)
        best_sse = np.sum(np.power(y - uniform_y, 2.0))
        best_model = None
        if best_sse > 0:
            # iterate over the different kinds of distributions that should be calculated (i.e., truncexpon)
            for distribution in distributions:
                warnings.filterwarnings('ignore')

                # fit distribution to data

                # TODO: why
                if sum(type_distribution) > 1000:
                    data_sample = np.random.choice(x, 1000, replace=True, p=y)
                    params = distribution.fit(data_sample, loc=0, scale=len(type_distribution))
                else:
                    data = []
                    for index, number_of_occurrences in enumerate(type_distribution):
                        data = data + [index] * int(number_of_occurrences)
                    params = distribution.fit(data, loc=0, scale=len(type_distribution))

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=0, scale=len(type_distribution), *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                if sse < best_sse:
                    best_model = (distribution.__class__.__name__.replace("_gen", ""), params)
                    best_sse = sse

        return best_model

    @classmethod
    def generate_from_tensor_and_model(cls, m_model: KBModelM1, input_path: str, debug: bool = False) -> 'KBModelEMi':
        """
        Generates an e model from the specified tensor file and M1, M2 or M3 model.
        :param m_model: the previously generated M1/M2/M3 model
        :param input_path: path to the numpy tensor file
        :param debug: boolean indicating if the logging level is on debug
        :return: an eMi model generated from the tensor file and M1/M2/M3 model
        """
        # the list of adjacency matrices of the object property relations created in load_tensor
        relation_adjaceny_matrices = load_graph_npz(input_path)

        # the entity type adjacency matrix created in load_tensor
        entity_types = load_types_npz(input_path).tocsr()

        # number of different relations (object properties)
        number_of_relations = len(relation_adjaceny_matrices)

        # points from relation id to a dictionary pointing from a multi type to a list of occurrences
        # these occurrences are the number of occurrences of the single instances of this multi type as subject in
        # relations of this relation type
        subject_distribution: Dict[int, Dict[MultiType, List[int]]] = [{} for _ in range(number_of_relations)]

        # points from relation id to a dictionary pointing from a multi type to a list of occurrences
        # these occurrences are the number of occurrences of the single instances of this multi type as object in
        # relations of this relation type
        object_distribution: Dict[int, Dict[MultiType, List[int]]] = [{} for _ in range(number_of_relations)]

        # TODO: what is truncexpon
        distributions = [truncexpon]

        for relation_id in range(number_of_relations):
            # adjacency matrix of the current relation type
            adjacency_matrix = relation_adjaceny_matrices[relation_id]

            # dictionary that points from a subject id to the number of occurrences it has in relations
            subject_sum = {}

            # dictionary that points from an object id to the number of occurrences it has in relations
            object_sum = {}

            # iterate over the column indicies in the adjacency matrix (= object ids that occur in relations)
            for object_id in adjacency_matrix.col:
                if object_id not in object_sum:
                    object_sum[object_id] = 1
                else:
                    object_sum[object_id] = object_sum[object_id] + 1

            # iterate over the row indicies in the adjacency matrix (= subject ids that occur in relations)
            for subject_id in adjacency_matrix.row:
                if subject_id not in subject_sum:
                    subject_sum[subject_id] = 1
                else:
                    subject_sum[subject_id] = subject_sum[subject_id] + 1

            # aggregate the number of occurrences of instances of entity types in relations as subjects
            for subject_id, count in subject_sum.items():
                # get the multi type of the subject and assert that it has a distribution
                subject_multi_type = MultiType(entity_types[subject_id].indices)
                assert subject_multi_type in m_model.entity_type_distribution

                # if the multi type does not exist in the subject distribution add it
                if subject_multi_type not in subject_distribution[relation_id]:
                    subject_distribution[relation_id][subject_multi_type] = []

                # append the number of occurrences of this instance of the multi type to the multi type in the subject
                # distribution
                subject_distribution[relation_id][subject_multi_type].append(count)

            # aggregate the number of occurrences of instances of entity types in relations as objects
            for object_id, count in object_sum.items():
                # get the multi type of the object and assert that it has a distribution
                object_multi_type = MultiType(entity_types[object_id].indices)
                assert object_multi_type in m_model.entity_type_distribution

                # if the multi type does not exist in the object distribution add it
                if object_multi_type not in object_distribution[relation_id]:
                    object_distribution[relation_id][object_multi_type] = []

                # append the number of occurrences of this instance of the multi type to the multi type in the object
                # distribution
                object_distribution[relation_id][object_multi_type].append(count)

        # TODO
        models_subjects = [{} for _ in range(number_of_relations)]
        models_objects = [{} for _ in range(number_of_relations)]

        # create
        for relation_id in range(number_of_relations):
            # TODO
            for multi_type, type_distribution in subject_distribution[relation_id].items():
                subject_distribution_model = cls.learn_best_dist_model(type_distribution, distributions)
                models_subjects[relation_id][multi_type] = subject_distribution_model

            # TODO
            for multi_type, type_distribution in object_distribution[relation_id].items():
                object_distribution_model = cls.learn_best_dist_model(type_distribution, distributions)
                models_objects[relation_id][multi_type] = object_distribution_model

        return KBModelEMi(m_model, models_subjects, models_objects)

    def synthesize(self, size=1.0, number_of_entities=None, number_of_edges=None, debug=False, pca=True):
        return self.base_model.synthesize(size, number_of_entities, number_of_edges, debug, pca)
