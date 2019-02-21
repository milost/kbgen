from datetime import datetime
from logging import Logger
from multiprocessing import Process, Queue

from numpy.random import choice
from rdflib import Graph
from tqdm import tqdm

from .interfaces import MultiProcessingTask
from ...util_models import URIEntity, URIRelation
from ...util import normalize
from ..model_m3 import KBModelM3


class M3Synthesization(MultiProcessingTask):
    def __init__(self, model: KBModelM3, num_processes: int):
        super(M3Synthesization, self).__init__(num_processes)
        self.model = model
        self.logger: Logger = None
        self.fact_queue: Queue = None
        self.process_type = M3FactGenerationProcess

    def synthesize(self, size: float = 1.0) -> Graph:
        # initialize graph and model
        graph: Graph = self.model.initialize_synthesization(size=size)
        self.logger = self.model.logger

        # create worker processes
        result_queue = Queue()
        self.processes = self.create_processes(model=self.model, result_queue=result_queue)

        print("Synthesizing M3 model in parallel")

        # set variables needed for progress logging
        self.model.start_t = datetime.now()
        self.model.progress_bar = tqdm(total=self.model.num_synthetic_facts)

        # start synthesization process
        self.start_processes()
        while self.model.fact_count < self.model.num_synthetic_facts:
            fact = result_queue.get(block=True)
            self.model.add_fact(graph, fact)

        self.kill_processes()

        return graph


class M3FactGenerationProcess(Process):
    def __init__(self, model: KBModelM3, result_queue: Queue):
        super(M3FactGenerationProcess, self).__init__()
        self.model = model
        self.result_queue = result_queue

    def run(self):
        while True:
            self.generate_fact()

    def generate_fact(self):
        relation_id = choice(
            list(self.model.adjusted_relations_distribution_copy.keys()),
            replace=True,
            p=normalize(self.model.adjusted_relations_distribution_copy.values()))

        # select the multi type of the subject
        selected_subject_type = choice(
            list(self.model.relation_domain_distribution_copy[relation_id].keys()),
            replace=True,
            p=normalize(self.model.relation_domain_distribution_copy[relation_id].values()))

        # entity ids of all synthetic entities that have the selected multi type
        possible_subject_entities = set(self.model.entity_types_to_entity_ids[selected_subject_type])

        # if the relation has a functionality score of 1.0 (i.e., it has a subject pool) choose only
        # the possible entities that are also in the subject pool
        if relation_id in self.model.relation_id_to_subject_pool:
            pool_subjects = self.model.relation_id_to_subject_pool[relation_id]
            possible_subject_entities = possible_subject_entities.intersection(pool_subjects)

        number_of_possible_subjects = len(possible_subject_entities)

        # only continue if the relation has a non-empty pool of possible subjects
        if number_of_possible_subjects > 0:
            subjects, object_type, objects = self.model.choose_object_type_and_instances(relation_id,
                                                                                         selected_subject_type,
                                                                                         possible_subject_entities)
            possible_subject_entities = subjects
            possible_object_entities = objects
            number_of_possible_subjects = len(possible_subject_entities)
            number_of_possible_objects = len(possible_object_entities)

            # only continue if the relation has non-empty pools of possible subjects and objects
            if number_of_possible_objects > 0 and number_of_possible_subjects > 0:

                # choose a random subject from the pool of possible subjects
                selected_subject_index = self.model.select_instance(number_of_possible_subjects, None)
                subject_id = list(possible_subject_entities)[selected_subject_index]

                # choose a random object from the pool of possible objects
                selected_object_index = self.model.select_instance(number_of_possible_objects, None)
                object_id = list(possible_object_entities)[selected_object_index]

                # create the actual entities
                subject_uri = URIEntity(subject_id).uri
                object_uri = URIEntity(object_id).uri
                relation_uri = URIRelation(relation_id).uri

                fact = (subject_uri, relation_uri, object_uri)

                self.result_queue.put(fact)

