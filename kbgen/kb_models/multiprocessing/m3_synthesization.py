from datetime import datetime
from logging import Logger
from multiprocessing import Process, Queue
from typing import List, Tuple

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
        self.aggregate_processes: List[Process] = None

    def split_process_number(self) -> Tuple[int, int]:
        if self.num_processes < 2:
            raise RuntimeError(f"Not enough processes for M3 synthesization ({self.num_processes})")

        aggregate_num_processes = max(1, int(self.num_processes * 0.2))
        fact_num_processes = self.num_processes - aggregate_num_processes
        return fact_num_processes, aggregate_num_processes

    def synthesize(self, size: float = 1.0) -> Graph:
        print("Synthesizing M3 model")
        print("Initializing graph...")
        graph: Graph = self.model.initialize_synthesization(size=size)
        self.logger = self.model.logger

        # create worker processes
        fact_queue = Queue()
        aggregate_queue = Queue()

        threshold = 1000
        num_fact_processes, num_aggregate_processes = self.split_process_number()
        print(f"Spawning {num_fact_processes} fact and {num_aggregate_processes} aggregate processes with threshold "
              f"of {threshold}")

        self.processes = self.create_processes(num_processes=num_fact_processes,
                                               process_type=M3FactGenerationProcess,
                                               model=self.model,
                                               result_queue=fact_queue)
        self.aggregate_processes = self.create_processes(num_processes=num_aggregate_processes,
                                                         process_type=M3AggregateProcess,
                                                         fact_queue=fact_queue,
                                                         result_queue=aggregate_queue,
                                                         threshold=threshold)

        print("Synthesizing facts in parallel")
        # set variables needed for progress logging
        self.model.start_t = datetime.now()
        progress_bar = tqdm(total=self.model.num_synthetic_facts)

        # start synthesization process
        self.start_processes(self.processes)
        self.start_processes(self.aggregate_processes)
        result = set()
        fact_count = 0
        while fact_count < self.model.num_synthetic_facts:
            subset = aggregate_queue.get(block=True)
            old_count = len(result)
            result = result.union(subset)
            new_count = len(result) - old_count
            if new_count:
                fact_count += new_count
                progress_bar.update(new_count)

        self.kill_processes(self.processes)
        self.kill_processes(self.aggregate_processes)

        print("Adding synthesized facts to graph...")
        for fact in tqdm(result):
            graph.add(fact)
        print()
        return graph


class M3AggregateProcess(Process):
    def __init__(self, fact_queue: Queue, result_queue: Queue, threshold: int):
        super(M3AggregateProcess, self).__init__()
        self.fact_queue = fact_queue
        self.result_queue = result_queue
        self.threshold = threshold
        self.result = set()

    def run(self):
        while True:
            fact = self.fact_queue.get(block=True)
            self.result.add(fact)
            if len(self.result) >= self.threshold:
                self.result_queue.put(self.result)
                self.result = set()


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

