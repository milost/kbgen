from multiprocessing import Queue
from typing import List, Dict

from tqdm import tqdm

from ..model_m1 import KBModelM1
from ..model_m2 import KBModelM2
from .interfaces import LearnProcess, ResultCollector, MultiProcessingTask
from .multitype_index import MultiTypeLearnProcess, MultiTypeResultCollector
from .m1_implementation import M1LearnProcess, M1ResultCollector
from .m2_implementation import M2LearnProcess, M2ResultCollector
from ...load_tensor_tools import num_adjacency_matrices


class ModelLoader(MultiProcessingTask):
    def __init__(self, input_dir: str, num_processes: int):
        super(ModelLoader, self).__init__(num_processes)
        self.input_dir = input_dir
        self.message: str = "Learning distributions..."

        self.process_type: type = None

        self.result_collector: ResultCollector = None

    def _load(self, **kwargs):
        """
        Builds a model given that a process type and result collector were set.
        :return: the built model
        """
        print(self.message)
        task_queue = Queue()
        result_queue = Queue()
        num_relations = num_adjacency_matrices(self.input_dir)

        self.processes = self.create_processes(input_dir=self.input_dir,
                                               task_queue=task_queue,
                                               result_queue=result_queue,
                                               **kwargs)

        print(f"Filling task queue with {num_relations} tasks")
        for relation_id in range(num_relations):
            task_queue.put(relation_id)

        self.start_processes()

        # parse the results added to the result queue
        progress_bar = tqdm(total=num_relations)
        finished = 0
        while True:
            result = result_queue.get(block=True)
            self.result_collector.handle_result(result)
            progress_bar.update(1)
            finished += 1
            if finished == num_relations:
                break

        # kill processes when we are done
        self.kill_processes()

        return self.result_collector.build_model()

    def load_m1(self):
        """
        First build the multi type index in parallel and aggregate it. Afterwards learn the distributions in parallel.

        For an in-depth explanation take a loot at the single core implementation in the M1-Model itself.
        :return: the trained m1 model
        """
        self.message = "Creating MultiType index..."
        self.result_collector = MultiTypeResultCollector(self.input_dir)
        self.process_type = MultiTypeLearnProcess
        multi_type_index: Dict[frozenset, int] = self._load(dense_entity_types=self.result_collector.dense_entity_types)

        self.message = "Learning distributions for M1 model..."
        self.result_collector = M1ResultCollector(self.input_dir, multi_type_index)
        self.process_type = M1LearnProcess
        return self._load(dense_entity_types=self.result_collector.dense_entity_types,
                          multitype_index=multi_type_index)

    def load_m2(self, m1_model: KBModelM1) -> KBModelM2:
        """
        For an in-depth explanation take a loot at the single core implementation in the M2-Model itself.
        :param m1_model the previously trained m1 model
        :return: the trained m2 model
        """
        self.message = "Learning distributions for M2 model..."
        self.result_collector = M2ResultCollector(m1_model=m1_model)
        self.process_type = M2LearnProcess
        return self._load()
