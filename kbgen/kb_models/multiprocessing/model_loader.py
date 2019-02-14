from multiprocessing import Queue
from typing import List

from tqdm import tqdm

from kbgen import KBModelM1, KBModelM2
from .learn_process import LearnProcess, ResultCollector
from .m2_process import M2LearnProcess, M2ResultCollector
from kbgen.load_tensor_tools import num_adjacency_matrices


class ModelLoader(object):
    def __init__(self, input_dir: str, num_processes: int):
        self.input_dir = input_dir
        self.num_processes = num_processes

        self.process_type: type = None

        self.result_collector: ResultCollector = None
        self.processes: List[LearnProcess] = []

    def start_processes(self):
        for process in self.processes:
            process.start()

    def kill_processes(self):
        for process in self.processes:
            process.terminate()

    def _load(self):
        print(f"Learning distributions...")
        task_queue = Queue()
        result_queue = Queue()
        num_relations = num_adjacency_matrices(self.input_dir)
        self.processes = []

        print(f"Creating {self.num_processes} worker processes")
        for _ in range(self.num_processes):
            process: LearnProcess = self.process_type(input_dir=self.input_dir,
                                                      task_queue=task_queue,
                                                      result_queue=result_queue)
            self.processes.append(process)

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

    def load_m2(self, m1_model: KBModelM1) -> KBModelM2:
        """
        For an in-depth explanation take a loot at the single core implementation in the M2-Model itself.
        :param m1_model the previously trained m1 model
        :return: the learned m2 model
        """
        self.result_collector = M2ResultCollector(m1_model=m1_model)
        self.process_type = M2LearnProcess
        return self._load()
