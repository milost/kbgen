from multiprocessing import Process
from multiprocessing.queues import Queue
from queue import Empty

from ..model_m1 import KBModelM1


class LearnProcess(Process):
    def __init__(self, input_dir: str, task_queue: Queue, result_queue: Queue):
        super(LearnProcess, self).__init__()
        self.input_dir = input_dir
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            try:
                relation_id = self.task_queue.get()
                self.learn_distributions(relation_id)
            except Empty:
                break

    def learn_distributions(self, relation_id: int):
        raise NotImplementedError


class ResultCollector(object):
    """
    Instances of this class should build the data that is learned in a model and collect the partial-results into
    the final complete result.
    They should also build the complete model out of the results.
    """

    def handle_result(self, result):
        raise NotImplementedError

    def build_model(self) -> KBModelM1:
        raise NotImplementedError
