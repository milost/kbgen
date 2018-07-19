import codecs
from typing import List, Iterable

import numpy as np
import logging

from rdflib import Graph


def create_logger(level=logging.INFO, name="kbgen", debug_to_console=False) -> logging.Logger:
    """
    Creates a logger that logs to a file and optionally to stdout as well.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if debug_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    file_handler = logging.FileHandler(f"{name}.log", mode='w')
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger


def dump_tsv(graph: Graph, output_file: str) -> None:
    """
    Dumps a Knowledge Graph into a .tsv file.
    :param graph: the graph object of the Knowledge Graph
    :param output_file: the file to write to
    """
    with codecs.open(output_file, "wb", encoding="utf-8") as file:
        for subject, predicate, rdf_object in graph:
            file.write(f"{subject}\t{predicate}\t{rdf_object}\n")


def normalize(l: List[float]) -> Iterable[float]:
    """
    Normalizes the input float vector by its sum.
    :param l: list of float values TODO: what do they represent
    :return: normalized TODO what are these values
    """
    l_array = np.array(list(l)).astype(float)
    l_array /= sum(l_array)
    return l_array.tolist()


################################################################################
# unused methods
################################################################################
# from copy import deepcopy
# from load_tensor_tools import get_roots
# def level_hierarchy(hier):
#     if hier is None:
#         return []
#
#     roots = get_roots(hier)
#     remaining = deepcopy(hier.keys())
#     level = roots
#     levels = []
#     while level:
#         next_level = []
#         for n in level:
#             for c in n.children:
#                 if c.node_id in remaining:
#                     next_level.append(c)
#                     remaining.remove(c.node_id)
#
#         levels.append(level)
#         level = next_level
#
#     return levels
