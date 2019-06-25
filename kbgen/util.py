import codecs
import sys
from typing import List, Iterable, ValuesView, Union, Set
import csv

import numpy as np
import logging

from rdflib import Graph
from tqdm import tqdm


def create_logger(level=logging.INFO, name="kbgen", log_to_console=False) -> logging.Logger:
    """
    Creates a logger that logs to a file and optionally to stdout as well.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if log_to_console:
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
    print(f"Saving graph to tsv file {output_file}")
    with codecs.open(output_file, "wb", encoding="utf-8") as file:
        for subject, predicate, rdf_object in tqdm(graph):
            file.write(f"{subject}\t{predicate}\t{rdf_object}\n")


def normalize(values_view: Union[ValuesView[float], List[float]]) -> Iterable[float]:
    """
    Normalizes the input float values by their sum. The result represents how often the associated value appears in the
    distribution.
    :param values_view: view of a list of float values that represent the number of occurrences of the associated values
    :return: normalized vector in which every value is in [0, 1] and represents the frequency of the appearence of the
             associated value
    """
    values = list(values_view)
    np_values = np.array(values).astype(float)
    np_values /= sum(np_values)
    return np_values.tolist()


def read_csv(file_name: str,
             delimiter: str = ",",
             quotechar: str = '"',
             has_header: bool = True,
             columns_to_keep: Set[str] = None) -> list:
    lines = []
    csv.field_size_limit(sys.maxsize)
    with open(file_name, 'r') as file:
        data: csv.DictReader = csv.reader(file, delimiter=delimiter, quotechar=quotechar)
        if has_header:
            header = next(data)
            for row in data:
                padded_row = row
                for _ in range(len(header) - len(row)):
                    padded_row.append(None)
                named_row = {name: padded_row[index] for index, name in enumerate(header)}
                if columns_to_keep:
                    named_row = {key: value for key, value in named_row.items() if key in columns_to_keep}
                lines.append(named_row)
        else:
            lines = [row for row in data]
    return lines
