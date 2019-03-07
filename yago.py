import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

from rdflib import Graph, URIRef, Literal, XSD
from tqdm import tqdm


def cli_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("core", type=str, default=None, help="path to the yago core file")
    parser.add_argument("-b", "--birth-dates", type=str, default=None, help="path to file with birth dates")
    return parser.parse_args()


def load_yago_core(file_name: str, graph: Graph) -> Graph:
    """
    Loads the core yago file containing relations between entities into the passed graph.
    :param file_name: path to the yago core file
    :param graph: the graph to which the facts will be added
    :return: the new graph containing the yago facts
    """
    print(f"Reading graph from {file_name}")
    dirty_chars = "<>"
    with open(file_name) as yago_file:
        total = len(yago_file.readlines())
    with open(file_name) as yago_file:
        for line in tqdm(yago_file, total=total):
            triple = line.strip().split("\t")
            cleaned_triple = []
            for element in triple:
                cleaned = "".join([char for char in element if char not in dirty_chars])
                cleaned = cleaned.replace('"', "''")
                cleaned = cleaned.replace("`", "'")
                cleaned = cleaned.replace("\\", "U+005C")
                cleaned = cleaned.replace("^", "U+005E")
                cleaned_triple.append(cleaned)
            graph.add(tuple([URIRef(element) for element in cleaned_triple]))
    print()
    print(f"Created graph with {len(graph)} triples")
    return graph


def clean_uri(uri: str):
    dirty_chars = "<>"
    cleaned = "".join([char for char in uri if char not in dirty_chars])
    cleaned = cleaned.replace('"', "''")
    cleaned = cleaned.replace("`", "'")
    cleaned = cleaned.replace("\\", "U+005C")
    cleaned = cleaned.replace("^", "U+005E")
    return cleaned


def load_birth_dates(file_name: str, graph: Graph):
    print(f"Reading birth dates from {file_name}")
    relation = URIRef(Path(file_name).stem)

    # get the total number of lines
    with open(file_name) as yago_file:
        num_lines = len(yago_file.readlines())

    with open(file_name) as yago_file:
        for line in tqdm(yago_file, total=num_lines):
            _, subject, object = line.strip().split("\t")
            subject = clean_uri(subject)
            object = Literal(clean_uri(object), datatype=XSD.date)
            graph.add((URIRef(subject), relation, object))
    print()
    print(f"Created graph with {len(graph)} triples")
    return graph


def main():
    args = cli_args()
    graph = Graph()
    graph = load_yago_core(args.core, graph)
    graph = load_birth_dates(args.birth_dates, graph)

    output = f"yago/graph.bin"
    print(f"Saving graph to {output}")
    with open(output, "wb") as graph_file:
        pickle.dump(graph, graph_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
