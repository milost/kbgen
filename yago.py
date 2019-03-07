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


def load_yago_dates(file_name: str, graph: Graph):
    print(f"Reading yago dates from {file_name}")
    graph_size = len(graph)

    # get the total number of lines
    with open(file_name) as yago_file:
        num_lines = len(yago_file.readlines())

    birth_date_str = "wasBornOnDate"
    num_unparseable_dates = 0
    with open(file_name) as yago_file:
        for line in tqdm(yago_file, total=num_lines):
            # skip non birth date lines
            if birth_date_str not in line:
                continue
            _, person, predicate, date = line.strip().split("\t")[:4]
            subject = URIRef(clean_uri(person))
            predicate = URIRef(clean_uri(predicate))

            # remove the data type and quotes from the date value
            object = "".join([char for char in date if char not in "\"^xsd:date"])
            object = Literal(clean_uri(object), datatype=XSD.date)

            # rdflib couldn't parse the date (i.e., negative dates)
            if object.value is None:
                num_unparseable_dates += 1
                continue

            graph.add((subject, predicate, object))
    print()
    print(f"Added {len(graph) - graph_size} date triples to graph")
    print(f"Skipped {num_unparseable_dates} dates due to parsing errors")
    return graph


def main():
    args = cli_args()
    graph = Graph()
    graph = load_yago_core(args.core, graph)
    date_file = args.birth_dates
    if "yago3_full" in date_file:
        graph = load_yago_dates(date_file, graph)
    else:
        graph = load_birth_dates(date_file, graph)

    output = f"yago2_default/graph.bin"
    print(f"Saving graph to {output}")
    with open(output, "wb") as graph_file:
        pickle.dump(graph, graph_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
