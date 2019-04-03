import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Callable

from rdflib import Graph, URIRef, Literal, XSD
from tqdm import tqdm


def cli_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("core", type=str, default=None, help="path to the yago core file (or directory of yago3 files)")
    parser.add_argument("--birth_dates", type=str, default=None, help="path to date file")
    parser.add_argument("--yago3", dest='yago3', action='store_true', help="set if yago3 should be loaded")
    return parser.parse_args()


def load_yago2_core(file_name: str, graph: Graph) -> Graph:
    """
    Loads the core yago file containing relations between entities into the passed graph.
    :param file_name: path to the yago core file
    :param graph: the graph to which the facts will be added
    :return: the new graph containing the yago facts
    """
    return load_yago_core(file_name=file_name, graph=graph, parse_line=lambda line: line.split("\t"))


def load_yago3_core(directory: str, graph: Graph) -> Graph:
    """
    Loads the core yago file containing relations between entities into the passed graph.
    :param directory: path to directory containing the yago3 files
    :param graph: the graph to which the facts will be added
    :return: the new graph containing the yago facts
    """
    file_name = f"{directory}/yagoFacts.tsv"
    return load_yago_core(file_name=file_name, graph=graph, parse_line=lambda line: line.split("\t")[1:4])


def load_yago_core(file_name: str, graph: Graph, parse_line: Callable[[str], tuple]) -> Graph:
    """
    Loads the core yago file containing relations between entities into the passed graph.
    :param file_name: path to the yago core file
    :param graph: the graph to which the facts will be added
    :param parse_line: splits the line into the three triple elements
    :return: the new graph containing the yago facts
    """
    print(f"Reading graph from {file_name}")
    dirty_chars = "<>"
    with open(file_name) as yago_file:
        total = len(yago_file.readlines())
    with open(file_name) as yago_file:
        for line in tqdm(yago_file, total=total):
            triple = parse_line(line.strip())
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

    date_str = "OnDate>"
    num_unparseable_dates = 0
    with open(file_name) as yago_file:
        for line in tqdm(yago_file, total=num_lines):
            # skip non date lines
            if date_str not in line:
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


def load_yago_literals(file_name: str, graph: Graph):
    def parse_literal(raw_literal: str):
        yago_types = {"<degrees>", "<dollar>", "<euro>", "<g>", "</km2>", "<km2>", "<m>", "<percent>", "<s>",
                      "<yagoMonetaryValue>"}
        value, yago_type = raw_literal.split("^^")
        if yago_type in yago_types:
            value = float(value[1:-1])
            yago_type = URIRef("".join([char for char in yago_type if char not in "<>"]))
            return Literal(value, datatype=yago_type)

    print(f"Reading yago literals from {file_name}")
    graph_size = len(graph)

    # get the total number of lines
    with open(file_name) as yago_file:
        num_lines = len(yago_file.readlines())

    num_unparseable_literals = 0
    with open(file_name) as yago_file:
        for line in tqdm(yago_file, total=num_lines):
            _, subject, predicate, literal = line.strip().split("\t")[:4]
            subject = URIRef(clean_uri(subject))
            predicate = URIRef(clean_uri(predicate))
            literal = parse_literal(literal)

            # rdflib couldn't parse the date (i.e., negative dates)
            if literal.value is None:
                num_unparseable_literals += 1
                continue

            graph.add((subject, predicate, object))
    print()
    print(f"Added {len(graph) - graph_size} literals to graph")
    print(f"Skipped {num_unparseable_literals} dates due to parsing errors")
    return graph


def main():
    args = cli_args()
    graph = Graph()
    date_file = args.birth_dates
    if args.yago3:
        graph = load_yago3_core(args.core, graph)
        if date_file:
            graph = load_yago_dates(date_file, graph)
    else:
        graph = load_yago2_core(args.core, graph)
        if date_file:
            graph = load_birth_dates(date_file, graph)

    output_dir = Path("yago3" if args.yago3 else "yago2")
    output_dir.mkdir(exist_ok=True)
    output = f"{output_dir}/graph.bin"
    print(f"Saving graph to {output}")
    with open(output, "wb") as graph_file:
        pickle.dump(graph, graph_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
