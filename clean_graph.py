from argparse import ArgumentParser, Namespace

from rdflib import Graph, RDF, RDFS
from tqdm import tqdm

from kbgen.util import dump_tsv


def cli_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("input", type=str, default=None, help="path to the knowledge base that will be filtered")
    return parser.parse_args()


def main():
    args = cli_args()

    graph = Graph()
    rdf_format = args.input[args.input.rindex(".") + 1:]
    print(f"Loading graph from {args.input}")
    graph.parse(args.input, format=rdf_format)

    subjects = set()

    print("Building subject index")
    for subject, predicate, object in tqdm(graph):
        if predicate != RDF.type and predicate != RDFS.subClassOf:
            subjects.add(subject)

    cleaned_graph = Graph()

    print("Building cleaned graph")
    for subject, predicate, object in tqdm(graph):
        if subject in subjects and object in subjects:
            cleaned_graph.add((subject, predicate, object))

    output = f"{args.input[:args.input.rindex('.')]}_cleaned"
    dump_tsv(cleaned_graph, f"{output}.tsv")
    print(f"Saving graph to {output}.{rdf_format}")
    cleaned_graph.serialize(f"{output}.{rdf_format}", format=rdf_format)


if __name__ == '__main__':
    main()
