from rdflib import Graph, RDF, RDFS, OWL
PATH = "ontology/databaseV7.ttl"

def compute_stats():
    g = Graph()
    g.parse(PATH, format="turtle")

    subjects = set()
    predicates = set()
    objects = set()

    for s, p, o in g:
        subjects.add(s)
        predicates.add(p)
        objects.add(o)

    classes = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class))
    properties = (
        set(g.subjects(RDF.type, RDF.Property)) |
        set(g.subjects(RDF.type, OWL.ObjectProperty)) |
        set(g.subjects(RDF.type, OWL.DatatypeProperty))
    )

    return {
        "triples": len(g),
        "subjects": len(subjects),
        "predicates": len(predicates),
        "objects": len(objects),
        "classes": len(classes),
        "properties": len(properties),
    }