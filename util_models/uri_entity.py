from rdflib import URIRef


class URIEntity(object):
    prefix = "http://dws.uni-mannheim.de/synthesized/Entity_"

    def __init__(self, entity_id: int):
        self.uri = URIRef(self.prefix + str(entity_id))
        self.id = entity_id

    def __eq__(self, other):
        return isinstance(other,self.__class__) and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.uri.__str__()

    @classmethod
    def extract_id(cls, uri):
        if type(uri) == URIRef or type(uri) == cls:
            uri = uri.__str__()
        assert uri.startswith(cls.prefix)
        id = int(uri[uri.rindex("_") + 1:])
        return cls(id)
