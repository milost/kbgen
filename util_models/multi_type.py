class MultiType(object):
    """
    Represents a set of multiple entity types that an entity is an instance of.
    """
    def __init__(self, types_list):
        self.types = set(types_list)

    def __eq__(self, other):
        return self.types == other.types

    def __hash__(self):
        return self.types.__hash__()

    def __str__(self):
        return str(self.types)
