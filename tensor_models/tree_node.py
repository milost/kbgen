class TreeNode(object):
    def __init__(self, node_id, name, parent=None, children=None):
        self.node_id = node_id
        self.name = name
        self.parent = parent
        self.children = children or []

    def __str__(self):
        return self.name.__str__()

    def print_tree(self, tab="", pool=None):
        print(tab + self.name)
        for child in self.children:
            if self != child and self == child.parent:
                child.print_tree(tab + "\t")

    def get_all_parents(self):
        parents = []
        nd = self
        while nd.parent is not None:
            parents.append(nd.parent)
            nd = nd.parent
        return parents

    def get_all_parent_ids(self):
        return [p.id for p in self.get_all_parents()]
