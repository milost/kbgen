from multiprocessing import Queue

from .tree_node import TreeNode


class DAGNode(object):
    def __init__(self, node_id, name, parents=None, children=None):
        self.node_id = node_id
        self.name = name
        self.parents = parents or []
        self.children = children or []

    def __str__(self):
        return self.name.__str__()

    def print_tree(self, tab="", pool=None):
        print(tab + self.name)
        for child in self.children:
            if self != child and self in child.parents:
                child.print_tree(tab + "\t")

    def get_all_parents(self):
        parents = set()
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            nd = queue.get()
            for p in nd.parents:
                if p not in parents:
                    parents.add(p)
                    queue.put(p)
        return parents

    def get_all_parent_ids(self):
        return [p.node_id for p in self.get_all_parents()]

    def to_tree(self):
        tree_node = TreeNode(self.node_id, self.name)
        tree_node.children = [c.to_tree for c in self.children]
        tree_node.parent = min(self.parents)
        return tree_node
