from collections import Counter


class MapNode:
    def __init__(self, entity_type, entity_phrase, cluster_label):
        # this will allow us to efficiently and cleanly build out a map
        self.entity_type = entity_type
        self.entity_phrase = entity_phrase
        self.cluster_label = cluster_label
        self.parents = set()
        self.children = set()
        self.lookup_texts = {}
        self.negation = False

    def __hash__(self):
        return hash((self.entity_type, self.cluster_label))

    def __eq__(self, other):
        if isinstance(other, MapNode):
            return self.entity_type == other.entity_type and self.cluster_label == other.cluster_label
        return False

    def add_parent(self, parent_node):
        if (parent_node == self):
            raise Exception("Cannot add self as parent")
        elif parent_node in self.parents:
            return True
        else:
            self.parents.add(parent_node)
            parent_node.add_child(self)
            return True

    def add_child(self, child_node):
        if (child_node == self):
            raise Exception("Cannot add self as child")
        elif child_node in self.children:
            return True
        else:
            self.children.add(child_node)
            child_node.add_parent(self)
            return True

    def __str__(self):
        if len(self.lookup_texts) == 0:
            return self.entity_phrase
        else:
            return max(self.lookup_texts.items(), key=lambda x: x[1])[0]


class NodeSpace:
    def __init__(self):
        self.node_dict = {}  # stores nodes with its hash as the key

    def get_node_by_label(self, entity_type, entity_phrase, cluster_label, lookup_text=None):
        node = MapNode(entity_type, entity_phrase, str(cluster_label))
        if node in self.node_dict:
            return self.node_dict[node]
        else:
            self.node_dict[node] = node
            if lookup_text is not None:
                node.lookup_texts = lookup_text
            return node

    def get_nodes_by_label(self, entity_type, entity_phrase, cluster_pointer):
        nodes = []
        try:
            for cluster in cluster_pointer.get_cluster_ids(entity_phrase, True):
                elements = {k: v for k, v in Counter(cluster.get_elements(expand_link_only=True)).items() if not (len(k) <= 8 and k.isupper())}

                node = self.get_node_by_label(entity_type, entity_phrase, cluster, elements)
                nodes.append(node)
        except:
            def is_iterable(obj):
                try:
                    _ = iter(obj)
                    return True
                except TypeError:
                    return False

            if is_iterable(cluster_pointer):
                for cluster in cluster_pointer:
                    node = self.get_node_by_label(entity_type, entity_phrase, cluster)
                    self.node_dict[node].lookup_texts[entity_phrase] = self.node_dict[node].lookup_texts.get(
                        entity_phrase, 0) + 1
                    nodes.append(node)
            else:
                node = self.get_node_by_label(entity_type, entity_phrase, cluster_pointer)
                self.node_dict[node].lookup_texts[entity_phrase] = self.node_dict[node].lookup_texts.get(entity_phrase,
                                                                                                         0) + 1
                nodes.append(node)

        return nodes
