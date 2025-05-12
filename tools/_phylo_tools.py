
def create_maps(all_nodes, tree):
    all_branches = []
    branch_lengths = {}
    for n, node in enumerate(all_nodes):
        if node.name != tree.root.name:
            all_branches.append(node.name)
            branch_lengths[node.name] = node.branch_length
    return all_branches, branch_lengths

def get_path(spA, spB, tree):
    MRCA = tree.common_ancestor(spA,spB)
    negative_path = [node.name for node in MRCA.get_path(spA)[::-1]]
    positive_path = [node.name for node in MRCA.get_path(spB)]
    path = negative_path + positive_path
    return path
