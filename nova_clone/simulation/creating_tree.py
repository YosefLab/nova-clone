import random
from typing import Dict, List, Set, Tuple
import numpy as np

def create_barcode_library(
        library_size: int,
        size_of_barcodes: int = 14,
        homoplasy_sigma: float = 0.0,
) -> List[str]:
    """
    Create a barcode library: unique barcodes (colonies), optionally replicated by a lognormal
    copy number (amplification), to mimic the real barcode library used for integration

    Args:
        library_size: number of unique barcodes in the library
        size_of_barcodes: size of the barcodes
        homoplasy_sigma: unevenness of barcode amplification, 0 gives equal copies for all barcodes,
            if 0 each barcode appears once (no replicates, no homoplasy).

    Returns:
        list of barcode molecules in the library, shuffled
    """
    if library_size > 4 ** size_of_barcodes:
        unique_barcodes = ["".join(random.choices("ATCG", k=size_of_barcodes)) for _ in range(library_size)]
    else:
        unique_barcodes: Set[str] = set()
        while len(unique_barcodes) < library_size:
            unique_barcodes.add("".join(random.choices("ATCG", k=size_of_barcodes)))
    unique_barcodes = list(unique_barcodes)
    if homoplasy_sigma == 0:
        barcode_library = unique_barcodes
    else:
        copy_numbers = np.random.lognormal(mean=0, sigma=homoplasy_sigma, size=library_size)
        copy_numbers = np.maximum(np.round(copy_numbers), 1).astype(int)
        barcode_library: List[str] = []
        for barcode, copies in zip(unique_barcodes, copy_numbers):
            barcode_library.extend([barcode] * copies)
    random.shuffle(barcode_library)
    return barcode_library


def add_new_barcodes(
        number_of_barcodes: int,
        barcode_library: List[str],
) -> List[str]:
    """
    Pull barcodes for a clone from the barcode library, pulled barcode molecules are deleted from the library

    Args:
        number_of_barcodes: number of barcodes to pull
        barcode_library: list of barcode molecules in the library (shuffled)

    Returns:
        list of new barcodes
    """
    if number_of_barcodes > len(barcode_library):
        raise ValueError("barcode library is too small for the number of integrations")
    barcodes: List[str] = []
    for _ in range(number_of_barcodes):
        barcodes.append(barcode_library.pop())
    return barcodes


def create_clone_names(
        superclone_number: int, 
        number_of_generations: int
) -> List[str]:
    """
    Create clone names for a superclone

    Args:
        superclone_number: superclone index number
        number_of_generations: number of generations in superclone

    Returns:
        list of clone names
    """
    total_number_of_clones = 2 ** (number_of_generations + 1) - 1
    clone_names: List[str] = []
    for i in range(total_number_of_clones):
        clone_names.append(f"superclone_{superclone_number}_clone_{i}")
    return clone_names


def generate_ordered_indices(
    current_index: int,
    current_generation: int,
    number_of_generations: int,
    pre_order_indices: List[int] = [],
):
    """
    Order the indices of barcodes numbers according to recursive binary tree

    Args:
        current_index: current index in the tree
        current_generation: current generation in the tree
        number_of_generations: total number of generations in the tree
        pre_order_indices: list of ordered indices

    Returns:
        list of ordered indices
    """
    if current_generation >= number_of_generations:
        return
    pre_order_indices.append(current_index)
    for i in (1, 2):
        child_index = 2 * current_index + i
        generate_ordered_indices(
            child_index,
            current_generation + 1,
            number_of_generations,
            pre_order_indices,
        )
    return pre_order_indices


def calculate_number_of_barcodes_per_superclone(
    number_of_generations: int,
    alpha: float = 1,
    minP: float = 0.01,
    maxP: float = 0.9,
    order_type: str = "depth",
    decay_speed: float = 1,
    sample_or_deterministic: str = "sample",
    max_barcodes_per_clone: int = 10,
) -> List[int]:
    """
    Calculate the number of barcodes that are added to the clone according to probability calculated by rate function

    Args:
        number_of_generations: number of generations in the superclone
        max_barcodes_per_clone: number of possible barcode integrations per clone
        alpha: increasing or decreasing rate of integration, if positive increasing, if negative decreasing
        minP: minimum value of probability
        maxP: maximum value of probability
        order_type: depth or breadth output order, depth is for recursive binary tree, breadth is for breadth first search
        decay_speed: speed of decay
        sample_or_deterministic: sample or deterministic integration number

    Returns:
        list of number of barcodes for each clone in the superclone
    """
    x = np.linspace(0, 1, number_of_generations)
    x = x ** (1 / decay_speed)
    if alpha > 0:
        values = minP + x * (maxP - minP) * alpha
    else:
        values = maxP + x * (maxP - minP) * alpha
    integration_numbers = []
    for generation, integration_ratio in enumerate(values):
        for cell_in_generation in range(2**generation):
            current_number_integations = 0
            if sample_or_deterministic == "sample":
                for _ in range(max_barcodes_per_clone):
                    if random.random() < integration_ratio:
                        current_number_integations += 1
            elif sample_or_deterministic == "deterministic":
                current_number_integations = round(max_barcodes_per_clone * integration_ratio)
            else:
                raise ValueError(
                    f"invalid sample_or_deterministic: {sample_or_deterministic}"
                )
            integration_numbers.append(current_number_integations)
    if order_type == "depth":
        pre_order_indices = generate_ordered_indices(
            current_index=0,
            current_generation=0,
            number_of_generations=number_of_generations,
            pre_order_indices=[],
        )
        ordered_sequence_integration_numbers = [
            integration_numbers[i] for i in pre_order_indices
        ]
        return ordered_sequence_integration_numbers
    elif order_type == "breadth":
        return integration_numbers
    else:
        raise ValueError(f"invalid order_type:{order_type}")


def calculate_epigenetic_silencing_barcodes_perc(
        generation: int,
        percentage_of_barcodes_silenced_total: float = 0.1,
) -> float:
    """
    Calculate the percentage of barcodes that are silenced in the superclone

    Args:
        generation: number of generation in superclone
        percentage_of_barcodes_silenced: percentage of barcodes that are silenced in total

    Returns:
        percentage of barcodes that are silenced in each generation
    """
    percentage_of_barcodes_remain = 1 - percentage_of_barcodes_silenced_total
    percentage_of_barcodes_remain_per_clone = percentage_of_barcodes_remain ** (1/(generation))
    percentage_of_barcodes_silenced_per_clone = 1 - percentage_of_barcodes_remain_per_clone
    return percentage_of_barcodes_silenced_per_clone


def remove_silenced_barcodes(
    barcodes: List[str], 
    percentage_of_barcodes_silenced_per_clone: float
) -> List[str]:
    """
    Remove silenced barcodes from the list of barcodes

    Args:
        barcodes: list of barcodes before silencing
        percentage_of_barcodes_silenced: percentage of barcodes that are silenced

    Returns:
        list of un-silenced barcodes
    """
    for i in range(len(barcodes)):
        if random.random() < percentage_of_barcodes_silenced_per_clone:
            barcodes[i] = None
    return [barcode for barcode in barcodes if barcode is not None]
    

class CloneNode:
    def __init__(
        self,
        clone_name: str,
        generation: int,
        clones_barcode_number: int,
        percentage_of_barcodes_silenced_per_clone: float,
        inherited_barcodes: List[str] = [],
        barcode_library: List[str] = [],
        size_of_barcodes: int = 14,
    ):
        self.barcodes = None
        self.clones_barcode_number = clones_barcode_number
        self.inherited_barcodes = inherited_barcodes
        self.size_of_barcodes = size_of_barcodes
        self.generation = generation
        self.name = clone_name
        self.childs: List[CloneNode] = []
        self.percentage_of_barcodes_silenced_per_clone = percentage_of_barcodes_silenced_per_clone
        self.barcode_library = barcode_library

    def add_barcodes(self) -> None:
        """
        Add new barcodes to the inherited barcodes of the clone
        """
        new_barcodes = add_new_barcodes(self.clones_barcode_number, self.barcode_library)
        self.barcodes = self.inherited_barcodes + new_barcodes

    def silence_barcodes(self) -> None:
        """
        Silence barcodes in the clone and return the un-silenced barcodes
        """
        barcodes_after_epi_silencing = remove_silenced_barcodes(
                                        barcodes=self.barcodes, 
                                        percentage_of_barcodes_silenced_per_clone=self.percentage_of_barcodes_silenced_per_clone)
        self.barcodes = barcodes_after_epi_silencing

    def add_child(self, child: "CloneNode") -> None:
        """
        Add a child (type: CloneNode) to the clone
        """
        self.childs.append(child)


class SuperClone:
    def __init__(
        self,
        superclone_number: int,
        number_of_generations: int,
        total_barcodes_per_superclone: int,
        barcode_library: List[str] = [],
        size_of_barcodes: int = 14,
        alpha: float = -1,
        decay_speed: float = 1,
        percentage_of_barcodes_silenced_total: int = 0.1,
        minP: float = 0.1,
        maxP: float = 0.9,
    ):
        self._superclone_number = superclone_number
        self._number_of_generations = number_of_generations
        self.size_of_barcodes = size_of_barcodes
        self.barcode_library = barcode_library
        self.minP = minP
        self.maxP = maxP
        self.clones_names = create_clone_names(
            self._superclone_number, self._number_of_generations
        )
        self.alpha = alpha
        self.decay_speed = decay_speed
        self.total_barcodes_per_superclone = total_barcodes_per_superclone
        self.max_barcodes_per_clone = total_barcodes_per_superclone // number_of_generations
        self.clones_barcodes_numbers = calculate_number_of_barcodes_per_superclone(
            number_of_generations=self._number_of_generations,
            max_barcodes_per_clone=self.max_barcodes_per_clone,
            decay_speed=self.decay_speed,
            alpha=self.alpha,
            minP=self.minP,
            maxP=self.maxP,
            sample_or_deterministic="sample",
        )
        self.current_clone_index = 0
        self.percentage_of_barcodes_silenced_total = percentage_of_barcodes_silenced_total
        self.percentage_of_barcodes_silenced_per_clone = calculate_epigenetic_silencing_barcodes_perc(
            generation=self._number_of_generations,
            percentage_of_barcodes_silenced_total=self.percentage_of_barcodes_silenced_total,
        )

    def generate_tree(
        self, current: CloneNode | None = None, current_generation: int = 1
    ) -> CloneNode:
        """
        Generate a superclone tree made of CloneNode objects

        Args:
            current: current node in the tree
            current_generation: current generation of the tree
        """
        if current_generation > self._number_of_generations:
            # end of tree
            return

        if current == None:
            # root node
            self.root = CloneNode(
                clone_name=self.clones_names[self.current_clone_index],
                clones_barcode_number=self.clones_barcodes_numbers[
                    self.current_clone_index
                ],
                size_of_barcodes=self.size_of_barcodes,
                barcode_library=self.barcode_library,
                generation=current_generation,
                percentage_of_barcodes_silenced_per_clone=self.percentage_of_barcodes_silenced_per_clone,
            )
            self.current_clone_index += 1
            self.root.add_barcodes()
            current = self.root
            current_generation += 1

        for _ in range(2):
            # generate children nodes
            child = CloneNode(
                clone_name=self.clones_names[self.current_clone_index],
                clones_barcode_number=self.clones_barcodes_numbers[
                    self.current_clone_index
                ],
                size_of_barcodes=self.size_of_barcodes,
                barcode_library=self.barcode_library,
                inherited_barcodes=current.barcodes,
                generation=current_generation,
                percentage_of_barcodes_silenced_per_clone=self.percentage_of_barcodes_silenced_per_clone,
            )
            self.current_clone_index += 1
            child.add_barcodes()
            child.silence_barcodes()
            current.add_child(child)
            self.generate_tree(current=child, current_generation=current_generation + 1)

        return current


def print_clone_tree(node: CloneNode | None, indent=0) -> None:
    """
    Print the superclone tree
    """
    if node is None:
        return
    print(" " * indent + f"- {node.name} {node.generation}")
    print(" " * (indent + 2) + f"barcodes: {node.barcodes}")
    for child in node.childs:
        print_clone_tree(child, indent + 4)


def extract_tree_hierarchy_clones_barcodes(
    root,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str], Set[str]]:
    """
    Extract information from the hierarchical tree structure

    Returns:
        clone_hierarchy: dictionary of clone hierarchy
        clone_barcodes: dictionary of clone barcodes
        clones: list of clone names
    """
    clone_hierarchy: Dict[str, List[str]] = {}
    clone_barcodes: Dict[str, List[str]] = {}
    clones: List[str] = []
    all_barcodes: List[str] = []

    def _traverse(node):
        if not node:
            return
        clones.append(node.name)
        clone_barcodes[node.name] = node.barcodes
        clone_hierarchy[node.name] = [child.name for child in node.childs]
        all_barcodes.extend(node.barcodes)
        for child in node.childs:
            _traverse(child)

    def get_leaf_subtree(node_name, clone_hierarchy):
        children = clone_hierarchy.get(node_name, [])
        if not children:
            return [node_name]
        leaves = []
        for child in children:
            leaves.extend(get_leaf_subtree(child, clone_hierarchy))
        return leaves

    _traverse(root)
    all_barcodes = set(all_barcodes)

    subtree_combinations = []
    for node in clones:
        subtree = get_leaf_subtree(node, clone_hierarchy)
        subtree_combinations.append(sorted(subtree))

    return clone_hierarchy, clone_barcodes, clones, all_barcodes, subtree_combinations