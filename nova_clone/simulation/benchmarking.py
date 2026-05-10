import re
from typing import Dict, List, Set
from sklearn.metrics.cluster import adjusted_rand_score
from nova_clone.algorithm._types import CellBCType, CloneType, IntBCType
from nova_clone.qc import *


def edit_input(
    cells_clone_dict_input: Dict[CellBCType, List[CloneType]]
) -> Dict[CellBCType, List[CloneType]]:
    """
    Edit the simulation ground truth dictionary to match the algorithm output dictionary by removing the 'clone' part
    of the clone names and remaining only the 'superclone' part

    Args:
        cells_clone_dict_input: simulation ground truth

    Returns:
        superclone_cells_dict_input: dictionary of cells and superclone assignment
    """
    superclone_cells_dict_input: Dict[str, List[int]] = {}
    for k, v in cells_clone_dict_input.items():
        v = [re.sub(r"_clone_\d+", "", clone.replace("super", "")) for clone in v]
        superclone_cells_dict_input[k] = v
    return superclone_cells_dict_input

def test_edit_input():
    cells_input_exp = {'cell_1':['superclone_0_clone_0','superclone_0_clone_1'], 'cell_2':['superclone_1_clone_0']}
    expected_output = {'cell_1':['clone_0','clone_0'], 'cell_2':['clone_1']}
    output_edit_input = edit_input(cells_clone_dict_input=cells_input_exp)
    assert output_edit_input == expected_output, f"Expected {expected_output}, but got {output_edit_input}"

def add_combined_subtree_sets(
    dict_input: Dict[CloneType, List[str]],
    subtree_combinations: List[List[CloneType]],
) -> Dict[CloneType, List[str]]:
    """
    Add to the dict_input the combined cell sets for each subclone 

    Args:
        dict_input: simulation ground truth
        subtree_combinations: list of subclone combinations

    Returns:
        dict_input: dictionary of cells and clone assignment with all combinations of subclones
    """
    new_cells_clone_dict_input = {}
    for combination in subtree_combinations:
        combination_name = "_".join(combination)
        all_cells_in_combination = []
        for clone_name in combination: 
            all_cells_in_combination += dict_input[clone_name]
        new_cells_clone_dict_input[combination_name] = set(all_cells_in_combination)
    return new_cells_clone_dict_input

def test_add_combined_subtree_sets():
    combinations_exp = [['superclone_0_clone_0','superclone_0_clone_1','superclone_0_clone_2'],['superclone_0_clone_0','superclone_0_clone_1'],['superclone_0_clone_0'],['superclone_0_clone_2'],['superclone_0_clone_1']]
    cells_input_exp = {'superclone_0_clone_0':{'cell_1','cell_2','cell_3'},'superclone_0_clone_1':{'cell_4','cell_5','cell_6'},'superclone_0_clone_2':{'cell_7','cell_8','cell_9'}}
    output_add_combined_subtree_sets = add_combined_subtree_sets(
        dict_input=cells_input_exp,
        subtree_combinations=combinations_exp
    )
    expected_output = {
        'superclone_0_clone_0_superclone_0_clone_1_superclone_0_clone_2': {'cell_1', 'cell_2', 'cell_3', 'cell_4', 'cell_5', 'cell_6', 'cell_7', 'cell_8', 'cell_9'},
        'superclone_0_clone_0_superclone_0_clone_1': {'cell_1', 'cell_2', 'cell_3', 'cell_4', 'cell_5', 'cell_6'},
        'superclone_0_clone_0': {'cell_1', 'cell_2', 'cell_3'},
        'superclone_0_clone_2': {'cell_7', 'cell_8', 'cell_9'},
        'superclone_0_clone_1': {'cell_4', 'cell_5', 'cell_6'}
    }
    assert output_add_combined_subtree_sets == expected_output, f"Expected {expected_output}, but got {output_add_combined_subtree_sets}"

def edit_output(
    cells_clone_dict_output: Dict[CellBCType, List[CloneType]]
) -> Dict[CellBCType, List[CloneType]]:
    """
    Edit the algorithm output dictionary to match the simulation ground truth dictionary by removing the 'sample' part
    of the cell names and remaining only the 'cell' part

    Args:
        cells_clone_dict_output: algorithm output of cells and clone assignment

    Returns:
        rm_sample_cells_clone_dict_output: dictionary of cells and clone assignment
    """
    rm_sample_cells_clone_dict_output: Dict[str, List[int]] = {}
    for k, v in cells_clone_dict_output.items():
        k = k.replace("sample0_", "")
        rm_sample_cells_clone_dict_output[k] = v
    return rm_sample_cells_clone_dict_output

def test_edit_output():
    cells_output_exp = {'sample0_cell_1':['clone_0','clone_1'], 'sample0_cell_2':['clone_2']}
    expected_output = {'cell_1':['clone_0','clone_1'], 'cell_2':['clone_2']}
    output_edit_output = edit_output(cells_clone_dict_output=cells_output_exp)
    assert output_edit_output == expected_output, f"Expected {expected_output}, but got {output_edit_output}"

def edit_clone_to_cells_dict(
        clone_to_cells_dict: Dict[CloneType, List[CellBCType]]
    ) -> Dict[CloneType, List[CellBCType]]:
    """
    Edit the clone to cells dictionary to match the simulation ground truth dictionary by removing the 'clone' part

    Args:
        clone_to_cells_dict: dictionary of clone names and their cells

    Returns:
        edit_clone_to_cells_dict: dictionary of clone names and their cells
    """
    edit_clone_to_cells_dict = {}
    for clone, cells in clone_to_cells_dict.items():
        clone = re.sub(r"_clone_\d+", "", clone.replace("super", ""))
        if clone not in edit_clone_to_cells_dict.keys():
            edit_clone_to_cells_dict[clone] = set()
        edit_clone_to_cells_dict[clone].update(cells)
    return edit_clone_to_cells_dict

def test_edit_clone_to_cells_dict():
    clone_to_cells_dict_exp = {
        'superclone_0_clone_0': ['cell_1', 'cell_2'],
        'superclone_0_clone_1': ['cell_3', 'cell_4'],
        'superclone_1_clone_0': ['cell_5', 'cell_6'],
    }
    expected_output = {
        'clone_0': {'cell_1', 'cell_2', 'cell_3', 'cell_4'},
        'clone_1': {'cell_5', 'cell_6'},
    }
    output_edit_clone_to_cells_dict = edit_clone_to_cells_dict(
        clone_to_cells_dict=clone_to_cells_dict_exp
    )
    assert output_edit_clone_to_cells_dict == expected_output, f"Expected {expected_output}, but got {output_edit_clone_to_cells_dict}"

def dict_to_set(
    clone_dict: Dict[CellBCType, List[CloneType]], 
    include_multiplets: bool = True
) -> Dict[CloneType, Set[CellBCType]]:
    """
    Transform the dictionary of cells and clone assignment to a set of frozensets of cells for each clone

    Args:
        clone_dict: dictionary of cells and clone assignment
        include_multiplets: boolean to include cells assigned to multiple clones

    Returns:
        clones_cells: dictionary of clone name and set of frozensets of cells for each clone
    """
    clones_cells = dict()
    for cell, clones in clone_dict.items():
        if len(clones) == 1 or include_multiplets:
            for clone in clones:
                if clone not in clones_cells.keys():
                    clones_cells[clone] = set()
                clones_cells[clone].add(cell)
    return clones_cells

def test_dict_to_set():
    clone_dict_exp = {
        'cell_1': ['clone_A', 'clone_B'],
        'cell_2': ['clone_A'],
        'cell_3': ['clone_B'],
        'cell_4': ['clone_C'],
        'cell_5': ['clone_C'],
    }
    expected_output = {
        'clone_A': {'cell_2'},
        'clone_B': {'cell_3'},
        'clone_C': {'cell_4','cell_5'},
    }
    output_dict_to_set = dict_to_set(clone_dict=clone_dict_exp, include_multiplets=False)
    assert output_dict_to_set == expected_output, f"Expected {expected_output}, but got {output_dict_to_set}"


def jaccard_similarity_calculation(
    clone_simulation: Set[CellBCType], clone_algorithm: Set[CellBCType]
) -> float:
    """
    Calculate the Jaccard similarity between two sets of cells

    Args:
        clone_simulation: set of cells from simulation clone
        clone_algorithm: set of cells from algorithm output clone

    Returns:
        jaccard_similarity
    """
    intersection = len(clone_simulation.intersection(clone_algorithm))
    union = len(clone_simulation.union(clone_algorithm))
    return intersection / union

def test_jaccard_similarity_calculation():
    clone_simulation = {'cell_1', 'cell_2', 'cell_3'}
    clone_algorithm = {'cell_2', 'cell_3', 'cell_4'}
    expected_jaccard_similarity = 0.5
    jaccard_similarity = jaccard_similarity_calculation(
        clone_simulation=clone_simulation, 
        clone_algorithm=clone_algorithm
    )
    assert jaccard_similarity == expected_jaccard_similarity, f"Expected {expected_jaccard_similarity}, but got {jaccard_similarity}"

def jaccard_similarity_between_clones(
    cells_clone_dict_input: Dict[CloneType, List[CellBCType]],
    cells_clone_dict_output: Dict[CloneType, List[CellBCType]],
) -> Dict[CloneType, Dict[CloneType, float]]:
    """
    Calculate the Jaccard similarity between the simulation ground truth clones and the algorithm output clones

    Args:
        cells_clone_dict_input: simulation ground truth clones as keys and cells as values
        cells_clone_dict_output: algorithm output clones as keys and cells as values

    Returns:
        jaccard_similarity: dict with clone output name as key and dict with clone ground truth name as key and
        jaccard similarity as value
    """
    jaccard_similarities = {clone: {} for clone in cells_clone_dict_output.keys()}
    for clone_output, cells_output in cells_clone_dict_output.items():
        for clone_input, cells_input in cells_clone_dict_input.items():
            jaccard_similarities[clone_output][clone_input] = jaccard_similarity_calculation(
                clone_simulation=set(cells_input), clone_algorithm=set(cells_output)
            )
        jaccard_similarities[clone_output]['no match'] = 0
    return jaccard_similarities

def test_jaccard_similarity_between_clones():
    clone_cells_input = {
        'clone_A': ['cell_1', 'cell_2', 'cell_3'],
        'clone_B': ['cell_4', 'cell_5'],
        'clone_C': ['cell_6', 'cell_7'],
    }
    clone_cells_output = {
        'clone_1': ['cell_1', 'cell_2','cell_3'],
        'clone_2': ['cell_4', 'cell_5','cell_6', 'cell_7'],
    }
    expected_jaccard_similarity = {
        'clone_1': {'clone_A': 1.0, 'clone_B': 0.0, 'clone_C': 0.0, 'no match': 0},
        'clone_2': {'clone_A': 0.0, 'clone_B': 0.5, 'clone_C': 0.5, 'no match': 0},
    }
    jaccard_similarity = jaccard_similarity_between_clones(
        cells_clone_dict_input=clone_cells_input,
        cells_clone_dict_output=clone_cells_output
    )
    assert jaccard_similarity == expected_jaccard_similarity, f"Expected {expected_jaccard_similarity}, but got {jaccard_similarity}"


def jaccard_similarity_between_barcodes_sets(
    clone_barcodes_input: Dict[CloneType, List[IntBCType]],
    clone_barcodes_output: Dict[CloneType, List[IntBCType]],
    matching_clones: Dict[CloneType, List[CloneType]],
) -> Dict[CloneType, Dict[CloneType, float]]:
    """
    Calculate the Jaccard similarity between the simulation ground truth barcodes and the matching algorithm output
    barcodes

    Args:
        clone_barcodes_input: simulation ground truth
        clone_barcodes_output: algorithm output
        matching_clones: dictionary of simulation clone names and their matching predeicted clones

    Returns:
        jaccard_similarity: dict with clone simulation name as key and dict with clone algorithm name as key and
        jaccard similarity as value
    """
    jaccard_similarities = {clone: {} for clone in clone_barcodes_output.keys()}
    for clone_output, clone_input in matching_clones.items():
        if clone_input != 'no match':
            barcodes_input = clone_barcodes_input[clone_input]
            barcodes_output = clone_barcodes_output[clone_output]
            jaccard_similarities[clone_output][clone_input] = jaccard_similarity_calculation(
                    clone_simulation=set(barcodes_input), clone_algorithm=set(barcodes_output)
                )
        else:
            jaccard_similarities[clone_output][clone_input] = 0
    return jaccard_similarities

def test_jaccard_similarity_between_barcodes_sets():
    clone_barcodes_input = {
        'clone_A': ['barcode_1', 'barcode_2', 'barcode_3'],
        'clone_B': ['barcode_4', 'barcode_5'],
        'clone_C': ['barcode_6', 'barcode_7'],
    }
    clone_barcodes_output = {
        'clone_1': ['barcode_1', 'barcode_2', 'barcode_3'],
        'clone_2': ['barcode_4', 'barcode_5', 'barcode_6', 'barcode_7'],
    }
    matching_clones = {
        'clone_1': 'clone_A',
        'clone_2': 'clone_B',
    }
    expected_jaccard_similarity = {
        'clone_1': {'clone_A': 1.0},
        'clone_2': {'clone_B': 0.5},
    }
    jaccard_similarity = jaccard_similarity_between_barcodes_sets(
        clone_barcodes_input=clone_barcodes_input,
        clone_barcodes_output=clone_barcodes_output,
        matching_clones=matching_clones
    )
    assert jaccard_similarity == expected_jaccard_similarity, f"Expected {expected_jaccard_similarity}, but got {jaccard_similarity}"


def ARI_between_clones(
    cells_clone_dict_input: Dict[CellBCType, List[CloneType]],
    cells_clone_dict_output: Dict[CellBCType, List[CloneType]],
    matching_clones: Dict[CloneType, CloneType],
) -> float:
    """
    Calculate the Adjusted Rand Index between the simulation ground truth clones and the algorithm output clones

    Args:
        cells_clone_dict_input: simulation ground truth
        cells_clone_dict_output: algorithm output
        matching_clones: dictionary of simulation clone names and their matching predicted clones

    Returns:
        ARI_score: ARI score
    """
    if len(matching_clones) > 0:
        cells_clone_dict_input = {
            cell: list(clones)[0]
            for cell, clones in cells_clone_dict_input.items()
            if len(clones) == 1
        }
        cells_clone_dict_output = {
            cell: list(clones)[0]
            for cell, clones in cells_clone_dict_output.items()
            if len(clones) == 1
        }
        cells_clone_dict_input = {
            cells: next(k for k, v in matching_clones.items() if v == clone)
            for cells, clone in cells_clone_dict_input.items()
            if clone in matching_clones.values()
        }
        shared_cells = set(cells_clone_dict_input.keys()).intersection(set(cells_clone_dict_output.keys()))
        all_cells = sorted(shared_cells)
        orderd_cells_gt = [cells_clone_dict_input[cell] for cell in all_cells]
        orderd_cells_pred = [cells_clone_dict_output[cell] for cell in all_cells]
        ARI_score = adjusted_rand_score(orderd_cells_gt, orderd_cells_pred)
    else:
        ARI_score = 0
    return ARI_score

def test_ARI_between_clones():
    cells_clone_dict_input = {
        'cell_1': ['clone_A'],
        'cell_2': ['clone_A','clone_B'],
        'cell_3': ['clone_B'],
        'cell_4': ['clone_B'],
    }
    cells_clone_dict_output = {
        'cell_1': ['clone_1'],
        'cell_2': ['clone_1'],
        'cell_3': ['clone_2'],
        'cell_4': ['clone_2'],
    }
    matching_clones = {
        'clone_1': 'clone_A',
        'clone_2': 'clone_B',
    }
    expected_ARI_score = 1.0
    ARI_score = ARI_between_clones(
        cells_clone_dict_input=cells_clone_dict_input,
        cells_clone_dict_output=cells_clone_dict_output,
        matching_clones=matching_clones
    )
    if ARI_score == expected_ARI_score:
        print("ARI score is correct")
    assert ARI_score == expected_ARI_score, f"Expected {expected_ARI_score}, but got {ARI_score}"


def ARI_between_barcode_sets(
    clone_barcodes_input: Dict[CloneType, List[IntBCType]],
    clone_barcodes_output: Dict[CloneType, List[IntBCType]],
    matching_clones: Dict[CloneType, CloneType],
) -> float:
    """
    Calculate the Adjusted Rand Index the barcodes per clone between the simulation ground truth and the algorithm output

    Args:
        clone_barcodes_input: simulation ground truth
        clone_barcodes_output: algorithm output
        matching_clones: dictionary of simulation clone names and their matching predeicted clones

    Returns:
        ARI_score: ARI score
    """
    if len(matching_clones) > 0:
        clone_barcodes_input = {
            barcode: clone
            for clone, barcodes in clone_barcodes_input.items()
            for barcode in barcodes
        }
        clone_barcodes_output = {
            barcode: clone
            for clone, barcodes in clone_barcodes_output.items()
            for barcode in barcodes if clone in matching_clones.keys()
        }
        clone_barcodes_input = {
            barcode: next(k for k, v in matching_clones.items() if v == clone)
            for barcode, clone in clone_barcodes_input.items()
            if clone in matching_clones.values()
        }   
        shared_barcodes = set(clone_barcodes_input.keys()).intersection(set(clone_barcodes_output.keys()))
        all_barcodes = sorted(shared_barcodes)
        orderd_barcodes_gt = [clone_barcodes_input[barcode] for barcode in all_barcodes]
        orderd_barcodes_pred = [clone_barcodes_output[barcode] for barcode in all_barcodes]
        ARI_score = adjusted_rand_score(orderd_barcodes_gt, orderd_barcodes_pred)
    else:
        ARI_score = 0
    return ARI_score

def test_ARI_between_barcode_sets():
    clone_barcodes_input = {
        'clone_A': ['barcode_1', 'barcode_2', 'barcode_3'],
        'clone_B': ['barcode_4', 'barcode_5'],
        'clone_C': ['barcode_6', 'barcode_7'],
    }
    clone_barcodes_output = {
        'clone_1': ['barcode_1', 'barcode_2', 'barcode_3'],
        'clone_2': ['barcode_4', 'barcode_5', 'barcode_6', 'barcode_7'],
    }
    matching_clones = {
        'clone_1': 'clone_A',
        'clone_2': 'clone_B',
    }
    expected_ARI_score = 1.0
    ARI_score = ARI_between_barcode_sets(
        clone_barcodes_input=clone_barcodes_input,
        clone_barcodes_output=clone_barcodes_output,
        matching_clones=matching_clones
    )
    assert ARI_score == expected_ARI_score, f"Expected {expected_ARI_score}, but got {ARI_score}"

def similarity_matrices_calculation_between_barcodes(
    input_clones: Dict[CloneType, List[IntBCType]],
    output_clones: Dict[CloneType, List[IntBCType]],
    matching_clones: Dict[CloneType, CloneType],
    clone_to_cells_dict: Dict[CloneType, List[CellBCType]],
) -> Dict[CloneType, Dict[CloneType, Dict[str, float | Set]]]:
    """
    Calculate precision, recall, accuracy and F1 score for each matching clone between the simulation ground truth

    Args:
        input_clones: simulation ground truth
        output_clones: algorithm output
        matching_clones: dictionary of matching clones between input (keys) and output clones (values)
        clone_to_cells_dict: dictionary of clone names and their cells for weighted calculation

    Returns:
        similarity_matrices: a dictionary for matching clones dictionary with precision, recall, 
        accuracy, F1 score, sets of false positive and false negative
        similarity_matrices_weighted: a dictionary for matching clones dictionary with precision, recall,
        F1 score calculated based on weighted barcodes
    """
    similarity_matrices = {clone: {} for clone in matching_clones.keys()}
    similarity_matrices_weighted = {clone: {} for clone in matching_clones.keys()}
    for clone_output, clone_input in matching_clones.items():
        if clone_input != 'no match':
            cells_list_input = input_clones[clone_input]
            cells_list_output = output_clones[clone_output]
            false_negative_clones = set(cells_list_input) - set(cells_list_output)
            false_positive_clones = set(cells_list_output) - set(cells_list_input)
            true_positive_clones = set(cells_list_input).intersection(set(cells_list_output))
            true_negative_clones = set(cells_list_input).union(set(cells_list_output)) - true_positive_clones - false_positive_clones - false_negative_clones
            similarity_matices_values = similarity_matrices_calculation(
                true_positive=true_positive_clones,
                false_positive=false_positive_clones,
                false_negative=false_negative_clones,
                true_negative=true_negative_clones,
            )
            similarity_matices_values_weighted = weighted_similarity_matrices_barcodes_calculation(
                true_positive=true_positive_clones,
                false_positive=false_positive_clones,
                false_negative=false_negative_clones,
                clone_to_barcodes_dict=input_clones,
                clone_to_cells_dict=clone_to_cells_dict,
            )
        else:
            similarity_matices_values = {
                "precision": 0,
                "recall": 0,
                "accuracy": 0,
                "F1": 0,
                "FP": set(),
                "FN": set(),
            }
            similarity_matices_values_weighted = { 
                "precision": 0,
                "recall": 0,
                "F1": 0,
            }
        similarity_matrices[clone_output][clone_input] = similarity_matices_values
        similarity_matrices_weighted[clone_output][clone_input] = similarity_matices_values_weighted
    return similarity_matrices, similarity_matrices_weighted

def test_similarity_matrices_calculation_between_barcodes():
    input_clones = {
        'clone_1': ['barcode_0', 'barcode_1', 'barcode_2'],
        'clone_2': ['barcode_3', 'barcode_4', 'barcode_5'],
        'clone_3': ['barcode_6', 'barcode_7'],
    }
    output_clones = {
        'clone_A': ['barcode_0', 'barcode_1', 'barcode_2'],
        'clone_B': ['barcode_3', 'barcode_4', 'barcode_5', 'barcode_6','barcode_7'],
    }
    matching_clones = {
        'clone_A': 'clone_1',
        'clone_B': 'clone_2',
    }
    clone_to_cells_dict = {
        'clone_1': ['cell_1', 'cell_2','cell_3'],
        'clone_2': ['cell_1', 'cell_3'],
        'clone_3': ['cell_4'],
    }
    expected_similarity_matrices = {
        'clone_A': {'clone_1': {'precision': 1.0, 'recall': 1.0, 'accuracy': 1.0, 'F1': 1.0, 'FP': set(), 'FN': set()}},
        'clone_B': {'clone_2': {'precision': 0.6, 'recall': 1.0, 'accuracy': 0.6, 'F1': 0.7499999999999999, 'FP': {'barcode_6','barcode_7'}, 'FN': set()}},
    }
    expected_similarity_matrices_weighted = {
        'clone_A': {'clone_1': {'precision': 1.0, 'recall': 1.0, 'F1': 1.0}},
        'clone_B': {'clone_2': {'precision': 0.75, 'recall': 1.0, 'F1': 0.8571428571428571}},
    }
    similarity_matrices, similarity_matrices_weighted = similarity_matrices_calculation_between_barcodes(
        input_clones=input_clones,
        output_clones=output_clones,
        matching_clones=matching_clones,
        clone_to_cells_dict=clone_to_cells_dict,
    )
    assert similarity_matrices == expected_similarity_matrices, f"Expected {expected_similarity_matrices}, but got {similarity_matrices}"
    assert similarity_matrices_weighted == expected_similarity_matrices_weighted, f"Expected {expected_similarity_matrices_weighted}, but got {similarity_matrices_weighted}"

def similarity_matrices_calculation_between_clones(
    input_clones: Dict[CellBCType, List[CloneType]],
    output_clones: Dict[CellBCType, List[CloneType]],
    matching_clones: Dict[CloneType, CloneType],
) -> Dict[CloneType, Dict[CloneType, Dict[str, float | Set]]]:
    """
    Calculate precision, recall, accuracy and F1 score for each matching clone between the simulation ground truth

    Args:
        input_clones: simulation ground truth
        output_clones: algorithm output
        matching_clones: dictionary of matching clones between input (keys) and output clones (values)

    Returns:
        similarity_matrices: a dictionary for matching clones dictionary with precision, recall, 
        accuracy, F1 score, sets of false positive and false negative
    """
    similarity_matrices = {clone: {} for clone in matching_clones.keys()}
    for clone_output, clone_input in matching_clones.items():
        if clone_input != 'no match':
            cells_list_input = input_clones[clone_input]
            cells_list_output = output_clones[clone_output]
            false_negative_clones = set(cells_list_input) - set(cells_list_output)
            false_positive_clones = set(cells_list_output) - set(cells_list_input)
            true_positive_clones = set(cells_list_input).intersection(set(cells_list_output))
            true_negative_clones = set(cells_list_input).union(set(cells_list_output)) - true_positive_clones - false_positive_clones - false_negative_clones
            similarity_matices_values = similarity_matrices_calculation(
                true_positive=true_positive_clones,
                false_positive=false_positive_clones,
                false_negative=false_negative_clones,
                true_negative=true_negative_clones,
            )
        else:
            similarity_matices_values = {
                "precision": 0,
                "recall": 0,
                "accuracy": 0,
                "F1": 0,
                "FP": set(),
                "FN": set(),
            }
        similarity_matrices[clone_output][clone_input] = similarity_matices_values            
    return similarity_matrices


def test_similarity_matrices_calculation_between_clones():
    input_clones = {
        'clone_1': ['cell_1', 'cell_2'],
        'clone_2': ['cell_3'],
        'clone_3': ['cell_4', 'cell_5'],
    }
    output_clones = {
        'clone_A': ['cell_1', 'cell_2'],
        'clone_B': ['cell_3', 'cell_4', 'cell_5'],
    }
    matching_clones = {
        'clone_A': 'clone_1',
        'clone_B': 'clone_3',
    }
    expected_similarity_matrices = {
        'clone_A': {'clone_1': {'precision': 1.0, 'recall': 1.0, 'accuracy': 1.0, 'F1': 1.0, 'FP': set(), 'FN': set()}},
        'clone_B': {'clone_3': {'precision': 2/3, 'recall': 1.0, 'accuracy': 2/3, 'F1': 0.8, 'FP': {'cell_3'}, 'FN': set()}},
    }
    similarity_matrices = similarity_matrices_calculation_between_clones(
        input_clones=input_clones,
        output_clones=output_clones,
        matching_clones=matching_clones,
    )
    assert similarity_matrices == expected_similarity_matrices, f"Expected {expected_similarity_matrices}, but got {similarity_matrices}"

def calculating_cells_per_barcodes(
        clone_to_barcodes_dict: Dict[CloneType, List[IntBCType]],
        clone_to_cells_dict: Dict[CloneType, List[CellBCType]],
    ) -> Dict[IntBCType, int]:
    """
    Calculate the number of cells per integration barcode

    Args:
        clone_to_barcodes_dict: dictionary of clone names and their barcodes
        clone_to_cells_dict: dictionary of clone names and their cells

    Returns:
        barcode_cell_counts: dictionary of integration barcodes and their cell counts
    """
    barcode_cell_counts = defaultdict(int)
    for clone_id in clone_to_barcodes_dict:
        barcodes = clone_to_barcodes_dict[clone_id]
        cells = clone_to_cells_dict.get(clone_id, [])
        n_cells = len(cells)
        for barcode in barcodes:
            barcode_cell_counts[barcode] += n_cells
    barcode_cell_counts = dict(barcode_cell_counts)
    return barcode_cell_counts

def test_calculating_cells_per_barcodes():
    clone_to_barcodes_dict = {
        'clone_A': ['barcode_1', 'barcode_2', 'barcode_3'],
        'clone_B': ['barcode_4', 'barcode_5'],
    }
    clone_to_cells_dict = {
        'clone_A': ['cell_1', 'cell_2'],
        'clone_B': ['cell_3', 'cell_4','cell_5'],
    }
    expected_output = {
        'barcode_1': 2,
        'barcode_2': 2,
        'barcode_3': 2,
        'barcode_4': 3,
        'barcode_5': 3,
    }
    output = calculating_cells_per_barcodes(
        clone_to_barcodes_dict=clone_to_barcodes_dict,
        clone_to_cells_dict=clone_to_cells_dict
    )
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

def similarity_matrices_calculation(
    true_positive: Set[str],
    false_positive: Set[str],
    false_negative: Set[str],
    true_negative: Set[str],
) -> Dict[str, float | Set]:
    """
    Calculate precision, recall, accuracy and F1 score

    Args:
        true_positive: set of true positive
        false_positive: set of false positive
        false_negative: set of false negative
        true_negative: set of true negative

    Returns:
        output: dictionary containing precision, recall, accuracy, F1 score, sets of false positive and false negative
    """
    precision = len(true_positive) / (len(true_positive) + len(false_positive)) if (len(true_positive) + len(false_positive)) > 0 else 0
    recall = len(true_positive) / (len(true_positive) + len(false_negative)) if (len(true_positive) + len(false_negative)) > 0 else 0
    accuracy = (len(true_positive) + len(true_negative)) / (
        len(true_positive)
        + len(true_negative)
        + len(false_positive)
        + len(false_negative)
    ) if (
        len(true_positive) + len(true_negative) + len(false_positive) + len(false_negative)
    ) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "F1": f1_score,
        "FP": false_positive,
        "FN": false_negative,
    }

def test_similarity_matrices_calculation():
    true_positive = {'cell_1', 'cell_2', 'cell_3'} 
    false_positive = {'cell_4'}
    false_negative = {'cell_5'}
    true_negative = {'cell_6', 'cell_7','cell_8'}
    expected_output = {
        "precision": 0.75,
        "recall": 0.75,
        "accuracy": 0.75,
        "F1": 0.75,
        "FP": {'cell_4'},
        "FN": {'cell_5'},
    }
    output = similarity_matrices_calculation(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative
    )
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

def weighted_similarity_matrices_barcodes_calculation(
    true_positive: Set[str],
    false_positive: Set[str],
    false_negative: Set[str],
    clone_to_barcodes_dict: Dict[CloneType, List[IntBCType]],
    clone_to_cells_dict: Dict[CloneType, List[CellBCType]],
) -> Dict[str, float | Set]:
    """
    Calculate precision, recall and F1 score based on weighted barcodes 

    Args:
        true_positive: set of true positive
        false_positive: set of false positive
        false_negative: set of false negative
        clone_to_barcodes_dict: dictionary of clone names and their barcodes
        clone_to_cells_dict: dictionary of clone names and their cells

    Returns:
        output: dictionary containing precision, recall and F1 score calculated based on weighted barcodes
    """
    frequency_barcodes = calculating_cells_per_barcodes(clone_to_barcodes_dict=clone_to_barcodes_dict, clone_to_cells_dict=clone_to_cells_dict)
    true_positive_weighted = sum(frequency_barcodes[bc] for bc in true_positive)
    false_positive_weighted = sum(frequency_barcodes.get(bc, 0) for bc in false_positive)
    false_negative_weighted = sum(frequency_barcodes[bc] for bc in false_negative)
    precision = true_positive_weighted / (true_positive_weighted + false_positive_weighted) if (true_positive_weighted + false_positive_weighted) > 0 else 0
    recall = true_positive_weighted / (true_positive_weighted + false_negative_weighted) if (true_positive_weighted + false_negative_weighted) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "precision": precision,
        "recall": recall,
        "F1": f1_score,
    }

def test_weighted_similarity_matrices_barcodes_calculation():
    true_positive = {'barcode_1', 'barcode_2', 'barcode_3'} 
    false_positive = {'barcode_4','barcode_5'}
    false_negative = {'barcode_6'}
    clone_to_barcodes_dict = {
        'clone_A': ['barcode_1', 'barcode_2', 'barcode_3','barcode_4','barcode_5'],
    }
    clone_to_cells_dict = {
        'clone_A': ['cell_1','cell_2'],
    }
    expected_output = {
        "precision": 0.6,
        "recall": 0.75,
        "F1": 0.666666666666667,
    }
    output = weighted_similarity_matrices_barcodes_calculation(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        clone_to_barcodes_dict=clone_to_barcodes_dict,
        clone_to_cells_dict=clone_to_cells_dict
    )
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

def compare_doublets(
    cells_clone_input: Dict[CellBCType, List[CloneType]],
    cells_clone_output: Dict[CellBCType, List[CloneType]],
) -> Dict[str, float | Set]:
    """
    Doublets identification

    Args:
        cells_clone_input: simulation ground truth
        cells_clone_output: algorithm output

    Returns:
        output: dictionary contining accuracy, precision, recall, false negative clones, false positive clones,
        jaccard similarity between all clones
    """
    cells_clone_input_doublets = [
        k for k, v in edit_input(cells_clone_input).items() if len(set(v)) > 1
    ]
    cells_clone_output_doublets = [
        k for k, v in edit_output(cells_clone_output).items() if len(v) > 1
    ]
    false_negative_doublets = set(cells_clone_input_doublets) - set(
        cells_clone_output_doublets
    )
    false_positive_doublets = set(cells_clone_output_doublets) - set(
        cells_clone_input_doublets
    )
    true_positive_doublets = set(cells_clone_input_doublets).intersection(
        set(cells_clone_output_doublets)
    )
    true_negative_doublets = (
        set(cells_clone_input_doublets).union(set(cells_clone_output_doublets))
        - true_positive_doublets
    )
    similarity_matices = similarity_matrices_calculation(
        true_positive=true_positive_doublets,
        false_positive=false_positive_doublets,
        false_negative=false_negative_doublets,
        true_negative=true_negative_doublets,
    )
    try:
        jaccard_similarity = len(true_positive_doublets) / (
            len(true_positive_doublets)
            + len(false_positive_doublets)
            + len(false_negative_doublets)
        )
    except ZeroDivisionError:
        print("No doublets found in the simulation or algorithm output")
        jaccard_similarity = 1
    output = {
        "similarity_matices": similarity_matices,
        "jaccard-similarity": jaccard_similarity,
        "false_negative_doublets": false_negative_doublets,
        "false_positive_doublets": false_positive_doublets,
    }
    return output

def test_compare_doublets():
    cells_clone_input = {
        'cell_1': ['clone_A', 'clone_B'],
        'cell_2': ['clone_A','clone_B'],
        'cell_3': ['clone_B'],
        'cell_4': ['clone_C', 'clone_D'],
    }
    cells_clone_output = {
        'cell_1': ['clone_1', 'clone_2'],
        'cell_2': ['clone_1'],
        'cell_3': ['clone_2'],
        'cell_4': ['clone_3', 'clone_4'],
    }
    expected_output = {
        "similarity_matices": {
            "precision": 1.0,
            "recall": 0.6666666666666666,
            "accuracy": 0.75,
            "F1": 0.8,
            "FP": set(),
            "FN": {'cell_2'},
        },
        "jaccard-similarity": 0.6666666666666666,
        "false_negative_doublets": {'cell_2'},
        "false_positive_doublets": set(),
    }
    output = compare_doublets(
        cells_clone_input=cells_clone_input,
        cells_clone_output=cells_clone_output,
    )
    assert output == expected_output, f"Expected {expected_output}, but got {output}"


def compare_clones(
    cells_clone_input: Dict[CellBCType, List[CloneType]],
    cells_clone_output: Dict[CellBCType, List[CloneType]],
    subtree_combinations: List[List[CloneType]],
    include_multiplets: bool = True,
    clones_type: str = 'clone',
) -> Dict[str, float | Set]:
    """
    Compare assigment of cells in clones between ground truth and algorithm output

    Args:
        cells_clone_input: simulation ground truth
        cells_clone_output: algorithm output
        subtree_combinations: list of subclone combinations
        include_multiplets: boolean to include cells assigned to multiple clones

    Returns:
        output: dictionary contining accuracy, precision, recall, false negative clones, false positive clones,
        jaccard similarity between all matching clones
    """
    if clones_type == 'clone':
        cells_clone_input = edit_input(cells_clone_input)
    cells_clone_output = edit_output(cells_clone_output)
    input_clones = dict_to_set(cells_clone_input, include_multiplets=include_multiplets)
    if clones_type == 'subclone':
        input_clones = add_combined_subtree_sets(input_clones, subtree_combinations)
    output_clones = dict_to_set(cells_clone_output, include_multiplets=include_multiplets)
    input_clones_frozenset = {frozenset(cells_sets) for clone, cells_sets in input_clones.items()}
    output_clones_frozenset = {frozenset(cells_sets) for clone, cells_sets in output_clones.items()}
    if len(output_clones) > 0:
        # jaccard similarity between all clones - to find the best matching clone for further analysis
        jaccard_similarity = jaccard_similarity_between_clones(
            cells_clone_dict_input=input_clones, 
            cells_clone_dict_output=output_clones
        )
        matching_clones = {
        clone_output: (max(clone_input, key=clone_input.get)
            if max(clone_input.values()) > 0.5 else 'no match')
        for clone_output, clone_input in jaccard_similarity.items()
        }
        similarity_matrices_between_clones = similarity_matrices_calculation_between_clones(
            input_clones=input_clones, 
            output_clones=output_clones, 
            matching_clones=matching_clones, 
        )
        input_cells_ARI = {cell:[clone] for clone, cells in input_clones.items() for cell in cells if clone in matching_clones.values()}
        ARI_score = ARI_between_clones(input_cells_ARI, cells_clone_output, matching_clones)
        output = {
            "similarity_matrices_between_clones": similarity_matrices_between_clones,
            "jaccard-similarity": jaccard_similarity,
            "matching_clones": matching_clones,
            "ARI_score": ARI_score,
        }
    else:
        print("No clones found in the algorithm output")
        output = {
            "similarity_matrices_between_clones": {'precision': 0, 'recall': 0, 'accuracy': 0, 'F1': 0, 'FP': set(), 'FN': input_clones_frozenset},
            "jaccard-similarity": {},
            "matching_clones": {},
            "ARI_score": 0,
        }
    return output


def test_compare_clones():
    cells_clone_input = {
        'cell_1': ['clone_A'],
        'cell_2': ['clone_A'],
        'cell_3': ['clone_A'],
        'cell_4': ['clone_B'],
        'cell_5': ['clone_B'],
        'cell_6': ['clone_B'],
        'cell_7': ['clone_B'],
        'cell_8': ['clone_C'],
        'cell_9': ['clone_C'],
        'cell_10': ['clone_C'],
    }
    cells_clone_output = {
        'cell_1': ['clone_1'],
        'cell_2': ['clone_1'],
        'cell_3': ['clone_1'],
        'cell_4': ['clone_1'],
        'cell_5': ['clone_2'],
        'cell_6': ['clone_2'],
        'cell_7': ['clone_2'],
        'cell_8': ['clone_2'],
        'cell_9': ['clone_3'],
        'cell_10': ['clone_3'],
    }
    subtree_combinations = [['clone_A', 'clone_B','clone_C'], ['clone_C'],['clone_B']]
    expected_output = {
        "similarity_matrices_between_clones": {
            'clone_1': {'clone_A': {'precision': 0.75, 'recall': 1.0, 'accuracy': 0.75, 'F1': 0.8571428571428571, 'FP': {'cell_4'}, 'FN': set()}},
            'clone_2': {'clone_B': {'precision': 0.75, 'recall': 0.75, 'accuracy': 0.6, 'F1': 0.75, 'FP': {'cell_8'}, 'FN': {'cell_4'}}},
            'clone_3': {'clone_C': {'precision': 1.0, 'recall': 2/3, 'accuracy': 2/3, 'F1': 0.8, 'FP': set(), 'FN': {'cell_8'}}},
        },
        "jaccard-similarity": {
            'clone_1': {'clone_A': 0.75, 'clone_B': 1/7, 'clone_C': 0.0, 'no match': 0},
            'clone_2': {'clone_A': 0.0, 'clone_B': 0.6, 'clone_C': 1/6, 'no match': 0},
            'clone_3': {'clone_A': 0.0, 'clone_B': 0.0, 'clone_C': 0.6666666666666666, 'no match': 0},
        },
        "matching_clones": {
            'clone_1': 'clone_A',
            'clone_2': 'clone_B',
            'clone_3': 'clone_C',
        },
        "ARI_score": adjusted_rand_score(['clone_1','clone_1','clone_1','clone_2','clone_2','clone_2','clone_2','clone_3','clone_3','clone_3'], ['clone_1','clone_1','clone_1','clone_1','clone_2','clone_2','clone_2','clone_2','clone_3','clone_3']),
    }
    output = compare_clones(
        cells_clone_input=cells_clone_input,
        cells_clone_output=cells_clone_output,
        subtree_combinations=subtree_combinations,
    )
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

def compare_superclone_barcodes(
    clone_barcodes_input: Dict[CloneType, List[IntBCType]],
    clone_barcodes_output: Dict[CloneType, List[IntBCType]],
    matching_clones: Dict[CloneType, List[CloneType]],
    subtree_combinations: List[List[CloneType]],
    clone_to_cells_dict: Dict[CloneType, List[CellBCType]],
    clones_type: str = 'clone',
) -> Dict[str, float | Set | Dict]:
    """
    Compare barcode recognition between ground truth and algorithm output per superclone

    Args:
        superclone_barcodes_input: simulation ground truth
        superclone_barcodes_output: algorithm output
        matching_clones: dictionary of simulation clone names and their matching predeicted clones
        subtree_combinations: list of subclone combinations
        clone_to_cells_dict: dictionary of clone names and their cells for weighted calculation
        clones_type: type of clones - clone or subclone

    Returns:
        output: dictionary contining accuracy, precision, recall, false negative barcodes, false positive barcodes,
        jaccard similarity between all matching clones
    """
    if clones_type == 'clone':
        clone_barcodes_input = {re.sub(r"_clone_\d+", "",k):v for k,v in clone_barcodes_input.items()}
        clone_barcodes_input = {re.sub("super", "",k):v for k,v in clone_barcodes_input.items()}
        clone_to_cells_dict = edit_clone_to_cells_dict(clone_to_cells_dict)
    if clones_type == 'subclone':
        clone_barcodes_input = add_combined_subtree_sets(clone_barcodes_input, subtree_combinations)
    if len(clone_barcodes_output) > 0:
        similarity_matrices_between_clones, similarity_matrices_between_clones_weighted = similarity_matrices_calculation_between_barcodes(
                                            input_clones=clone_barcodes_input,
                                            output_clones=clone_barcodes_output,
                                            matching_clones=matching_clones,
                                            clone_to_cells_dict=clone_to_cells_dict,)
        
        jaccard_similarity = jaccard_similarity_between_barcodes_sets(
                clone_barcodes_input=clone_barcodes_input, 
                clone_barcodes_output=clone_barcodes_output, 
                matching_clones=matching_clones
            )
        ARI_score = ARI_between_barcode_sets(
                clone_barcodes_input=clone_barcodes_input, 
                clone_barcodes_output=clone_barcodes_output, 
                matching_clones=matching_clones
            )
        output = {
                "similarity_matrices_between_clones": similarity_matrices_between_clones,
                "similarity_matrices_between_clones_weighted": similarity_matrices_between_clones_weighted,
                "jaccard-similarity": jaccard_similarity,
                "matching_clones": matching_clones,
                "ARI_score": ARI_score,
            }
    else:
        output = {
            "similarity_matrices_between_clones": {'precision': 0, 'recall': 0, 'accuracy': 0, 'F1': 0, 'FP': set(), 'FN': set()},
            "similarity_matrices_between_clones_weighted": {'precision': 0, 'recall': 0, 'F1': 0},
            "jaccard-similarity": {},
            "matching_clones": {},
            "ARI_score": 0,
        }
    return output

def test_compare_superclone_barcodes():
    clone_barcodes_input = {
        'clone_A': ['barcode_1', 'barcode_2', 'barcode_3','barcode_x'],
        'clone_B': ['barcode_4', 'barcode_5', 'barcode_6'],
        'clone_C': ['barcode_7', 'barcode_8'],
    }
    clone_barcodes_output = {
        'clone_1': ['barcode_1', 'barcode_2', 'barcode_3'],
        'clone_2': ['barcode_4', 'barcode_5', 'barcode_6', 'barcode_7','barcode_8'],
    }
    matching_clones = {
        'clone_1': 'clone_A',
        'clone_2': 'clone_B',
    }
    subtree_combinations = [['clone_A', 'clone_B','clone_C'], ['clone_C'],['clone_B']]
    clone_to_cells_dict = {
        'clone_A': ['cell_1', 'cell_2', 'cell_3','cell_4'],
        'clone_B': ['cell_3', 'cell_4','cell_5'],
        'clone_C': ['cell_6', 'cell_7'],
    }
    expected_output = {
        "similarity_matrices_between_clones": {
            "clone_1": {"clone_A": {"precision": 1.0, "recall": 0.75, "accuracy": 0.75, "F1": 0.8571428571428571, "FP": set(), "FN": {'barcode_x'}}},
            "clone_2": {"clone_B": {"precision": 0.6, "recall": 1.0, "accuracy": 0.6, "F1": 0.7499999999999999, "FP": {'barcode_7','barcode_8'}, "FN": set()}},
        },
        "similarity_matrices_between_clones_weighted": { #TODO
            "clone_1": {"clone_A": {"precision": 1.0, "recall": 0.75, "F1": 0.8571428571428571}},
            "clone_2": {"clone_B": {"precision": 9/13, "recall": 1.0, "F1": 0.8181818181818181}},
        },
        "jaccard-similarity": {
            "clone_1": {"clone_A": 0.75},
            "clone_2": {"clone_B": 0.6},
        },
        "matching_clones": {
            "clone_1": "clone_A",
            "clone_2": "clone_B",
        },
        "ARI_score": adjusted_rand_score(
            ['clone_1', 'clone_1', 'clone_1','clone_2', 'clone_2', 'clone_2'],
            ['clone_1', 'clone_1', 'clone_1', 'clone_2', 'clone_2', 'clone_2']
        ),
    }
    output = compare_superclone_barcodes(
        clone_barcodes_input=clone_barcodes_input,  
        clone_barcodes_output=clone_barcodes_output,
        matching_clones=matching_clones,
        subtree_combinations=subtree_combinations,
        clone_to_cells_dict=clone_to_cells_dict,
    )   
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

def subclonal_cells_recovery_rate(
        output_cells_to_clones: Dict[CloneType, List[CellBCType]],
        output_recursive_clones: Dict,
    ) -> float:
    """
    Calculate the number of cells that were assigned to subclones in the algorithm output

    Args:
        output_clones_to_cells: dictionary of output clones and their cells
        output_recursive_clones: dictionary of recursive clones and their subclones structure of cells

    Returns:
        subclonal_cells_recovery_rate: float value of the number of cells assigned to subclones divided by the total number of cells
    """
    if not output_recursive_clones:
        return 0.0
    single_clone_cells = {k: v for k, v in output_cells_to_clones.items() if len(v) == 1}
    total_cells = len(single_clone_cells)
    subclones_cells_count = 0
    for clone in output_recursive_clones.keys():
        if output_recursive_clones[clone] is not None:
            for subclone in output_recursive_clones[clone].clone_call.clone_to_cells_mapping.keys():
                subclones_cells_count += len(output_recursive_clones[clone].clone_call.clone_to_cells_mapping[subclone])
    return subclones_cells_count / total_cells if total_cells > 0 else 0.0

def results_summary(
    results: Dict[str, float | Dict]
) -> Dict[str, List]:
    """
    Get the results from the benchmarking

    Args:
        results: dictionary containing the results

    Returns:
        results: dictionary containing the results ordered by the metric
    """
    organized_results = {"F1": [], "precision": [], "recall": [], "accuracy": [], "jaccard-similarity": [], "ARI_score": results["ARI_score"]}
    if not all(matching_input == 'no match' for matching_input in results['matching_clones'].values()):
        for output_clones, gt_clone in results['matching_clones'].items():
            organized_results["jaccard-similarity"].append(results["jaccard-similarity"][output_clones][gt_clone])
            for metric in ["F1", "precision", "recall", "accuracy"]:
                organized_results[metric].append(results["similarity_matrices_between_clones"][output_clones][gt_clone][metric])
    else:
        organized_results = {'F1': [0], "precision": [0], "recall": [0], "accuracy": [0], "jaccard-similarity": [0], "ARI_score": results["ARI_score"]}
    return organized_results

def test_results_summary():
    results = {
        'matching_clones': {'clone_1': 'clone_A', 'clone_2': 'clone_B'},
        'similarity_matrices_between_clones': {
            'clone_1': {'clone_A': {'precision': 0.75, 'recall': 1.0, 'accuracy': 0.75, 'F1': 0.75, 'FP': set(), 'FN': set()}},
            'clone_2': {'clone_B': {'precision': 1.0, 'recall': 0.6, 'accuracy': 0.6, 'F1': 0.6, 'FP': set(), 'FN': set()}},
        },
        'jaccard-similarity': {
            'clone_1': {'clone_A': 0.75},
            'clone_2': {'clone_B': 0.6},
        },
        "ARI_score": 0.8,
    }
    expected_output = {
        "F1": [0.75, 0.6],
        "precision": [0.75, 1.0],
        "recall": [1.0, 0.6],
        "accuracy": [0.75, 0.6],
        "jaccard-similarity": [0.75, 0.6],
        "ARI_score": 0.8,
    }
    output = results_summary(results)
    assert output == expected_output, f"Expected {expected_output}, but got {output}"


def results_summary_weighted(
    results: Dict[str, float | Dict]
) -> Dict[str, List]:   
    """
    Get the results from the benchmarking

    Args:
        results: dictionary containing the results
        
    Returns:
        results: dictionary containing the results ordered by the metric
    """
    organized_results = {"F1": [], "precision": [], "recall": []}
    if not all(matching_input == 'no match' for matching_input in results['matching_clones'].values()):
        for output_clones, gt_clone in results['matching_clones'].items():
            for metric in ["F1", "precision", "recall"]:
                organized_results[metric].append(results["similarity_matrices_between_clones_weighted"][output_clones][gt_clone][metric])
    else:
        organized_results = {'F1': [0], "precision": [0], "recall": [0]}
    return organized_results



def evaluation_results(
    cell_clone_dict_input,
    cells_clone_dict_output,
    clone_barcode_dict_input,
    clone_barcode_dict_output,
    subtree_combinations,
    clone_to_cells_dict,
    recursive_clone_calling_results
):
    test_doublets = compare_doublets(
        cell_clone_dict_input, cells_clone_dict_output
    )
    results_evaluation = {}
    for clones_type in ['clone', 'subclone']:
        test_cell_in_clone = compare_clones(
            cells_clone_input=cell_clone_dict_input, 
            cells_clone_output=cells_clone_dict_output, 
            include_multiplets=False, 
            clones_type=clones_type, 
            subtree_combinations=subtree_combinations
        )
        test_barcodes_in_clone = compare_superclone_barcodes(
            clone_barcodes_input=clone_barcode_dict_input, 
            clone_barcodes_output=clone_barcode_dict_output, 
            matching_clones=test_cell_in_clone['matching_clones'], 
            clones_type=clones_type, 
            subtree_combinations=subtree_combinations,
            clone_to_cells_dict=clone_to_cells_dict,
        )
        subclonal_recovery = subclonal_cells_recovery_rate(
            output_cells_to_clones=cells_clone_dict_output,
            output_recursive_clones=recursive_clone_calling_results,
        )
        results_cells_clones = results_summary(test_cell_in_clone)
        results_barcodes_clones = results_summary(test_barcodes_in_clone)
        results_barcodes_clones_weighted = results_summary_weighted(test_barcodes_in_clone)
        results_evaluation[clones_type] = {
                                        "doublets": test_doublets,
                                        "cell_in_clones": test_cell_in_clone,
                                        "barcodes_in_clones": test_barcodes_in_clone,
                                        "results_cells_summary": results_cells_clones,
                                        "results_barcodes_summary": results_barcodes_clones,
                                        "results_barcodes_summary_weighted": results_barcodes_clones_weighted,
                                        "subclonal_cells_recovery_rate": subclonal_recovery,
                                        }
    return results_evaluation