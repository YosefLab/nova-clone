"""
Operations for e.g. merging clones, dropping clones, in a clone call.
"""
import typing as t
from ._types import CloneCallType
from copy import deepcopy
from collections import defaultdict


def merge_clones(
    clones: t.List[str],
    clone_call: CloneCallType,
    new_clone_name: str,
) -> CloneCallType:
    """
    Merge the given clones into one.

    clone_to_cells_mapping: Dict[CloneType, Set[CellBCType]]
        => we replace the clones by a new key with union of cells.
    cell_to_clones_mapping: Dict[CellBCType, Set[CloneType]]
        => we collapse the merged clones into one. To be conservative, multiplets contained in exclusively the `clones` become unmapped cells.
    clone_intbc_sets: Dict[CloneType, Set[IntBCType]]
        => we take the union.
    cell_intbc_sets: Dict[CellBCType, Set[IntBCType]]
        => Stays as-is.
    """
    clone_to_cells_mapping = deepcopy(clone_call.clone_to_cells_mapping)
    cell_to_clones_mapping = deepcopy(clone_call.cell_to_clones_mapping)
    clone_intbc_sets = deepcopy(clone_call.clone_intbc_sets)
    cell_intbc_sets = deepcopy(clone_call.cell_intbc_sets)
    new_clone_to_cells_mapping = defaultdict(set)
    new_cell_to_clones_mapping = {}
    new_clone_intbc_sets = defaultdict(set)
    new_cell_intbc_sets = {}
    for clone in clone_to_cells_mapping:
        if clone not in clones:
            new_clone_to_cells_mapping[clone] = clone_to_cells_mapping[clone]
        else:
            new_clone_to_cells_mapping[new_clone_name] = new_clone_to_cells_mapping[new_clone_name].union(
                clone_to_cells_mapping[clone]
            )
    for cell in cell_to_clones_mapping:
        cell_clones = cell_to_clones_mapping[cell]
        new_cell_clones = set()
        if all([clone in cell_clones for clone in clones]) and len(cell_clones) >= 2:
            # This is a multiplet that mapped only to the merged clones.
            # We unmap it to be safe, i.e. we leave `new_cell_clones` empty.
            pass
        else:
            for clone in cell_clones:
                if clone in clones:
                    new_cell_clones.add(new_clone_name)
                else:
                    new_cell_clones.add(clone)
        new_cell_to_clones_mapping[cell] = new_cell_clones
    for clone in clone_intbc_sets:
        intbcs = clone_intbc_sets[clone]
        if clone not in clones:
            new_clone_intbc_sets[clone] = intbcs
        else:
            new_clone_intbc_sets[new_clone_name] = new_clone_intbc_sets[new_clone_name].union(intbcs)
    new_cell_intbc_sets = cell_intbc_sets
    return CloneCallType(
        clone_to_cells_mapping=new_clone_to_cells_mapping,
        cell_to_clones_mapping=new_cell_to_clones_mapping,
        clone_intbc_sets=new_clone_intbc_sets,
        cell_intbc_sets=new_cell_intbc_sets,
    )



def _get_test_clone_call():
    clone_call = CloneCallType(
        clone_to_cells_mapping={
            "clone_1": {"cell_1", "cell_2", "cell_3"},
            "clone_2": {"cell_4", "cell_5", "cell_6"},
            "clone_3": {"cell_7", "cell_8"},
            "clone_4": {"cell_10"},
        },
        cell_to_clones_mapping={
            "cell_1": {"clone_1"},
            "cell_2": {"clone_1"},
            "cell_3": {"clone_1"},
            "cell_4": {"clone_2"},
            "cell_5": {"clone_2"},
            "cell_6": {"clone_2"},
            "cell_7": {"clone_3"},
            "cell_8": {"clone_3"},
            "cell_9": {"clone_2", "clone_3"},
            "cell_10": {"clone_4"},
            "cell_11": set(),
            "cell_12": {"clone_1", "clone_4"}
        },
        clone_intbc_sets={
            "clone_1": {"A", "B"},
            "clone_2": {"C", "D"},
            "clone_3": {"E", "F"},
            "clone_4": {"F", "G"},
        },
        cell_intbc_sets={
            "cell_1": {"A", "B"},
            "cell_2": {"A", "B"},
            "cell_3": {"A", "B"},
            "cell_4": {"C", "D"},
            "cell_5": {"C", "D"},
            "cell_6": {"C", "D"},
            "cell_7": {"E", "F"},
            "cell_8": {"E", "F"},
            "cell_9": {"C", "D", "E", "F"},
            "cell_10": {"F", "G"},
            "cell_11": set(),
            "cell_12": {"A", "B", "F", "G"},
        }
    )
    return clone_call


def test_merge_clones():
    clone_call = _get_test_clone_call()
    clone_call = merge_clones(
        clones=["clone_2", "clone_3", "clone_4"],
        clone_call=clone_call,
        new_clone_name="clone_2_3_4"
    )
    assert clone_call.clone_to_cells_mapping == {
        "clone_1": {"cell_1", "cell_2", "cell_3"},
        "clone_2_3_4": {"cell_4", "cell_5", "cell_6", "cell_7", "cell_8", "cell_10"},
    }
    assert clone_call.cell_to_clones_mapping == {
        "cell_1": {"clone_1"},
        "cell_2": {"clone_1"},
        "cell_3": {"clone_1"},
        "cell_4": {"clone_2_3_4"},
        "cell_5": {"clone_2_3_4"},
        "cell_6": {"clone_2_3_4"},
        "cell_7": {"clone_2_3_4"},
        "cell_8": {"clone_2_3_4"},
        "cell_9": {"clone_2_3_4"},
        "cell_10": {"clone_2_3_4"},
        "cell_11": set(),
        "cell_12": {"clone_1", "clone_2_3_4"},
    }
    assert clone_call.clone_intbc_sets == {
        "clone_1": {"A", "B"},
        "clone_2_3_4": {"C", "D", "E", "F", "G"},
    }
    assert clone_call.cell_intbc_sets == {
        "cell_1": {"A", "B"},
        "cell_2": {"A", "B"},
        "cell_3": {"A", "B"},
        "cell_4": {"C", "D"},
        "cell_5": {"C", "D"},
        "cell_6": {"C", "D"},
        "cell_7": {"E", "F"},
        "cell_8": {"E", "F"},
        "cell_9": {"C", "D", "E", "F"},
        "cell_10": {"F", "G"},
        "cell_11": set(),
        "cell_12": {"A", "B", "F", "G"},
    }


def drop_clones(
    clones: t.List[str],
    clone_call: CloneCallType
) -> CloneCallType:
    """
    Drop the given clones.

    clone_to_cells_mapping: Dict[CloneType, Set[CellBCType]]
        => we drop the clones from here.
    cell_to_clones_mapping: Dict[CellBCType, Set[CloneType]]
        => We unmap every cell which maps to *any* of the dropped clones.
    clone_intbc_sets: Dict[CloneType, Set[IntBCType]]
        => we drop the clones from here
    cell_intbc_sets: Dict[CellBCType, Set[IntBCType]]
        => we clear the intBC set of every cell that we unmapped
    """
    clone_to_cells_mapping = deepcopy(clone_call.clone_to_cells_mapping)
    cell_to_clones_mapping = deepcopy(clone_call.cell_to_clones_mapping)
    clone_intbc_sets = deepcopy(clone_call.clone_intbc_sets)
    cell_intbc_sets = deepcopy(clone_call.cell_intbc_sets)
    new_clone_to_cells_mapping = defaultdict(set)
    new_cell_to_clones_mapping = {}
    new_clone_intbc_sets = defaultdict(set)
    new_cell_intbc_sets = {}
    for clone in clone_to_cells_mapping:
        if clone not in clones:
            new_clone_to_cells_mapping[clone] = clone_to_cells_mapping[clone]
    for cell in cell_to_clones_mapping:
        cell_clones = cell_to_clones_mapping[cell]
        new_cell_clones = set()
        if any([clone in clones for clone in cell_clones]):
            new_cell_to_clones_mapping[cell] = set()
        else:
            new_cell_to_clones_mapping[cell] = cell_clones
    for clone in clone_intbc_sets:
        intbcs = clone_intbc_sets[clone]
        if clone not in clones:
            new_clone_intbc_sets[clone] = intbcs
    for cell in cell_intbc_sets:
        if len(new_cell_to_clones_mapping[cell]) == 0:
            new_cell_intbc_sets[cell] = set()
        else:
            new_cell_intbc_sets[cell] = cell_intbc_sets[cell]
    return CloneCallType(
        clone_to_cells_mapping=new_clone_to_cells_mapping,
        cell_to_clones_mapping=new_cell_to_clones_mapping,
        clone_intbc_sets=new_clone_intbc_sets,
        cell_intbc_sets=new_cell_intbc_sets,
    )


def test_drop_clones():
    clone_call = _get_test_clone_call()
    clone_call = drop_clones(
        clones=["clone_2", "clone_3", "clone_4"],
        clone_call=clone_call,
    )
    assert clone_call.clone_to_cells_mapping == {
        "clone_1": {"cell_1", "cell_2", "cell_3"},
    }
    assert clone_call.cell_to_clones_mapping == {
        "cell_1": {"clone_1"},
        "cell_2": {"clone_1"},
        "cell_3": {"clone_1"},
        "cell_4": set(),
        "cell_5": set(),
        "cell_6": set(),
        "cell_7": set(),
        "cell_8": set(),
        "cell_9": set(),
        "cell_10": set(),
        "cell_11": set(),
        "cell_12": set(),
    }
    assert clone_call.clone_intbc_sets == {
        "clone_1": {"A", "B"},
    }
    assert clone_call.cell_intbc_sets == {
        "cell_1": {"A", "B"},
        "cell_2": {"A", "B"},
        "cell_3": {"A", "B"},
        "cell_4": set(),
        "cell_5": set(),
        "cell_6": set(),
        "cell_7": set(),
        "cell_8": set(),
        "cell_9": set(),
        "cell_10": set(),
        "cell_11": set(),
        "cell_12": set(),
    }
