# NovaClone

NovaClone is a network-based *clone calling* algorithm for datasets generated through static or evolving transgene-integration events such as lineage tracing experiments.

## Overview

NovaClone assigns cells to clones in a principled, noise-aware way and provides QC metrics for evaluating dataset quality and clone calling results.
NovaClone works with barcode tables and supports multi-sample datasets.

## Features

- Network-based clone calling algorithm robust to sequencing errors, missing data, multiplets, background noise and homoplasy.
- Applicable to static and evolving transgene-integration barcoding.
- Built-in QC metrics (pre- and post-clone-calling)
- Fast Python implementation.
- Simple API.
- Default parameters should work well for most datasets.

## Installation

```
git clone git@github.com:YosefLab/nova-clone.git
cd nova-clone

# Option 1 (conda)
conda env create -f environment.yml
conda activate nova_clone-env

# Option 2 (pip)
pip install -r requirements.txt
```

You may verify your installation by running the test suite:

```
python -m pytest tests
```

## System requirements

This package requires Python version >= 3.10.

## Input format

NovaClone expects a dictionary mapping sample names to their **intBC (integration barcode) table**, where each intBC table is a pandas DataFrame with columns:

- **cellBC** — cell barcode  
- **intBC** — integration barcode (transgene insertion site)  
- **UMI_count** — UMI count or evidence score for that integration event

Example:

```
intBC_table_per_sample = {"sample0": intBC_table, "sample1": another_intBC_table}
```
Example `intBC_table`:
| Row number | cellBC | intBC | UMI_count |
|---|---|---|---|
| 1 | AAACCCATCAAACGAA | AAACCCATCAAACGAA | 27 |
| 2 | AAACCCATCAAACGAA | ACCAGTCGTATTGA | 2 |
| 3 | TTGCTACGCGTATCGC | ACCAGTCGTATTGT | 14 |

## Quick start

The typical workflow is:
(0) Read your data and build the intBC_table_per_sample dictionary that NovaClone expects.
(1) Run NovaClone's pre-QC on your intBC_table_per_sample, which will provide quality metrics and highlight potential issues with your data *before* the NovaClone is run.
(2) Run NovaClone on your intBC_table_per_sample.
(3) Run QC on the clone call found by NovaClone. NOTE: you can run this QC on clone calls from any other algorithm you are using, the QC is not specific to the NovaClone algorithm.

```
import pandas as pd
from nova_clone import clone_calling_algorithm
from nova_clone.qc import pre_qc_complete_analysis, qc_complete_analysis

# load your intBC table(s)
intBC_table_sample0 = pd.read_csv('intBC_table_sample0.csv')
intBC_table_sample1 = pd.read_csv('intBC_table_sample1.csv')
intBC_table_per_sample = {"sample0": intBC_table_sample0, "sample1": intBC_table_sample1}  # The sample names can be anything you want e.g. "sample0", "right_lung", etc.

# run pre-QC
pre_qc_complete_analysis(intBC_table_per_sample)

# run clone calling
clone_calling_results = clone_calling_algorithm(intBC_table_per_sample)

# run post-QC
qc_complete_analysis(intBC_table_per_sample, clone_call=clone_calling_results.clone_call)
```

See [Example.ipynb](notebooks/Example.ipynb) for a self-contained example on the macsGESTALT data, and [clone_calling_macsGESTALT_fig2Res.ipynb](notebooks/macsGESTALT_data/clone_calling_macsGESTALT_fig2Res.ipynb) for a longer version of the example which reproduces the figures in our paper.

## Output

NovaClone's `clone_calling_algorithm` returns a `results` tuple which contains the `clone_call`. The main entries of interest are thus:

- `results.clone_call.cell_to_clones_mapping (Dict[str, Set[str]])`: Dictionary mapping cellBCs to clone IDs. A cellBCs mapping to more than one clone is a multiplet, and a cellBC mapping to no clone is an empty cellBC or a dropped cellBC (e.g. due to low quality).
- `results.clone_call.clone_to_cells_mapping (Dict[str, Set[str]])`: Dictionary clone IDs to cellBCs.
- `results.clone_call.clone_intbc_sets (Dict[str, Set[str]])`: Dictionary mapping clone IDs to its set of IntBCs.
- `results.clone_call.cell_intbc_sets (Dict[str, Set[str]])`: Dictionary mapping cellBCs to its set of IntBCs.

## Full API

While we expect the default settings to work well on most datasets (as we show in our paper), here is the full list of parameters.

```
def clone_calling_algorithm(
    ##### Public parameters (defaults should work well on most datasets) #####
    # Phase 1 parameters
    intbc_table_per_sample: Dict[str, pd.DataFrame],
    min_intbc_umi_count: int = 10,
    custom_intbc_filtering_function: Callable[[str], bool] = lambda intbc: False,
    min_intbc_cellbc_count: int = 10,
    edit_distance_cutoff: int = 4,
    # Phase 2 parameters
    min_cells_assigned: int = 10,
    min_intbc_fraction: float = 0.05,
    edit_distance_cutoff_phase_2: Optional[int] = None, # If `None`, defaults to `edit_distance_cutoff` value
    # Recursive step parameters
    recursive: bool = False,
    recursive_min_fraction_uniquely_mapped_cells_to_recurse: float = 0.7,
    recursive_min_cells_assigned: int = 10,
    # Single-intBC clone recovery
    recover_single_intbc_clones: bool = False,
) -> CloneCallingAlgorithmReturnType:
    """
    Args:
        intbc_table_per_sample: Each value should be a pandas DataFrame with the following columns:
            "cellBC" (str), "intBC" (str), "UMI_count" (int). For example, a row might be:
            ["AAACCCAAGAGCCTGA", "ACCTCATGATATTC", 45]
            indicating that the intBC "ACCTCATGATATTC" was observed 45 times in the cellBC "AAACCCAAGAGCCTGA".
            There should be one intBC table per sample; the sample name acts as the key of the dictionary.
        min_intbc_umi_count: IntBCs with less than this number of UMI_counts will be discarded by the algorithm.
            The default value is 10, which is effective at removing sequencing errors and background RNA and
            sequencing errors. For some datasets, such as the macsGESTALT which we analyze, we lower this
            threshold to two UMIs, as described in the main text. More generally, we recommend to set the UMI
            cutoff by considering a histogram of UMI counts across all (cellBC, intBC) pairs (rows in the input
            table) and use the leftmost mode (in case the count distribution is multi-modal) or a low quantile
            (otherwise).
        custom_intbc_filtering_function: IntBCs for which this function gives
            `True` will be discarded.
        min_intbc_cellbc_count: IntBCs appearing in less than this number of
            cellBCs will be discarded.
        edit_distance_cutoff: Minimum *allowed* edit distance. E.g. if you expect there to
            exist sequencing errors at edit distance 3 but not 4, please set this to 4.
            To be precise about how this is used: When building clonal signatures (phase 1 of
            the algorithm), we identify putative sequencing errors by checking for
            intBCs `x` frequently co-occurring with another (more frequent) intBC
            `y` at an edit distance <= edit_distance_cutoff. If so, we remove `x`.
            In the second phase of the algorithm, when we reconstruct
            the intBC set for each clone de novo, we require that the edit distance
            to any other more abundant intBC is >= edit_distance_cutoff.
            Note that our cutoff for phase 1 is extra stringent to be extra safe, since
            sequencing error in phase 1 have a bigger consequence (overmerging clones) than
            in phase 2 (just getting some aliased characters which will likely get dropped
            due to low frequency via `min_intbc_fraction` anyway). We allow setting the
            cutoff for phase 2 differently with `edit_distance_cutoff_phase_2`.
        min_cells_assigned: Coresets with less than this number of mapped cells
            will be dropped. If you wish to recover smaller clones, you can lower
            this, however, note that since all \intBCs are supported by at least
            min_intbc_cellbc_count cellBCs (by default 10, and we do not recommend
            changing this), all clones will have at least 10 mapped cellBCs.
        min_intbc_fraction:
            Following up on the above argument, we can also specify the
            minimum percentage of cells in the clone containing the intBC
            needed to include the intBC in the clone's intBC set. We set this
            to 5% by default to recover the intBC tail too. Set this to e.g.
            20% if you only wish to retain only high-frequency intBCs.
       edit_distance_cutoff_phase_2: See description of parameter edit_distance_cutoff.
            If `None` (default) this is set equal to edit_distance_cutoff. This should work
            well in most cases. Reasons for setting this value to something different than
            `edit_distance_cutoff` are (i) increase retention of intBCs after clone calling;
            note that while removing sequencing errors during the first phase (finding clonal
            signatures) is critical (or else we might overmerge clones), when building intBC
            sets de novo is it less critical because in the worst case, sequencing errors
            lead to "duplicate" characters which will likely get dropped due to low frequency
            via `min_intbc_fraction` anyway, and (ii) the user might want to curate the intBCs
            in each clone themselves, in which case they can set edit_distance_cutoff_phase_2=0
            to turn off this second sequencing error filter.
        recursive: If True, then we will recursively call the algorithm on each
            clone, thus resolving the subclonal structure derived by the
            hierarchy of integrations. When using this option, we recommend
            (and will enforce) using:
            min_intbc_fraction <= 0.01
        recursive_min_fraction_uniquely_mapped_cells_to_recurse: If recursive
            is True, then this determines the min fraction of cells that have
            to be uniquely mapped to a clone's subclones before we accept the
            subclonal decomposition of that clone. This thus serves as a
            stopping criteria. For example, a clone might have two resolvable
            subclones comprising 35% and 31% of the clone's cells respectively,
            with the remaining 34% unmapped (i.e. discarded). Hence, if this
            argument is set to 0.7 (i.e. 70%), then we would NOT accept this
            recursive decomposition and just retain the original clone. Instead,
            if the subclones comprised 35% and 35% of the clone's cells, then
            we would exactly meet the 70% threshold and accept the recursive
            decomposition.
        recursive_min_cells_assigned: If recursive
            is True, a lower bound on the size of the subclones we want to find.
            Subclones tend to be smaller than the parent clone, so it is a good
            idea to set this lower than the original
            min_cells_assigned.
        recover_single_intbc_clones: If `True`, single-intBC clones will also be
            recovered. Use this if your data contains single-intBC clones. There
            is an inherent tradeoff in trying to recover single-intBC clones: since
            these clones have weaker statistical support, unless you are certain
            that your data has negligible levels of homoplasy, you might end up
            accidentally overmerging clones. If evolving tracer marks are
            available, we recommend using these to confirm that the single-intBC
            clones are high quality, and if there is overmerging we recommend
            splitting into different clones based on evolving lineage marks.

    Returns:
        A `results` NamedTuple of type CloneCallingAlgorithmReturnType, which contains
        the clone call in the field `clone_call`. The `clone_call` itself is also a
        NamedTuple; the most important fields are:
        - `results.clone_call.cell_to_clones_mapping (Dict[str, Set[str]])`:
            Dictionary mapping cellBCs to clone IDs. A cellBCs mapping to more than one
            clone is a multiplet, and a cellBC mapping to no clone is an empty cellBC or
            a dropped cellBC (e.g. due to low quality).
        - `results.clone_call.clone_to_cells_mapping (Dict[str, Set[str]])`:
            Dictionary clone IDs to cellBCs.
        - `results.clone_call.clone_intbc_sets (Dict[str, Set[str]])`:
            Dictionary mapping clone IDs to its set of IntBCs.
        - `results.clone_call.cell_intbc_sets (Dict[str, Set[str]])`:
            Dictionary mapping cellBCs to its set of IntBCs.
        NOTE: the cellBCs are prefixed with the sample name in the `results` (to avoid
        cellBC collision between different samples.)
    """
```

### Return type

In addition to the `clone_call`, the `results` returned by NovaClone's `clone_calling_algorithm` also contains other information such as the recursive clone call (when NovaClone is run with `recursive=True`).
Below is the exact definition of the return type.

```
CellBCType = str
IntBCType = str
UMICountType = int
CloneType = str


class CloneCallType(NamedTuple):
    clone_to_cells_mapping: Dict[CloneType, Set[CellBCType]]
    cell_to_clones_mapping: Dict[CellBCType, Set[CloneType]]
    clone_intbc_sets: Dict[CloneType, Set[IntBCType]]
    cell_intbc_sets: Dict[CellBCType, Set[IntBCType]]


class CloneCallingAlgorithmReturnType(NamedTuple):
    """
    Object returned by `clone_calling_algorithm`

    Attributes:
        clone_call: The clone call
        recursive_clone_call: The recursive clone call.
        recursive_clone_call_integration_hierarchy: Describes the subclonal
            structure of each clone in a succint way, and hence allows easy
            parsing of the recursive_clone_call. E.g.:
            {
                "clone_0": ({"AAA", "BBB"},
                {
                    "clone_0": ({"CCC"},
                    None),
                    "clone_1": ({"DDD"},
                    None),
                }),
            }
        metadata: General metadata for debugging etc.
        total_time: Total time taken
    """

    clone_call: CloneCallType
    recursive_clone_call: Dict[CloneType, "CloneCallingAlgorithmReturnType"]
    recursive_clone_call_integration_hierarchy: Any = None
    total_time: Optional[float] = None
    metadata: Any = None
```
