from collections.abc import Iterable, Sequence, Sized
import copy
from enum import auto
import logging
from math import floor

from astartes import train_test_split, train_val_test_split
from astartes.molecules import train_test_split_molecules, train_val_test_split_molecules
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from chemprop.data.datapoints import LazyMoleculeDatapoint, MoleculeDatapoint, ReactionDatapoint
from chemprop.utils.utils import EnumMapping

logger = logging.getLogger(__name__)

Datapoints = Sequence[MoleculeDatapoint] | Sequence[ReactionDatapoint]
MulticomponentDatapoints = Sequence[Datapoints]


class SplitType(EnumMapping):
    SCAFFOLD_BALANCED = auto()
    RANDOM_WITH_REPEATED_SMILES = auto()
    RANDOM = auto()
    KENNARD_STONE = auto()
    KMEANS = auto()


def make_split_indices(
    mols: Sequence[Chem.Mol] | Sized,
    split: SplitType | str = "random",
    sizes: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    num_replicates: int = 1,
    num_folds: None = None,
    cross_validation: str | None = None,
) -> tuple[list[list[int]], ...]:
    """Splits data into training, validation, and test splits.

    Parameters
    ----------
    mols : Sequence[Chem.Mol] | Sized
        Sequence of RDKit molecules to use for structure based splitting or any object with a length
        equal to the number of datapoints if using random splitting
    split : SplitType | str, optional
        Split type, one of ~chemprop.data.utils.SplitType, by default "random"
    sizes : tuple[float, float, float], optional
        3-tuple with the proportions of data in the train, validation, and test sets, by default
        (0.8, 0.1, 0.1). Set the middle value to 0 for a two way split.
    seed : int, optional
        The random seed passed to astartes, by default 0
    num_replicates : int, optional
        Number of replicates, by default 1
    num_folds : None, optional
        This argument was removed in v2.1 - use `num_replicates` instead.
    cross_validation : str | None, optional
        If "cv" or "cv_notest", use `num_replicates` as the number of cross-validation folds.

    Returns
    -------
    tuple[list[list[int]], ...]
        2- or 3-member tuple containing num_replicates length lists of training, validation, and testing indexes.

        .. important::
            Validation may or may not be present

    Raises
    ------
    ValueError
        Requested split sizes tuple not of length 3
    ValueError
        Unsupported split method requested
    """
    if num_folds is not None:
        raise RuntimeError("This argument was removed in v2.1 - use `num_replicates` instead.")
    if num_replicates == 1 and cross_validation is None:
        logger.warning(
            "The return type of make_split_indices has changed in v2.1 - see help(make_split_indices)"
        )
    if (num_splits := len(sizes)) != 3:
        raise ValueError(
            f"Specify sizes for train, validation, and test (got {num_splits} values)."
        )
    if cross_validation is not None:
        return _make_cross_validation_split_indices(
            mols, split, seed, num_replicates, cross_validation
        )
    if sizes[0] == 1.0 and sizes[1] == 0.0 and sizes[2] == 0.0:
        return (
            [list(range(len(mols)))] * num_replicates,
            [[]] * num_replicates,
            [[]] * num_replicates,
        )
    # typically include a validation set
    include_val = True
    split_fun = train_val_test_split
    mol_split_fun = train_val_test_split_molecules
    # default sampling arguments for astartes sampler
    astartes_kwargs = dict(
        train_size=sizes[0], test_size=sizes[2], return_indices=True, random_state=seed
    )
    # if no validation set, reassign the splitting functions
    if sizes[1] == 0.0 or sizes[2] == 0.0:
        # flip val and test size if test size is 0 (to bypass astartes check)
        if sizes[2] == 0.0:
            astartes_kwargs["test_size"] = sizes[1]
        include_val = False
        split_fun = train_test_split
        mol_split_fun = train_test_split_molecules
    else:
        astartes_kwargs["val_size"] = sizes[1]

    n_datapoints = len(mols)
    train_replicates, val_replicates, test_replicates = [], [], []
    for _ in range(num_replicates):
        train, val, test = None, None, None
        match SplitType.get(split):
            case SplitType.SCAFFOLD_BALANCED:
                mols_without_atommaps = []
                for mol in mols:
                    copied_mol = copy.deepcopy(mol)
                    for atom in copied_mol.GetAtoms():
                        atom.SetAtomMapNum(0)
                    mols_without_atommaps.append(copied_mol)
                result = mol_split_fun(
                    np.array(mols_without_atommaps), sampler="scaffold", **astartes_kwargs
                )
                train, val, test = _unpack_astartes_result(result, include_val)

            # Use to constrain data with the same smiles go in the same split.
            case SplitType.RANDOM_WITH_REPEATED_SMILES:
                # get two arrays: one of all the smiles strings, one of just the unique
                all_smiles = np.array([Chem.MolToSmiles(mol) for mol in mols])
                unique_smiles = np.unique(all_smiles)

                # save a mapping of smiles -> all the indices that it appeared at
                smiles_indices = {}
                for smiles in unique_smiles:
                    smiles_indices[smiles] = np.where(all_smiles == smiles)[0].tolist()

                # randomly split the unique smiles
                result = split_fun(
                    np.arange(len(unique_smiles)), sampler="random", **astartes_kwargs
                )
                train_idxs, val_idxs, test_idxs = _unpack_astartes_result(result, include_val)

                # convert these to the 'actual' indices from the original list using the dict we made
                train = sum((smiles_indices[unique_smiles[i]] for i in train_idxs), [])
                val = sum((smiles_indices[unique_smiles[j]] for j in val_idxs), [])
                test = sum((smiles_indices[unique_smiles[k]] for k in test_idxs), [])

            case SplitType.RANDOM:
                result = split_fun(np.arange(n_datapoints), sampler="random", **astartes_kwargs)
                train, val, test = _unpack_astartes_result(result, include_val)

            case SplitType.KENNARD_STONE:
                result = mol_split_fun(
                    np.array(mols),
                    sampler="kennard_stone",
                    hopts=dict(metric="jaccard"),
                    fingerprint="morgan_fingerprint",
                    fprints_hopts=dict(n_bits=2048),
                    **astartes_kwargs,
                )
                train, val, test = _unpack_astartes_result(result, include_val)

            case SplitType.KMEANS:
                result = mol_split_fun(
                    np.array(mols),
                    sampler="kmeans",
                    hopts=dict(metric="jaccard"),
                    fingerprint="morgan_fingerprint",
                    fprints_hopts=dict(n_bits=2048),
                    **astartes_kwargs,
                )
                train, val, test = _unpack_astartes_result(result, include_val)

            case _:
                raise RuntimeError("Unreachable code reached!")

        # flip val and test back if test size is 0
        if sizes[2] == 0.0:
            val, test = test, val
        train_replicates.append(train)
        val_replicates.append(val)
        test_replicates.append(test)
        astartes_kwargs["random_state"] += 1
    return train_replicates, val_replicates, test_replicates


def _make_cross_validation_split_indices(
    mols: Sequence[Chem.Mol] | Sized,
    split: SplitType | str,
    seed: int,
    num_folds: int,
    cross_validation: str,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    cross_validation = cross_validation.lower().replace("-", "_")
    if cross_validation == "cv_no_test":
        cross_validation = "cv_notest"
    if cross_validation not in {"cv", "cv_notest"}:
        raise ValueError(
            "cross_validation must be one of 'cv' or 'cv_notest' "
            f"(got {cross_validation!r})."
        )

    n_datapoints = len(mols)
    min_folds = 3 if cross_validation == "cv" else 2
    if not (min_folds <= num_folds <= n_datapoints):
        raise ValueError(
            f"invalid number of CV folds requested! got: {num_folds}, but expected "
            f"between {min_folds} and {n_datapoints}, inclusive, for {cross_validation}."
        )

    fold_units = _make_cv_fold_units(mols, split, seed, num_folds)
    if len(fold_units) < num_folds:
        raise ValueError(
            f"Cannot create {num_folds} non-empty CV folds for split type "
            f"{SplitType.get(split)!r}; only {len(fold_units)} split groups are available."
        )

    folds = _assign_units_to_folds(fold_units, num_folds)
    train_replicates, val_replicates, test_replicates = [], [], []
    for fold_idx in range(num_folds):
        test_fold_idx = fold_idx
        val_fold_idx = (fold_idx + 1) % num_folds

        val = folds[val_fold_idx]
        if cross_validation == "cv":
            test = folds[test_fold_idx]
            train = [
                idx
                for i, fold in enumerate(folds)
                if i not in {test_fold_idx, val_fold_idx}
                for idx in fold
            ]
        else:
            test = []
            train = [
                idx
                for i, fold in enumerate(folds)
                if i != val_fold_idx
                for idx in fold
            ]

        train_replicates.append(train)
        val_replicates.append(val)
        test_replicates.append(test)

    return train_replicates, val_replicates, test_replicates


def _make_cv_fold_units(
    mols: Sequence[Chem.Mol] | Sized,
    split: SplitType | str,
    seed: int,
    num_folds: int,
) -> list[list[int]]:
    n_datapoints = len(mols)
    rng = np.random.default_rng(seed)

    match SplitType.get(split):
        case SplitType.RANDOM:
            indices = np.arange(n_datapoints)
            rng.shuffle(indices)
            return [[int(i)] for i in indices]

        case SplitType.RANDOM_WITH_REPEATED_SMILES:
            all_smiles = np.array([Chem.MolToSmiles(mol) for mol in mols])
            groups = [
                np.where(all_smiles == smiles)[0].tolist() for smiles in np.unique(all_smiles)
            ]
            return _shuffle_groups_for_cv(groups, rng)

        case SplitType.SCAFFOLD_BALANCED:
            scaffold_to_indices = {}
            for i, mol in enumerate(_remove_atom_maps(mols)):
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                scaffold_to_indices.setdefault(scaffold, []).append(i)
            return _shuffle_groups_for_cv(list(scaffold_to_indices.values()), rng)

        case SplitType.KENNARD_STONE:
            result = train_test_split_molecules(
                np.array(mols),
                sampler="kennard_stone",
                train_size=(n_datapoints - 1) / n_datapoints,
                test_size=1 / n_datapoints,
                hopts=dict(metric="jaccard"),
                fingerprint="morgan_fingerprint",
                fprints_hopts=dict(n_bits=2048),
                return_indices=True,
            )
            train_idxs, _, test_idxs = _unpack_astartes_result(result, include_val=False)
            return [[int(i)] for i in train_idxs + test_idxs]

        case SplitType.KMEANS:
            from astartes.utils.aimsim_featurizer import featurize_molecules
            from sklearn.cluster import KMeans as sk_KMeans

            x = featurize_molecules(np.array(mols), "morgan_fingerprint", dict(n_bits=2048))
            labels = sk_KMeans(
                n_clusters=max(floor(n_datapoints * 0.1) + 1, num_folds),
                n_init=1,
                random_state=seed,
            ).fit(x).labels_
            groups = [np.where(labels == label)[0].tolist() for label in np.unique(labels)]
            return _shuffle_groups_for_cv(groups, rng)

        case _:
            raise RuntimeError("Unreachable code reached!")


def _remove_atom_maps(mols: Sequence[Chem.Mol]) -> list[Chem.Mol]:
    mols_without_atommaps = []
    for mol in mols:
        copied_mol = copy.deepcopy(mol)
        for atom in copied_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        mols_without_atommaps.append(copied_mol)

    return mols_without_atommaps


def _shuffle_groups_for_cv(groups: list[list[int]], rng: np.random.Generator) -> list[list[int]]:
    rng.shuffle(groups)
    return sorted(groups, key=len, reverse=True)


def _assign_units_to_folds(units: list[list[int]], num_folds: int) -> list[list[int]]:
    folds = [[] for _ in range(num_folds)]
    fold_sizes = [0] * num_folds
    for unit in units:
        fold_idx = min(range(num_folds), key=lambda i: (fold_sizes[i], i))
        folds[fold_idx].extend(int(idx) for idx in unit)
        fold_sizes[fold_idx] += len(unit)

    return folds


def _unpack_astartes_result(
    result: tuple, include_val: bool
) -> tuple[list[int], list[int], list[int]]:
    """Helper function to partition input data based on output of astartes sampler

    Parameters
    -----------
    result: tuple
        Output from call to astartes containing the split indices
    include_val: bool
        True if a validation set is included, False otherwise.

    Returns
    ---------
    train: list[int]
    val: list[int]
    .. important::
        validation possibly empty
    test: list[int]
    """
    train_idxs, val_idxs, test_idxs = [], [], []
    # astartes returns a set of lists containing the data, clusters (if applicable)
    # and indices (always last), so we pull out the indices
    if include_val:
        train_idxs, val_idxs, test_idxs = result[-3], result[-2], result[-1]
    else:
        train_idxs, test_idxs = result[-2], result[-1]
    return list(train_idxs), list(val_idxs), list(test_idxs)


def split_data_by_indices(
    data: Datapoints | MulticomponentDatapoints,
    train_indices: Iterable[Iterable[int]] | None = None,
    val_indices: Iterable[Iterable[int]] | None = None,
    test_indices: Iterable[Iterable[int]] | None = None,
):
    """Splits data into training, validation, and test groups based on split indices given."""

    train_data = _splitter_helper(data, train_indices)
    val_data = _splitter_helper(data, val_indices)
    test_data = _splitter_helper(data, test_indices)

    return train_data, val_data, test_data


def _splitter_helper(data, indices):
    if indices is None:
        return None

    if isinstance(data[0], (MoleculeDatapoint, LazyMoleculeDatapoint, ReactionDatapoint)):
        datapoints = data
        idxss = indices
        return [[datapoints[idx] for idx in idxs] for idxs in idxss]
    else:
        datapointss = data
        idxss = indices
        return [[[datapoints[idx] for idx in idxs] for datapoints in datapointss] for idxs in idxss]
