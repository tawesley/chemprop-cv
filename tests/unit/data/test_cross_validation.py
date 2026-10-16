"""Tests for k-fold cross-validation splitting (``--cross-validation``)."""

import pytest
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from chemprop.data.splitting import make_split_indices

SPLIT_TYPES = [
    "random",
    "scaffold_balanced",
    "random_with_repeated_smiles",
    "kennard_stone",
    "kmeans",
]


@pytest.fixture
def mol_data():
    """Twelve distinct molecules over four Murcko scaffolds."""
    smis = [
        "c1ccccc1C",
        "c1ccccc1CC",
        "c1ccccc1CCC",
        "c1ccncc1C",
        "c1ccncc1CC",
        "c1ccncc1CCC",
        "C1CCNCC1C",
        "C1CCNCC1CC",
        "C1CCNCC1CCC",
        "c1ccsc1C",
        "c1ccsc1CC",
        "c1ccsc1CCC",
    ]
    return [Chem.MolFromSmiles(smi) for smi in smis]


@pytest.fixture
def mol_data_with_repeated_mols():
    """A dataset in which three molecules appear more than once."""
    smis = ["C", "CC", "CC", "CN", "CO", "CO", "CCN", "CCO", "CCO"]
    return [Chem.MolFromSmiles(smi) for smi in smis]


def _replicates(mols, split, cross_validation, num_replicates=3, seed=0):
    return list(
        zip(
            *make_split_indices(
                mols=mols,
                split=split,
                seed=seed,
                num_replicates=num_replicates,
                cross_validation=cross_validation,
            )
        )
    )


@pytest.mark.parametrize("split", SPLIT_TYPES)
@pytest.mark.parametrize("cross_validation", ["cv", "cv_notest"])
def test_replicates_partition_the_dataset(mol_data, split, cross_validation):
    """Every replicate must partition the data: no overlap, nothing dropped."""
    for train, val, test in _replicates(mol_data, split, cross_validation):
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0
        assert sorted(train + val + test) == list(range(len(mol_data)))


@pytest.mark.parametrize("split", SPLIT_TYPES)
@pytest.mark.parametrize("cross_validation", ["cv", "cv_notest"])
def test_every_datapoint_validated_exactly_once(mol_data, split, cross_validation):
    """The point of k-fold: each datapoint is held out for validation once."""
    val_indices = [i for _, val, _ in _replicates(mol_data, split, cross_validation) for i in val]
    assert sorted(val_indices) == list(range(len(mol_data)))


@pytest.mark.parametrize("split", SPLIT_TYPES)
def test_every_datapoint_tested_exactly_once(mol_data, split):
    """In ``cv`` the test fold rotates too, so each datapoint is tested once."""
    test_indices = [i for _, _, test in _replicates(mol_data, split, "cv") for i in test]
    assert sorted(test_indices) == list(range(len(mol_data)))


@pytest.mark.parametrize("split", SPLIT_TYPES)
def test_cv_notest_leaves_test_empty(mol_data, split):
    """``cv_notest`` trains on everything but the validation fold."""
    for train, val, test in _replicates(mol_data, split, "cv_notest"):
        assert test == []
        assert len(train) == len(mol_data) - len(val)


def test_scaffolds_do_not_leak_between_folds(mol_data):
    """A scaffold must stay inside one fold, as it does for a single scaffold split."""
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=mol) for mol in mol_data]
    for groups in _replicates(mol_data, "scaffold_balanced", "cv"):
        seen = [{scaffolds[i] for i in group} for group in groups if group]
        for i, left in enumerate(seen):
            for right in seen[i + 1 :]:
                assert left & right == set()


def test_repeated_smiles_do_not_leak_between_folds(mol_data_with_repeated_mols):
    """Likewise, copies of one SMILES all land in the same fold."""
    smis = [Chem.MolToSmiles(mol) for mol in mol_data_with_repeated_mols]
    for groups in _replicates(mol_data_with_repeated_mols, "random_with_repeated_smiles", "cv"):
        seen = [{smis[i] for i in group} for group in groups if group]
        for i, left in enumerate(seen):
            for right in seen[i + 1 :]:
                assert left & right == set()


@pytest.mark.parametrize("num_replicates", [2, 3, 4, 6])
def test_fold_count_follows_num_replicates(mol_data, num_replicates):
    assert len(_replicates(mol_data, "random", "cv_notest", num_replicates)) == num_replicates


def test_same_seed_gives_same_folds(mol_data):
    assert _replicates(mol_data, "random", "cv", seed=0) == _replicates(
        mol_data, "random", "cv", seed=0
    )


def test_different_seed_gives_different_folds(mol_data):
    assert _replicates(mol_data, "random", "cv", seed=0) != _replicates(
        mol_data, "random", "cv", seed=1
    )


@pytest.mark.parametrize("alias", ["cv_notest", "cv-no-test", "CV_No_Test", "CV-NOTEST"])
def test_cv_notest_aliases_accepted(mol_data, alias):
    _, _, test = make_split_indices(
        mols=mol_data, num_replicates=3, cross_validation=alias
    )
    assert all(fold == [] for fold in test)


def test_unknown_cross_validation_mode_raises(mol_data):
    with pytest.raises(ValueError, match="must be one of"):
        make_split_indices(mols=mol_data, num_replicates=3, cross_validation="kfold")


@pytest.mark.parametrize(
    "cross_validation,num_replicates", [("cv", 1), ("cv", 2), ("cv_notest", 1)]
)
def test_too_few_folds_raises(mol_data, cross_validation, num_replicates):
    """``cv`` needs a train, a validation and a test fold; ``cv_notest`` needs two."""
    with pytest.raises(ValueError, match="invalid number of CV folds"):
        make_split_indices(
            mols=mol_data, num_replicates=num_replicates, cross_validation=cross_validation
        )


def test_more_folds_than_datapoints_raises(mol_data):
    with pytest.raises(ValueError, match="invalid number of CV folds"):
        make_split_indices(
            mols=mol_data, num_replicates=len(mol_data) + 1, cross_validation="cv"
        )


def test_more_folds_than_split_groups_raises(mol_data):
    """Only four scaffolds exist, so a fifth scaffold fold cannot be filled."""
    with pytest.raises(ValueError, match="only 4 split groups"):
        make_split_indices(
            mols=mol_data,
            split="scaffold_balanced",
            num_replicates=5,
            cross_validation="cv",
        )
