"""PSF and CRD writing, reading, and round trips."""

import warnings

import pytest

from crimm.IO import (
    CRDParser,
    get_crd_str,
    get_psf_str,
    read_psf,
    write_crd,
    write_psf,
)
from crimm.Modeller import TopologyGenerator

# Fixtures whose PSF and CRD agree today. The others are covered by the xfail tests below.
CONSISTENT = ["1ubq", "1crn", "2igd"]


@pytest.fixture
def written(shared_prepared, tmp_path):
    """Factory: write PSF and CRD for a fixture, return (model, psf_path, crd_path)."""

    def write(pdb_id):
        model = shared_prepared(pdb_id)
        psf_path = tmp_path / f"{pdb_id}.psf"
        crd_path = tmp_path / f"{pdb_id}.crd"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_psf(model, str(psf_path))
            write_crd(model, str(crd_path))
        return model, str(psf_path), str(crd_path)

    return write


def crd_atoms(crd_path):
    return list(CRDParser().get_structure(crd_path).get_atoms())


@pytest.mark.parametrize("pdb_id", CONSISTENT)
def test_psf_matches_model(written, pdb_id):
    model, psf_path, _ = written(pdb_id)
    psf = read_psf(psf_path)
    topology = model.topology
    assert len(psf.atoms) == len(list(model.get_atoms()))
    assert len(psf.bonds) == len(topology.bonds)
    assert len(psf.angles) == len(topology.angles)
    assert len(psf.dihedrals) == len(topology.dihedrals)
    assert len(psf.impropers) == len(topology.impropers)
    assert psf.extended and psf.xplor


@pytest.mark.parametrize("pdb_id", CONSISTENT)
def test_psf_charge_matches_model(written, pdb_id):
    model, psf_path, _ = written(pdb_id)
    psf_charge = sum(atom.charge for atom in read_psf(psf_path).atoms)
    model_charge = sum(atom.topo_definition.charge for atom in model.get_atoms())
    assert psf_charge == pytest.approx(model_charge, abs=1e-4)
    assert psf_charge == pytest.approx(round(psf_charge), abs=1e-4)


@pytest.mark.parametrize("pdb_id", CONSISTENT)
def test_psf_and_crd_have_the_same_atoms(written, pdb_id):
    _, psf_path, crd_path = written(pdb_id)
    assert len(read_psf(psf_path).atoms) == len(crd_atoms(crd_path))


def test_psf_segids(written):
    _, psf_path, _ = written("1ubq")
    assert {atom.segid for atom in read_psf(psf_path).atoms} == {"PROA", "SOLV"}


def test_protein_psf_has_one_cmap_per_residue(written):
    _, psf_path, _ = written("1ubq")
    psf = read_psf(psf_path)
    assert psf.has_cmap
    assert len(psf.cmap) == 76


def test_bond_indices_are_valid(written):
    _, psf_path, _ = written("1crn")
    psf = read_psf(psf_path)
    n_atoms = len(psf.atoms)
    for i, j in psf.bonds:
        assert 1 <= i <= n_atoms and 1 <= j <= n_atoms and i != j


def test_string_and_file_output_agree(shared_prepared, tmp_path):
    """Apart from the title block (timestamp, user), both routes give the same text."""
    model = shared_prepared("1crn")
    path = tmp_path / "out.crd"
    write_crd(model, str(path))

    def body(text):
        return [line for line in text.splitlines() if not line.startswith("*")]

    assert body(path.read_text()) == body(get_crd_str(model))
    assert "PSF" in get_psf_str(model).splitlines()[0]


def atom_key(atom):
    return atom.parent.segid, atom.parent.id[1], atom.name


@pytest.mark.parametrize("pdb_id", CONSISTENT)
def test_crd_round_trip_preserves_coordinates(written, pdb_id):
    model, _, crd_path = written(pdb_id)
    original = sorted(model.get_atoms(), key=atom_key)
    reread = sorted(crd_atoms(crd_path), key=atom_key)
    assert [atom_key(atom) for atom in reread] == [atom_key(atom) for atom in original]
    for before, after in zip(original, reread):
        assert after.coord == pytest.approx(before.coord, abs=1e-5), atom_key(before)


@pytest.mark.parametrize("pdb_id", CONSISTENT)
def test_load_psf_crd_round_trip(written, pdb_id):
    model, psf_path, crd_path = written(pdb_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reloaded = TopologyGenerator().load_psf_crd(psf_path, crd_path, QUIET=True)
    assert type(reloaded).__name__ == "OrganizedModel"
    assert len(list(reloaded.get_atoms())) == len(list(model.get_atoms()))
    assert [len(chain) for chain in reloaded.protein] == [
        len(chain) for chain in model.protein
    ]
    original = sorted(model.get_atoms(), key=atom_key)
    reread = sorted(reloaded.get_atoms(), key=atom_key)
    assert [atom_key(atom) for atom in reread] == [atom_key(atom) for atom in original]
    for before, after in zip(original, reread):
        assert after.coord == pytest.approx(before.coord, abs=1e-5), atom_key(before)


def reload(psf_path, crd_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TopologyGenerator().load_psf_crd(psf_path, crd_path, QUIET=True)


def patches(chain):
    """(resseq, patch_with, patch_loc) of each patched residue of a chain."""
    return [
        (
            res.id[1],
            res.topo_definition.patch_with,
            getattr(res.topo_definition, "patch_loc", None),
        )
        for res in chain
        if getattr(res.topo_definition, "patch_with", None) is not None
    ]


def assert_same_definitions(old, new):
    """Every atom has the same charge and type, and each residue the same patches."""
    assert all(atom.topo_definition is not None for atom in new.get_atoms())
    assert new.total_charge == pytest.approx(old.total_charge, abs=1e-4)
    before = {atom_key(atom): atom.topo_definition for atom in old.get_atoms()}
    after = {atom_key(atom): atom.topo_definition for atom in new.get_atoms()}
    assert after.keys() == before.keys()
    for key, definition in after.items():
        assert definition.charge == pytest.approx(before[key].charge, abs=1e-6), key
        assert definition.atom_type == before[key].atom_type, key
    assert patches(new) == patches(old)
    for kind in ("bonds", "angles", "dihedrals", "impropers"):
        assert len(getattr(new.topology, kind)) == len(getattr(old.topology, kind)), (
            kind
        )


@pytest.mark.parametrize("pdb_id", CONSISTENT)
def test_load_psf_crd_restores_charges(written, pdb_id):
    """ACE and CT3 caps (and DISU on 1CRN) come back with their patch definitions."""
    model, psf_path, crd_path = written(pdb_id)
    reloaded = reload(psf_path, crd_path)
    assert_same_definitions(model.protein[0], reloaded.protein[0])


@pytest.mark.parametrize(
    "first, last, start",
    [
        ("NTER", "CTER", 1),
        ("NTER", "CT2", 10),  # GLY 10: NTER becomes GLYP
        ("ACE", "CTER", 19),  # PRO 19: ACE becomes PROP
        ("ACP", "CNEU", 19),
        ("NNEU", "PCTE", 1),
    ],
)
def test_load_psf_crd_restores_terminal_patches(
    organized_model, tmp_path, first, last, start
):
    model = organized_model("1ubq")
    chain = model.protein[0]
    for residue in list(chain):
        if residue.id[1] < start:
            chain.detach_child(residue.id)
    psf_path, crd_path = str(tmp_path / "t.psf"), str(tmp_path / "t.crd")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TopologyGenerator().generate_model(
            model, prot_first_patch=first, prot_last_patch=last, QUIET=True
        )
        write_psf(model, psf_path)
        write_crd(model, crd_path)
    reloaded = reload(psf_path, crd_path)
    assert_same_definitions(model.protein[0], reloaded.protein[0])


@pytest.mark.xfail(
    strict=True,
    reason="#162: load_psf_crd drops the disulfide bonds: the CRD has no connectivity, so "
    "model.topology has 658 bonds for 1CRN where the original has 661",
)
def test_load_psf_crd_restores_disulfide_bonds(written):
    model, psf_path, crd_path = written("1crn")
    reloaded = reload(psf_path, crd_path)
    assert len(reloaded.topology.bonds) == len(model.topology.bonds)


def test_loading_infers_dna_patches(shared_prepared):
    """5TER and DEO5 on the first residue, DEOX, then 3PHO and DEOX on the last.

    load_psf_crd cannot load 1BNA yet (#45 and the two xfail tests below), so this
    checks the patch inference of the loading step on the prepared model itself.
    """
    from crimm.IO.PSFReader import PSFAtom
    from crimm.Modeller.TopoLoader import ResidueTopologySet

    model = shared_prepared("1bna")
    generator = TopologyGenerator()
    generator.cur_defs = ResidueTopologySet("nucleic")
    for chain in model.DNA:
        psf_atom_map = {}
        for serial, atom in enumerate(chain.get_atoms(), start=1):
            definition = atom.topo_definition
            segid = (atom.parent.segid or "").strip()
            psf_atom_map[segid, str(atom.parent.id[1]), atom.name] = PSFAtom(
                serial,
                "",
                "",
                "",
                atom.name,
                definition.atom_type,
                definition.charge,
                0.0,
            )
        residues = list(chain)
        for i, residue in enumerate(residues):
            patch_loc = {0: "NTER", len(residues) - 1: "CTER"}.get(i, "MIDCHAIN")
            inferred = generator._infer_patched_definition(
                residue, generator.cur_defs[residue.resname], patch_loc, psf_atom_map
            )
            assert inferred.is_modified, residue
            assert set(inferred.atom_dict) == {atom.name for atom in residue}
            for atom in residue:
                assert inferred[atom.name].charge == pytest.approx(
                    atom.topo_definition.charge, abs=1e-6
                ), (residue, atom.name)
                assert inferred[atom.name].atom_type == atom.topo_definition.atom_type


@pytest.mark.xfail(
    strict=True,
    reason="#161: CRD-parsed DNA strands of 1BNA are organized as four Ligand chains, so "
    "load_psf_crd cannot restore DNA",
)
def test_crd_dna_is_organized_as_dna(shared_prepared, tmp_path):
    from crimm.StructEntities.OrganizedModel import OrganizedModel

    model = shared_prepared("1bna")
    crd_path = str(tmp_path / "d.crd")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        get_psf_str(model)  # assigns the segids the CRD needs
        write_crd(model, crd_path)
        structure = CRDParser(include_solvent=True, QUIET=True).get_structure(crd_path)
        loaded = OrganizedModel(structure.models[0], fetch_web_data=False)
    assert len(loaded.DNA) == 2


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="#161: load_psf_crd reads Ligand chains with ResidueTopologySet('cgenff'), which "
    "fails to parse cgenff.rtf (H_donors unpack)",
)
def test_load_path_reads_cgenff_topology():
    from crimm.Modeller.TopoLoader import ResidueTopologySet

    ResidueTopologySet("cgenff")


def test_write_psf_reports_untyped_ligand(shared_prepared, tmp_path):
    messages = write_psf(shared_prepared("3ptb"), str(tmp_path / "t.psf"))
    assert any("BEN" in message and "No topology" in message for message in messages)


@pytest.mark.xfail(
    strict=True,
    reason="#46: with an untyped ligand the PSF skips its atoms but the CRD writes them "
    "(3PTB: 3416 vs 3425), so the pair cannot be loaded together",
)
def test_psf_and_crd_agree_with_untyped_ligand(written):
    _, psf_path, crd_path = written("3ptb")
    assert len(read_psf(psf_path).atoms) == len(crd_atoms(crd_path))


@pytest.mark.xfail(
    strict=True,
    reason="#45: PSFWriter writes 394 atoms per DNA strand where the model has 383 "
    "(one duplicated atom in 11 of 12 residues)",
)
def test_dna_psf_atom_count_matches_model(written):
    model, psf_path, _ = written("1bna")
    assert len(read_psf(psf_path).atoms) == len(list(model.get_atoms()))


@pytest.mark.xfail(
    strict=True,
    reason="#45: DNA PSF total charge is +9 where the model's is -24",
)
def test_dna_psf_charge_matches_model(written):
    model, psf_path, _ = written("1bna")
    psf_charge = sum(atom.charge for atom in read_psf(psf_path).atoms)
    model_charge = sum(atom.topo_definition.charge for atom in model.get_atoms())
    assert psf_charge == pytest.approx(model_charge, abs=1e-4)
