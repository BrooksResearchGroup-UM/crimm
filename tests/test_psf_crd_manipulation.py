"""Tests for manipulating models loaded from PSF/CRD files.

Verifies that PSF/CRD-loaded models support the same manipulation
operations as PDB-loaded models: chain truncation, ion replacement,
and point mutation.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest

from crimm.Fetchers import fetch_rcsb
from crimm.StructEntities.OrganizedModel import OrganizedModel
from crimm.Modeller.TopoLoader import TopologyGenerator
from crimm.Modeller.TopoFixer import ResidueFixer, fix_chain
from crimm.Modeller.Solvator import Solvator
from crimm.IO import write_psf, write_crd


def _remove_nonbackbone_atoms(residue):
    """Remove sidechain atoms and hydrogens, keep only backbone (N, CA, C, O)."""
    to_remove = [a.name for a in residue if a.name not in ("CA", "N", "C", "O")]
    for name in to_remove:
        residue.detach_child(name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def psf_crd_files():
    """Create PSF/CRD files for 4PTI via standard PDB workflow (runs once)."""
    structure = fetch_rcsb("4PTI")
    model = OrganizedModel(structure)
    topo = TopologyGenerator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topo.generate_model(model, build_coords=True, QUIET=True)

    tmpdir = tempfile.mkdtemp()
    psf_path = os.path.join(tmpdir, "4pti.psf")
    crd_path = os.path.join(tmpdir, "4pti.crd")
    write_psf(model, psf_path)
    write_crd(model, crd_path)
    return {"psf": psf_path, "crd": crd_path, "tmpdir": tmpdir, "ref_model": model}


@pytest.fixture(scope="module")
def solvated_psf_crd_files(psf_crd_files):
    """Create a solvated + ionized PSF/CRD for ion replacement tests."""
    # Start from a fresh PDB-loaded model (solvation modifies in place)
    structure = fetch_rcsb("4PTI")
    model = OrganizedModel(structure)
    topo = TopologyGenerator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topo.generate_model(model, build_coords=True, QUIET=True)
        solv = Solvator(model)
        solv.solvate(cutoff=7.0, box_type="cube")
        solv.add_ions(concentration=0.15, cation="SOD", anion="CLA")

    tmpdir = psf_crd_files["tmpdir"]
    psf_path = os.path.join(tmpdir, "4pti_solv.psf")
    crd_path = os.path.join(tmpdir, "4pti_solv.crd")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_psf(model, psf_path)
    write_crd(model, crd_path)
    return {"psf": psf_path, "crd": crd_path, "tmpdir": tmpdir}


def _load_fresh(psf_path, crd_path):
    """Load a fresh model from PSF/CRD files."""
    topo = TopologyGenerator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = topo.load_psf_crd(psf_path, crd_path)
    return loaded, topo


# ---------------------------------------------------------------------------
# Test 3: Chain Truncation
# ---------------------------------------------------------------------------


class TestTruncateChain:
    """Truncate a protein chain on a PSF/CRD-loaded model."""

    def test_truncate_reduces_residue_count(self, psf_crd_files):
        """Truncating removes residues outside the specified range."""
        model, topo = _load_fresh(psf_crd_files["psf"], psf_crd_files["crd"])
        chain = model.protein[0]
        original_count = len(list(chain.get_residues()))

        # Keep residues at 0-indexed positions 10 through 39 (30 residues)
        chain.truncate(start=10, end=40)
        new_count = len(list(chain.get_residues()))

        assert new_count == 30, f"Expected 30 residues, got {new_count}"
        assert new_count < original_count

    def test_truncate_reduces_atom_count(self, psf_crd_files):
        """Atom count decreases after truncation."""
        model, topo = _load_fresh(psf_crd_files["psf"], psf_crd_files["crd"])
        original_atoms = len(list(model.get_atoms()))

        chain = model.protein[0]
        chain.truncate(start=10, end=40)
        new_atoms = len(list(model.get_atoms()))

        assert new_atoms < original_atoms

    def test_truncated_model_reexports(self, psf_crd_files):
        """A truncated model can be re-exported to PSF/CRD."""
        model, topo = _load_fresh(psf_crd_files["psf"], psf_crd_files["crd"])
        chain = model.protein[0]
        chain.truncate(start=10, end=40)

        tmpdir = psf_crd_files["tmpdir"]
        out_psf = os.path.join(tmpdir, "truncated.psf")
        out_crd = os.path.join(tmpdir, "truncated.crd")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_psf(model, out_psf)
        write_crd(model, out_crd)

        assert os.path.getsize(out_psf) > 0
        assert os.path.getsize(out_crd) > 0


# ---------------------------------------------------------------------------
# Test 4: Ion Replacement
# ---------------------------------------------------------------------------


class TestReplaceIons:
    """Replace ions in a PSF/CRD-loaded solvated model."""

    def test_model_has_ions(self, solvated_psf_crd_files):
        """Solvated model loaded from PSF/CRD contains ion chains."""
        model, topo = _load_fresh(
            solvated_psf_crd_files["psf"], solvated_psf_crd_files["crd"]
        )
        ions = model.ion
        assert len(ions) > 0, "Expected ion chains in solvated model"

    def test_replace_ion_changes_resname(self, solvated_psf_crd_files):
        """replace_ion changes residue and atom names."""
        model, topo = _load_fresh(
            solvated_psf_crd_files["psf"], solvated_psf_crd_files["crd"]
        )
        ion_chains = model.ion
        assert len(ion_chains) > 0, "No ion chains found"

        # Find first SOD (sodium) ion and replace with POT (potassium)
        replaced = False
        for ion_chain in ion_chains:
            for res in ion_chain:
                if res.resname == "SOD":
                    model.replace_ion(res, "POT")
                    assert res.resname == "POT"
                    for atom in res:
                        assert atom.name == "POT"
                    replaced = True
                    break
            if replaced:
                break

        if not replaced:
            # Try replacing CLA instead
            for ion_chain in ion_chains:
                for res in ion_chain:
                    if res.resname == "CLA":
                        model.replace_ion(res, "BRO")
                        assert res.resname == "BRO"
                        replaced = True
                        break
                if replaced:
                    break

        assert replaced, "No SOD or CLA ions found to replace"

    def test_replace_ion_chain(self, solvated_psf_crd_files):
        """replace_ion_chain changes all ions in a chain."""
        model, topo = _load_fresh(
            solvated_psf_crd_files["psf"], solvated_psf_crd_files["crd"]
        )
        ion_chains = model.ion
        assert len(ion_chains) > 0

        # Find a SOD chain and replace all with POT
        for ion_chain in ion_chains:
            first_res = list(ion_chain.get_residues())[0]
            if first_res.resname == "SOD":
                model.replace_ion_chain(ion_chain, "POT")
                for res in ion_chain:
                    assert res.resname == "POT"
                break

    def test_ion_replaced_model_reexports(self, solvated_psf_crd_files):
        """Model with replaced ions can be re-exported."""
        model, topo = _load_fresh(
            solvated_psf_crd_files["psf"], solvated_psf_crd_files["crd"]
        )
        # Replace all SOD with POT
        for ion_chain in model.ion:
            first_res = list(ion_chain.get_residues())[0]
            if first_res.resname == "SOD":
                model.replace_ion_chain(ion_chain, "POT")

        tmpdir = solvated_psf_crd_files["tmpdir"]
        out_psf = os.path.join(tmpdir, "ions_replaced.psf")
        out_crd = os.path.join(tmpdir, "ions_replaced.crd")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_psf(model, out_psf)
        write_crd(model, out_crd)

        assert os.path.getsize(out_psf) > 0
        assert os.path.getsize(out_crd) > 0


# ---------------------------------------------------------------------------
# Test 5: Point Mutation
# ---------------------------------------------------------------------------


class TestPointMutation:
    """Mutate a residue on a PSF/CRD-loaded model."""

    def _find_ala_residue(self, chain):
        """Find a mid-chain ALA residue for mutation."""
        residues = list(chain.get_residues())
        for res in residues[5:-5]:  # skip termini
            if res.resname == "ALA":
                return res
        pytest.skip("No mid-chain ALA found in 4PTI chain")

    def test_mutate_ala_to_gly(self, psf_crd_files):
        """Mutate ALA→GLY: strip side chain, regenerate topology."""
        model, topo = _load_fresh(psf_crd_files["psf"], psf_crd_files["crd"])
        chain = model.protein[0]
        target = self._find_ala_residue(chain)
        resseq = target.id[1]

        # Mutate: change name and strip non-backbone atoms
        target.resname = "GLY"
        _remove_nonbackbone_atoms(target)

        # Regenerate topology for the mutated residue
        topo._load_residue_definitions(chain.chain_type, preserve=True)
        topo._generate_residue_topology(target, QUIET=True)

        # Rebuild missing atoms
        fixer = ResidueFixer()
        fixer.load_residue(target)
        fixer.remove_undefined_atoms()
        fixer.build_missing_atoms()

        assert target.resname == "GLY"
        # GLY should have no CB atom
        atom_names = [a.name for a in target]
        assert "CB" not in atom_names, "GLY should not have CB"

    def test_mutate_gly_to_ala(self, psf_crd_files):
        """Mutate GLY→ALA: needs new CB side chain built."""
        model, topo = _load_fresh(psf_crd_files["psf"], psf_crd_files["crd"])
        chain = model.protein[0]

        # Find a mid-chain GLY
        target = None
        residues = list(chain.get_residues())
        for res in residues[5:-5]:
            if res.resname == "GLY":
                target = res
                break
        if target is None:
            pytest.skip("No mid-chain GLY found in 4PTI chain")

        target.resname = "ALA"
        _remove_nonbackbone_atoms(target)

        topo._load_residue_definitions(chain.chain_type, preserve=True)
        topo._generate_residue_topology(target, QUIET=True)

        fixer = ResidueFixer()
        fixer.load_residue(target)
        fixer.remove_undefined_atoms()
        fixer.build_missing_atoms()

        assert target.resname == "ALA"
        atom_names = [a.name for a in target]
        assert "CB" in atom_names, "ALA should have CB after mutation"

    def test_mutated_model_reexports(self, psf_crd_files):
        """Model with a mutated residue can be re-exported to PSF/CRD."""
        model, topo = _load_fresh(psf_crd_files["psf"], psf_crd_files["crd"])
        chain = model.protein[0]
        target = self._find_ala_residue(chain)

        # Mutate ALA → GLY
        target.resname = "GLY"
        _remove_nonbackbone_atoms(target)
        topo._load_residue_definitions(chain.chain_type, preserve=True)
        topo._generate_residue_topology(target, QUIET=True)
        fixer = ResidueFixer()
        fixer.load_residue(target)
        fixer.remove_undefined_atoms()
        fixer.build_missing_atoms()

        tmpdir = psf_crd_files["tmpdir"]
        out_psf = os.path.join(tmpdir, "mutated.psf")
        out_crd = os.path.join(tmpdir, "mutated.crd")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_psf(model, out_psf)
        write_crd(model, out_crd)

        assert os.path.getsize(out_psf) > 0
        assert os.path.getsize(out_crd) > 0
