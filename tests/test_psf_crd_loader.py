"""Test that load_psf_crd produces a model equivalent to the PDB workflow."""

import os
import tempfile
import warnings

import pytest

from crimm.Fetchers import fetch_rcsb
from crimm.StructEntities.OrganizedModel import OrganizedModel
from crimm.Modeller.TopoLoader import TopologyGenerator
from crimm.IO import write_psf, write_crd


def _create_reference_model(pdb_id):
    """Create a reference model via the standard PDB workflow."""
    structure = fetch_rcsb(pdb_id)
    model = OrganizedModel(structure)
    topo = TopologyGenerator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topo.generate_model(model, build_coords=True, QUIET=True)
    return model, topo


@pytest.fixture(scope="module")
def reference_4pti():
    """4PTI reference model + PSF/CRD files on disk."""
    model, topo = _create_reference_model("4PTI")
    tmpdir = tempfile.mkdtemp()
    psf_path = os.path.join(tmpdir, "4pti.psf")
    crd_path = os.path.join(tmpdir, "4pti.crd")
    write_psf(model, psf_path)
    write_crd(model, crd_path)
    return {
        "model": model,
        "topo": topo,
        "psf_path": psf_path,
        "crd_path": crd_path,
        "tmpdir": tmpdir,
    }


class TestLoadPsfCrd:
    """Tests for TopologyGenerator.load_psf_crd()."""

    def test_returns_organized_model(self, reference_4pti):
        """load_psf_crd returns an OrganizedModel."""
        topo = TopologyGenerator()
        loaded = topo.load_psf_crd(
            reference_4pti["psf_path"],
            reference_4pti["crd_path"],
        )
        assert isinstance(loaded, OrganizedModel)

    def test_residue_topo_definitions_populated(self, reference_4pti):
        """Every residue has a topo_definition after loading."""
        topo = TopologyGenerator()
        loaded = topo.load_psf_crd(
            reference_4pti["psf_path"],
            reference_4pti["crd_path"],
        )
        for chain in loaded:
            for residue in chain.get_residues():
                assert residue.topo_definition is not None, (
                    f"Residue {residue.resname} {residue.id} missing topo_definition"
                )

    def test_atom_topo_definitions_populated(self, reference_4pti):
        """Atoms that match the RTF definition have topo_definition set."""
        topo = TopologyGenerator()
        loaded = topo.load_psf_crd(
            reference_4pti["psf_path"],
            reference_4pti["crd_path"],
        )
        atoms_with_topo = 0
        for atom in loaded.get_atoms():
            if atom.topo_definition is not None:
                atoms_with_topo += 1
        # Most atoms should have topo_definition (some terminal atoms may not)
        total_atoms = len(list(loaded.get_atoms()))
        assert atoms_with_topo > total_atoms * 0.9, (
            f"Only {atoms_with_topo}/{total_atoms} atoms have topo_definition"
        )

    def test_atom_charges_from_psf(self, reference_4pti):
        """Atom charges match the PSF values (reflecting patches)."""
        ref_model = reference_4pti["model"]
        topo = TopologyGenerator()
        loaded = topo.load_psf_crd(
            reference_4pti["psf_path"],
            reference_4pti["crd_path"],
        )
        ref_atoms = list(ref_model.get_atoms())
        loaded_atoms = list(loaded.get_atoms())
        assert len(ref_atoms) == len(loaded_atoms)
        for ref_atom, loaded_atom in zip(ref_atoms, loaded_atoms):
            if ref_atom.topo_definition and loaded_atom.topo_definition:
                assert (
                    abs(
                        ref_atom.topo_definition.charge
                        - loaded_atom.topo_definition.charge
                    )
                    < 1e-4
                ), (
                    f"Charge mismatch for {loaded_atom.name}: "
                    f"{loaded_atom.topo_definition.charge} vs "
                    f"{ref_atom.topo_definition.charge}"
                )

    def test_chain_topology_exists(self, reference_4pti):
        """Each chain has a topology object."""
        topo = TopologyGenerator()
        loaded = topo.load_psf_crd(
            reference_4pti["psf_path"],
            reference_4pti["crd_path"],
        )
        for chain in loaded:
            assert chain.topology is not None, f"Chain {chain.id} missing topology"

    def test_model_topology_exists(self, reference_4pti):
        """Model has a ModelTopology."""
        topo = TopologyGenerator()
        loaded = topo.load_psf_crd(
            reference_4pti["psf_path"],
            reference_4pti["crd_path"],
        )
        assert loaded.topology is not None

    def test_chain_classification_matches(self, reference_4pti):
        """OrganizedModel classifies chains the same as PDB path."""
        ref_model = reference_4pti["model"]
        topo = TopologyGenerator()
        loaded = topo.load_psf_crd(
            reference_4pti["psf_path"],
            reference_4pti["crd_path"],
        )
        ref_types = sorted(getattr(c, "chain_type", "Unknown") for c in ref_model)
        loaded_types = sorted(getattr(c, "chain_type", "Unknown") for c in loaded)
        assert ref_types == loaded_types

    def test_coordinates_preserved(self, reference_4pti):
        """Atom coordinates survive the round-trip."""
        import numpy as np

        ref_model = reference_4pti["model"]
        topo = TopologyGenerator()
        loaded = topo.load_psf_crd(
            reference_4pti["psf_path"],
            reference_4pti["crd_path"],
        )
        ref_coords = [a.coord for a in ref_model.get_atoms()]
        loaded_coords = [a.coord for a in loaded.get_atoms()]
        assert len(ref_coords) == len(loaded_coords)
        for ref_c, load_c in zip(ref_coords, loaded_coords):
            assert np.allclose(ref_c, load_c, atol=1e-3)

    def test_reexport_psf_crd(self, reference_4pti):
        """Model loaded from PSF/CRD can be re-exported."""
        topo = TopologyGenerator()
        loaded = topo.load_psf_crd(
            reference_4pti["psf_path"],
            reference_4pti["crd_path"],
        )
        # Re-export
        tmpdir = reference_4pti["tmpdir"]
        new_psf = os.path.join(tmpdir, "reexport.psf")
        new_crd = os.path.join(tmpdir, "reexport.crd")
        write_psf(loaded, new_psf)
        write_crd(loaded, new_crd)

        # Verify files were written
        assert os.path.exists(new_psf)
        assert os.path.exists(new_crd)
        assert os.path.getsize(new_psf) > 0
        assert os.path.getsize(new_crd) > 0
