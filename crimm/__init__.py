from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("crimm")
except PackageNotFoundError:  # imported from a source tree that is not installed
    __version__ = "unknown"

from crimm.Fetchers import fetch_alphafold, fetch_alphafold_from_chain, fetch_rcsb
from crimm.IO.PDBString import get_pdb_str
from crimm.Modeller.TopoFixer import ResidueFixer
from crimm.Modeller.TopoLoader import ParameterLoader, TopologyGenerator
