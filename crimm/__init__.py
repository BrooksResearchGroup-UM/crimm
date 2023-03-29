from Bio import BiopythonWarning

class ChainConstructionException(Exception):
    """Define class ChainConstructionException."""
    pass

class AtomAltLocException(Exception):
    """Define class AtomAltLocException."""
    pass

class ChainConstructionWarning(BiopythonWarning):
    """Define class ChainConstructionWarning."""
    pass

class LigandBondOrderException(Exception):
    """Define class LigandBondOrderException."""
    pass

class SmilesQueryException(Exception):
    """Define class SmilesQueryException."""
    pass

class LigandBondOrderWarning(Warning):
    """Define class LigandBondOrderException."""
    pass

class SmilesQueryWarning(Warning):
    """Define class SmilesQueryException."""
    pass