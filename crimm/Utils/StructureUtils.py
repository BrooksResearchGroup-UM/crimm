from string import ascii_uppercase
import numpy as np

def index_to_letters(index, letter = ''):
    """Enumerate a sequence of letters based on the alphabet from a integer number.
    The length of the sequence corresponds to the number multiples of 26 from 
    the given index, and the last letter is determined by (index % 26).
    e.g. 
        index_to_letters(0) = 'A'
        index_to_letters(25) = 'Z'
        index_to_letters(26) = 'ZA'
        index_to_letters(99) = 'ZZZV'
    """
    if index < 26:
        letter += ascii_uppercase[index]
        return letter

    letter += 'Z'
    index -= 26
    return index_to_letters(index, letter)

def get_coords(entity, include_alt=False):
    """Get atom coordinates from any structure entity level"""
    if entity.level == 'A':
        return entity.coord
    return np.array([a.coord for a in entity.get_atoms(include_alt)])