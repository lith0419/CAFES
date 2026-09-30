"""Shared QM9 source identity, numeric parsing, and molecular canonicalization."""
from __future__ import annotations

import hashlib


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def numeric(value):
    return float(value.replace('*^', 'e'))


def canonical(mol, stereo=True):
    from rdkit import Chem
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=stereo)


OFFICIAL_SOURCE_FILES = {'dsgdb9nsd.xyz.tar.bz2': ('1057646', '3195389', 'ad1ebd51ee7f5b3a6e32e974e5d54012'), 'uncharacterized.txt': ('1057644', '3195404', 'a361887bacb427b8a0ce7903d92a53b4'), 'readme.txt': ('1057641', '3195392', 'c6581a03f673746528c57acfc6b79679')}
DIFFICULT_QM9_IDS = {129152, 129158, 130535, 59818, 128113, 129053, 117523, 59827, 21725, 6620, 87037}
