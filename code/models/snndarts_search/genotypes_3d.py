from collections import namedtuple

Genotype = namedtuple('Genotype', 'cell cell_concat')

PRIMITIVES = [
    'skip_connect',
    'snn_b3',
    'snn_b5']
