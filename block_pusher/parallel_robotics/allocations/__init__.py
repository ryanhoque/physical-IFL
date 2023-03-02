from .random_allocation import RandomAllocation
from .CRU_allocation import CRUAllocation

# Mappings from CLI option strings to allocation strategies
allocation_map = {
    "random": RandomAllocation,
    "CRU": CRUAllocation
}

allocation_cfg_map = {
    "CRU": "CRU_allocation.yaml",
    "random": "random_allocation.yaml"
}