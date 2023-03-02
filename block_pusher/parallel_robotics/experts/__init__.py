from .analytic_expert import AnalyticExpert
from .reset_expert import ResetExpert

# Mappings from CLI option strings to experts
expert_map = {
    "Analytic": AnalyticExpert,
    "Reset": ResetExpert
}

expert_cfg_map = {
}