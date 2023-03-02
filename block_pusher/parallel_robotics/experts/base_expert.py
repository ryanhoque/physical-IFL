
class ParallelExpert:
    """
    An abstract class containing an API for all parallel experts to implement.
    """

    def __init__(self, envs, cfg):
        raise NotImplementedError

    def get_action(self, state):
        """
        return the expert action for an individual agent state
        """
        raise NotImplementedError


