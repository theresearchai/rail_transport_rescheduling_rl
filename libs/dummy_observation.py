from flatland.envs.observations import ObservationBuilder

class DummyObservationBuilder(ObservationBuilder):
    """
    DummyObservationBuilder class which returns dummy observations
    This is used in the evaluation service
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get_many(self, handles = None) -> bool:
        return True

    def get(self, handle: int = 0) -> bool:
        return True
