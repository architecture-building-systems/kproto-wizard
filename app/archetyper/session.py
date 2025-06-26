class KPrototyperSession:
    def __init__(self, name):
        self.name = name
        self.input_data = None
        self.clustered_data = None
        self.cluster_overview = None

    def is_complete(self):
        return self.input_data is not None and self.clustered_data is not None