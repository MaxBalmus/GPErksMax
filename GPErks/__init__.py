# import pkg_resources

# version = pkg_resources.get_distribution(__package__).version

import importlib.metadata
version = importlib.metadata.version(__package__)
