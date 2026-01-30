from torch_geometric.data import Data
import sys
sys.path.insert(0,'/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/dgn4cfd')
from plot import *


class Graph(Data):
    r"""A data object describing a graph. Same as torch_geometric.data.Data but with some plotting methods."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch = None

    def plot_pos(self, **kwargs):
        pos(self.pos, **kwargs)

    def plot_field(self, *args, **kwargs):
        field(self.pos, bound=getattr(self, 'bound') if hasattr(self, 'bound') else None, *args, **kwargs)

    def plot_pos_field(self, *args, **kwargs):
        pos_field(self.pos, *args, **kwargs)
    
    def plot_pos_field_2D(self, *args, **kwargs):
        pos_field_2D(self.pos, *args, **kwargs)
