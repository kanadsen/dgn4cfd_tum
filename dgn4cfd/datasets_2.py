import os
import torch
import random
import h5py
import numpy as np
from typing import Callable, Dict, Union
from enum import Enum
import requests

from .graph import Graph
from .transforms import cells_to_edge_index


  
class Dataset(torch.utils.data.Dataset):
    r"""A base class for representing a Dataset.

    Args:
        path (string): Path to the h5 file.
        transform (callable, optional): A function/transform that takes in a :obj:`graphs4cfd.graph.Graph` object
            and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        max_indegree (int, optional): The maximum number of edges per node. Applies only if the mesh is provided. (default: :obj:`None`)
        training_info (dict, optional): A dictionary containing values of type :obj:`ìnt` for the keys `n_in`,
            `step` and `T`. (default: :obj:`None`)
        idx (int, optional): The index of the simulation to load. If :obj:`None`, then all the simulations are loaded. (default: :obj:`None`)
        preload (bool, optional): If :obj:`True`, then the data is loaded in memory. If :obj:`False`, then the data
            is loaded from the h5 file at every access. (default: :obj:`False`)
    """

    def __init__(
        self,
        path:          str,
        max_indegree:  int              = None,
        transform:     Callable         = None,
        training_info: Dict             = None,
        idx:           Union[int, list] = None,
        preload:       bool             = False,
    ) -> None:
        self.path          = path
        self.max_indegree  = max_indegree
        self.transform     = transform
        self.training_info = training_info
        if self.training_info is None:
            self.training_info = {}
        self.training_info["n_in"]  = 1
        self.training_info["step"]  = 1
        self.training_sequences_length = self.training_info["n_in"] * self.training_info["step"] - self.training_info["step"] + 1
        self.preload = preload
        # Load only the given simulation idx
        if idx is not None:
            if preload == False:
                raise ValueError(
                    'If input argument to Dataset.__init__() idx is not None, then argument preload must be True.')
            h5_file = h5py.File(self.path, "r")
            self.h5_data = torch.tensor(np.array(h5_file["data"][idx]), dtype=torch.float32)
            self.h5_mesh = torch.tensor(np.array(h5_file['mesh'][idx]), dtype=torch.long) if 'mesh' in h5_file else None
            if self.h5_data.ndim == 2:
                self.h5_data = self.h5_data.unsqueeze(0)
                if self.h5_mesh is not None:
                    self.h5_mesh = self.h5_mesh.unsqueeze(0)
            h5_file.close()
        # Load all the simulations
        else:
            if self.preload:
                self.load()
            else:
                self.h5_data = None
                self.h5_mesh = None

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        if self.h5_data is not None:
            return self.h5_data.shape[0]
        else:
            h5_file = h5py.File(self.path, 'r')
            num_samples = h5_file['data'].shape[0]
            h5_file.close()
            return num_samples

    def __getitem__(
        self,
        idx: int
    ) -> Graph:
        r"""Get the idx-th training sequence."""
        assert self.training_info is not None, "Training info must be provided."
        assert "T" in self.training_info, "T must be provided in the training info."
        if isinstance(self.training_info["T"], int):
            T = self.training_info["T"]
        elif isinstance(self.training_info["T"], np.ndarray):
            T = self.training_info["T"][idx]
        else:
            raise ValueError("T must be of type int or np.ndarray.")
        sequence_start = random.randint(0, T - self.training_sequences_length)
        return self.get_sequence(idx, sequence_start, n_in=self.training_info["n_in"], step_size=self.training_info["step"], cell_list=getattr(self, 'cell_list', False))

    def get_sequence(
        self, 
        idx:            int,
        sequence_start: int = 0,
        n_in:           int = 1, 
        step_size:      int = 1, 
        cell_list:      bool = False,
    ) -> Graph:
        r"""Get the idx-th sequence.

        Args:
            idx (int): The index of the sample.
            sequence_start (int, optional): The starting index of the sequence. (default: :obj:`0`)
            n_in (int, optional): The number of input time-steps. (default: :obj:`1`)
            step_size (int, optional): The step between two consecutive time-steps. (default: :obj:`1`)
            cell_list (bool, optional): If :obj:`True`, then the mesh cells are stored in a list if the mesh is provided. (default: :obj:`False`)

        Returns:
            :obj:`graphs4cfd.graph.Graph`: The graph containing the sequence.
        """
        # Load the data
        if self.preload:
            data = self.h5_data[idx]
            mesh = self.h5_mesh[idx] if self.h5_mesh is not None else None
        else:
            h5_file = h5py.File(self.path, 'r')
            data = torch.tensor(h5_file['data'][idx], dtype=torch.float32)
            mesh = torch.tensor(h5_file['mesh'][idx], dtype=torch.long) if 'mesh' in h5_file else None
            h5_file.close()
        # Compute the indices
        idx0 = sequence_start
        idx1 = sequence_start + n_in * step_size
        # Create the graph (only point cloud)
        graph = self.data2graph(data, idx0, idx1)
        # Transform the mesh cells to graph edges if the mesh is provided
        if mesh is not None:
            # Remove the cells with only negative indices (ghost cells)
            mask = torch.logical_not((mesh < 0).all(dim=1))
            mesh = mesh[mask]
            graph.edge_index = cells_to_edge_index(mesh, max_indegree=self.max_indegree, pos=graph.pos)
            if hasattr(graph, 'pos'):
                graph.edge_attr = graph.pos[graph.edge_index[1]] - graph.pos[graph.edge_index[0]]
            if cell_list:
                mask = (mesh >= 0)
                graph.cell_list = [cell[cell_mask] for cell, cell_mask in zip(mesh, mask)]
        else:
            if self.max_indegree is not None:
                print("Warning: max_indegree parameter is not used because the mesh is not provided.")
        # Apply the transformations
        return self.transform(graph) if self.transform is not None else graph

    def load(self):
        r"""Load the dataset in memory."""
        print("Loading dataset:", self.path)
        h5_file = h5py.File(self.path, "r")
        self.h5_data = torch.tensor(np.array(h5_file["data"]), dtype=torch.float32)
        self.h5_mesh = torch.tensor(np.array(h5_file['mesh']), dtype=torch.long) if 'mesh' in h5_file else None
        h5_file.close()
        self.preload = True
  
    def data2graph(
        self,
        data: torch.Tensor,
        idx0: int,
        idx1: int,
    ) -> Graph:
        r"""Convert the data to a `Graph` object."""
        graph = Graph()
        '''
        graph.pos    = ...
        graph.glob   = ...
        graph.loc    = ...
        graph.target  = ...
        graph.omega  = ...
        .
        .
        .
        '''
        return graph
    


