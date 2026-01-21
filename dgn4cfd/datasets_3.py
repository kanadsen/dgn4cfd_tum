import os
import torch
import random
import h5py
import numpy as np
from typing import Callable, Dict, Union
from enum import Enum
import requests

from .graph import Graph
from glob import glob
import pyvista as pv
import re

"""
Dataset format: .vtu
Data Information: nodes

Receive several .vtu files in a specific number format and arrange them in index format

The original paper adds 5000 simulation with 101 timesteps in each simulation.

Make changes accordingly. The vtu file should contain all transient simulation equation

"""


  
class Dataset(torch.utils.data.Dataset):
    r"""A base class for representing a Dataset.

    Args:
        path (string): Path to the folder containing the .vtu files
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
        #self.preload = preload
        self.vtu_files = self.find_all_files_and_params()

        
        self.data = None
        self.mesh = None



    
    
    def find_all_files_and_params(self) -> list:
        """
        #For each sample_<idx> folder:
        #- extracts ONLY VTU files of form <idx>_0.vtu
        #- extracts ALL .txt files inside that folder

        #Returns a dict:
        #    {
        #    "000": {
        #        "vtu": [".../sample_000/000_0.vtu"],
        #        "txt": [".../sample_000/000.txt", "..."]
        #    },
        #    ...
        #    }
        """

        if not hasattr(self, "path"):
            raise AttributeError("self.path is not defined")

        folder_pattern = re.compile(r"sample_(\d{4})$")
        #print(folder_pattern)

        output = []

        for name in sorted(os.listdir(self.path)):
            full_path = os.path.join(self.path, name)
            match = folder_pattern.match(name)
            #print(name,match)

            if not (os.path.isdir(full_path) and match):
                continue

            sample_id = match.group(1)  # e.g., "000"

            # Ensure encas pattern exists
            encas_pattern = os.path.join(full_path,"**","*.encas")
            encas_files = glob(encas_patter,recursive=True)

            if len(encas_files)==0:
                continue
            # -------- Extract ONLY *_0.vtu --------
            vtu_pattern = os.path.join(full_path, "**", f"{sample_id}_5.vtu")
            vtu_files = sorted(glob(vtu_pattern, recursive=True))

            # -------- Extract ALL .txt files --------
            txt_pattern = os.path.join(full_path, "*.txt")
            txt_files = sorted(glob(txt_pattern))

            output.append((vtu_files[0],txt_files[0]))

            print(sample_id)

        return output
    

    def read_conditions(self,txt_path):
        """
        Reads a .txt file where each line contains 4 numbers:
            id  cond1  cond2  cond3
        Returns:
            A list of lists: [[cond1, cond2, cond3], ...]
        """
        conditions = []

        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue  # skip bad/empty lines

                # Convert to floats
                nums = list(map(float, parts))

                # Extract last 3 values
                cond = nums[-3:]
                conditions.append(cond)

        return conditions

        
    def mesh_to_edge_index(self,mesh):
        
        n_cells = mesh.n_cells
        edges = []
        
        for i in range(n_cells):
            
            cell = mesh.get_cell(i) # Get the ith cell
            n_edges = cell.n_edges
            
            for j in range(n_edges):
                    
                    edge = cell.get_edge(j)
                    c = edge.point_ids
                    edges.append((c[0], c[1]))
                    edges.append((c[1], c[0])) # Make it undirected
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edge_index
        

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        
        return len(self.vtu_files)

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
        data_file,text_file = self.vtu_files[idx]
        cond_input = self.read_conditions(text_file)
        mesh = pv.read(data_file)
        points = mesh.points
        #print(mesh.point_data.keys())
        target = mesh.point_data['pressure'].reshape(-1, 1) 
        
        #data = mesh.point_data(['pressure']).reshape(-1, 1) # Reshape to (-1,1)
        #data = torch.tensor(data, dtype=torch.float32) #-> Shape (N,1)

        # Create the graph (only point cloud)
        graph = self.data2graph(points,cond_input,target)
        # Transform the mesh cells to graph edges if the mesh is provided

        graph.edge_index = self.mesh_to_edge_index(mesh) # Already returns torch tensor
      
        if hasattr(graph, 'pos'):
            graph.edge_attr = graph.pos[graph.edge_index[1]] - graph.pos[graph.edge_index[0]]
        
        # Apply the transformations
        return self.transform(graph) if self.transform is not None else graph
  
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
    


class Shock(Dataset):

    def __init__(
        self,
        *args,
        T,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(T, int):
            T = np.array([T] * super().__len__())
        self.training_info = {'n_in': 1, 'step': 1, 'T': T}

    def data2graph(
        self,
        points,
        conditions,
        target
    ) -> Graph:
     
        # Build graph
        # Build graph
        graph = Graph()
        graph.pos = torch.tensor(points, dtype=torch.float32) # x, y, z
        params = torch.tensor(conditions,dtype=torch.float32).repeat(points.shape[0], 1)
        graph.glob = params[:,0:1]
        graph.loc = params[:,1:2]
        #graph.param_3 = params[:,2:3]
        #graph.loc = torch.tensor(conditions,dtype=torch.float32).repeat(points.shape[0], 1)
        graph.target = torch.tensor(target, dtype=torch.float32)
       
        return graph
