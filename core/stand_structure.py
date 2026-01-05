# SmartQSM - project-lightlin.github.io
# 
# Copyright (C) 2025-, YANG Jie <nj_yang_jie@foxmail.com>
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.spatial import KDTree
from typing import List, Union, Set
from utils.numpy_extra import calculate_heading_angle
from collections import OrderedDict

class StandStructure:
    def __init__(
            self, 
            *,
            x: Union[np.ndarray, List[float]], 
            y: Union[np.ndarray, List[float]], 
            names: List[str], 
            species: List[str], 
            dbhs: Union[np.ndarray, List[float]], 
            heights: Union[np.ndarray, List[float]], 
            crown_widths: Union[np.ndarray, List[float]], 
            unit_size: int = 4,
            standard_angle: float = 72
    ):
        assert len(x) == len(y) == len(names) == len(species) == len(dbhs) == len(heights) == len(crown_widths), "Inconsistent dimensions"
        assert len(x) >= 5, "At least 5 trees are required"
        self._points = np.column_stack((x, y))
        kd_tree = KDTree(self._points)
        self._num_points = self._points.shape[0]
        self._list_of_distances_to_neighbors, self._list_of_neighbors = kd_tree.query(self._points, k=unit_size + 1, p=2) # including self
        self._list_of_distances_to_neighbors = self._list_of_distances_to_neighbors.tolist()
        self._list_of_neighbors = self._list_of_neighbors.tolist()
        for i in range(self._num_points):
            for j in range(unit_size + 1):
                if self._list_of_neighbors[i][j] == i: # In case of trees with same coordinates when no filter operation was done
                    del self._list_of_neighbors[i][j]
                    del self._list_of_distances_to_neighbors[i][j]
                    break
        self._names = names
        self._species = species
        self._heights = heights
        self._dbhs = dbhs
        self._crown_widths = crown_widths
        self.unit_size = unit_size
        self._standard_angle = standard_angle
        return

    @property
    def crowdednesses(self):
        crowdednesses = np.zeros(shape=self._num_points, dtype=np.float32)
        for i in range(self._num_points):
            crown_width_i = self._crown_widths[i]
            crowdednesses[i] = 0.
            for t in range(self.unit_size):
                distance_ij = self._list_of_distances_to_neighbors[i][t]
                j = self._list_of_neighbors[i][t]
                crown_width_j = self._crown_widths[j]
                if  0.5 * (crown_width_i + crown_width_j) > distance_ij:
                    crowdednesses[i] += 1.
            crowdednesses[i] /= float(self.unit_size)
        return crowdednesses

    @property
    def minglings(self):
        minglings = np.zeros(self._num_points, dtype=np.float32)
        for i in range(self._num_points):
            species_i = self._species[i]
            for t in range(self.unit_size):
                j = self._list_of_neighbors[i][t]
                species_j = self._species[j]
                if species_i != species_j:
                    minglings[i] += 1.
            minglings[i] /= float(self.unit_size)
        return minglings
    
    @property
    def within_unit_species_richnesses(self):
        within_unit_species_richnesses = np.zeros(self._num_points, dtype=np.int32)
        for i in range(self._num_points):
            tree_types: Set[str] = set([self._species[i]])
            for t in range(self.unit_size):
                j = self._list_of_neighbors[i][t]
                tree_types.add(self._species[j])
            within_unit_species_richnesses[i] = len(tree_types)
        return within_unit_species_richnesses
    
    @property
    def diameter_dominances(self):
        diameter_dominances = np.zeros(self._num_points, dtype=np.float32)
        for i in range(self._num_points):
            dbh_i = self._dbhs[i]
            for t in range(self.unit_size):
                j = self._list_of_neighbors[i][t]
                dbh_j = self._dbhs[j]
                if dbh_i < dbh_j:
                    diameter_dominances[i] += 1.
            diameter_dominances[i] /= float(self.unit_size)
        return diameter_dominances

    @property
    def uniform_angle_indices(self):
        uniform_angle_indices = np.zeros(self._num_points, dtype=np.float32)
        for i in range(self._num_points):
            j1 = self._list_of_neighbors[i][0]
            v_i1 = self._points[j1] - self._points[i]
            dict_of_azimuths = OrderedDict()
            for t in range(1, self.unit_size):
                j = self._list_of_neighbors[i][t]
                v_ij = self._points[j] - self._points[i]
                dict_of_azimuths[j] = calculate_heading_angle(v_i1, v_ij)
            dict_of_azimuths = OrderedDict(sorted(dict_of_azimuths.items(), key=lambda item: item[1], reverse=False))
            sorted_neighbor_indices = [j1] + list(dict_of_azimuths.keys())
            for t in range(self.unit_size):
                j1 = sorted_neighbor_indices[t]
                j2 = sorted_neighbor_indices[(t + 1) % self.unit_size]
                direction_ij1 = self._points[j1] - self._points[i]
                direction_ij1 /= np.linalg.norm(direction_ij1)
                direction_ij2 = self._points[j2] - self._points[i]
                direction_ij2 /= np.linalg.norm(direction_ij2)
                angle = np.arccos(np.dot(direction_ij1, direction_ij2)) / np.pi * 180
                if angle < self._standard_angle:
                    uniform_angle_indices[i] += 1.
            uniform_angle_indices[i] /= float(self.unit_size)
        return uniform_angle_indices

    @property
    def hegyi_competition_indices(self):
        hegyi_competition_indices = np.zeros(self._num_points, dtype=np.float32)
        for i in range(self._num_points):
            dbh_i = self._dbhs[i]
            for t in range(self.unit_size):
                j = self._list_of_neighbors[i][t]
                dbh_j = self._dbhs[j]
                distance_ij = self._list_of_distances_to_neighbors[i][t]
                hegyi_competition_indices[i] += dbh_j / (dbh_i * distance_ij + 1.e-6) # In case of trees at the same position
            
        return hegyi_competition_indices 


    @property
    def opennesses(self):
        opennesses = np.zeros(self._num_points, dtype=np.float32)
        for i in range(self._num_points):
            for t in range(self.unit_size):
                j = self._list_of_neighbors[i][t]
                height_ij = self._heights[j]
                distance_ij = self._list_of_distances_to_neighbors[i][t]
                opennesses[i] += distance_ij / (height_ij + 1.e-3) #Tree must have height
            opennesses[i] /= float(self.unit_size)
        return opennesses
    
    @property
    def neighbor_names_per_point(self):
        neighbor_names_per_point = []
        for i in range(self._num_points):
            neighbor_names_of_point_i = []
            for t in range(self.unit_size):
                j = self._list_of_neighbors[i][t]
                neighbor_names_of_point_i.append(self._names[j])
            neighbor_names_per_point.append(neighbor_names_of_point_i)
        return neighbor_names_per_point
    
    @property
    def distances_to_neighbors_per_point(self):
        return self._list_of_distances_to_neighbors
    
    @property
    def tree_species_diversity_minglings(self):
        tree_species_diversity_minglings = np.zeros(self._num_points, dtype=np.float32)
        
        for i in range(self._num_points):
            species_i = self._species[i]
            
            neighboring_tree_types: Set[str] = set([])
            
            for t in range(self.unit_size):
                j = self._list_of_neighbors[i][t]
                species_j = self._species[j]
                neighboring_tree_types.add(species_j)
                if species_i != species_j:
                    tree_species_diversity_minglings[i] += 1.
            tree_species_diversity_minglings[i] = np.sqrt(
                len(neighboring_tree_types) / (self.unit_size ** 2) * tree_species_diversity_minglings[i]
            )
        return tree_species_diversity_minglings
