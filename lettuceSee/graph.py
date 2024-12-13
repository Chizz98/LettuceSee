#!/usr/bin/env python3
"""
Author: Chris Dijkstra
Date: 13/12/2024

Functions for creating and analyzing graphs from skeletonized binary images.
"""

import skimage as sk
import numpy as np
import networkx as nx


class SkeletonNetwork:
    """ Class used to create network representations of skeletonized images

    :param skel_im: A skeletonized image (for example output from skimage.morphology.skeletonize)
    """
    network: nx.Graph
    "**Attribute**: the network representation of the skeleton image"
    skel_im: np.ndarray[int, ...]
    "**Attribute**: Skeleton image with nodes as 2, edges as 1 and background as 0"
    label_edge_dict: dict[int, tuple[int, int]]
    "**Attribute**: LUT for edge labels in labelled_edges and the corresponding edge"
    labelled_edges: np.ndarray[int, ...]
    "**Attribute**: Skeleton image with all edges labelled sequentially"

    def __init__(self, skel_im: np.ndarray[int, ...]):
        """ Constructor method """
        self.skel_im = np.pad(skel_im, (1, 1))
        self.labelled_edges = np.zeros_like(self.skel_im)
        self.surr_mods = self._neighbours()
        self.nodes = {}
        self.edges = []
        self.label_edge_dict = {}

        self._mark()
        self._parse_structure()
        self.network = self.construct_network()

    def _flat_to_coord(self, flat_index) -> tuple[int, int]:
        """ Takes the index of 1D representation and returns 2D coordinates

        :param flat_index: the position in the 1D representation of a 2D array
        :return: the x and y coordinates of the  input coordinate in the 2D
            array
        """
        cols = self.skel_im.shape[1]
        x = flat_index % cols
        y = flat_index // cols
        return x, y

    def _neighbours(self) -> np.ndarray[int]:
        """ Returns index modifiers for _neighbours in 1D version of 2D array

        :return: np.array, 1D array with the index modifiers to find the
            neighbours in the flattened image
        """
        row_len = self.skel_im.shape[1]
        mod = np.array([-1, 0, 1])
        top = np.array([-row_len] * 3) + mod
        mid = np.array([-1, 1])
        bottom = (np.array([row_len] * 3) + mod)
        return np.concatenate([top, mid, bottom])

    def _mark(self) -> np.ndarray[int, ...]:
        """ Reads a skeletonized image and marks all pixels based on their neighbours

        :return: A 2d array containing 0s for background, 1s for nodes in a
            line and 2s for nodes with 1 or more than 2 connections.
        """
        flat_im = self.skel_im.ravel()
        for i in range(len(flat_im)):
            if flat_im[i] != 0:
                s = 0
                for mod in self.surr_mods:
                    if flat_im[i + mod] != 0:
                        s += 1
                if s == 2:
                    flat_im[i] = 1
                else:
                    flat_im[i] = 2
        return flat_im

    def _construct_nodes(self) -> None:
        """ Reads the marked image output by self._mark and constructs the nodes

        :return: None, appends all nodes with less or more than 2 edges to
            self.nodes
        """
        flat_im = self.skel_im.ravel()
        labelled_nodes = sk.measure.label(self.skel_im == 2).ravel()
        node_labs = set(labelled_nodes)
        node_labs.remove(0)
        for label in node_labs:
            node = tuple(np.where(labelled_nodes == label)[0])
            edge_count = 0
            for point in node:
                for mod in self.surr_mods:
                    if flat_im[point + mod] == 1:
                        edge_count += 1
            if edge_count != 2:
                self.nodes[len(self.nodes) + 1] = node
            else:
                flat_im[list(node)] = 1

    def _construct_edges(self) -> None:
        """ Creates an edge graph of a marked skeletonized image

        :return: None, appends edges as tuples to self.edges
        """
        labelled_edges = sk.measure.label(self.skel_im == 1)
        labelled_edges_flat = labelled_edges.ravel()
        # Create dictionary for edges
        edge_labels = set(labelled_edges_flat)
        edge_labels.remove(0)
        edges = {label: [] for label in edge_labels}
        # Finds all edges for each node
        for node_id, node_co in self.nodes.items():
            for point in node_co:
                for neigh in self.surr_mods + point:
                    if labelled_edges_flat[neigh] > 0:
                        edges[labelled_edges_flat[neigh]].append(node_id)
        self.edges = [tuple(sorted(val)) for val in edges.values()]
        self.labelled_edges = labelled_edges
        self.label_edge_dict = edges

    def _parse_structure(self) -> None:
        """ Reads the marked skeleton output of self._mark and annotates it

        :return: None, fills self.edges and self.nodes with edges and nodes.
        """
        # Get nodes
        self._construct_nodes()
        self._construct_edges()

    def construct_network(self) -> nx.Graph:
        """ **Method**: Takes defined nodes and edges and adds them to self.network

        :return: networkx.Graph object, contains the edges and nodes as defined
            in self.edges and self.nodes
        """
        network = nx.Graph()
        network.add_nodes_from(self.nodes.keys())
        network.add_edges_from(self.edges)
        return network

    def node_dict(self) -> dict[int, tuple[int, int]]:
        """ **Method**: Outputs a dictionary of nodes with coordinates

        :return: dict, keys are sequential integers for the nodes, values are
            tuples in the form of (x, y) with x and y being ints.
        """
        out_dict = {}
        for node in self.nodes:
            coords = self.nodes[node]
            x = []
            y = []
            for coord in coords:
                col, row = self._flat_to_coord(coord)
                x.append(col)
                y.append(row)
            x = int(sum(x) / len(x))
            y = int(sum(y) / len(y))
            out_dict[node] = x, y
        return out_dict
