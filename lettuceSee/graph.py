#!/usr/bin/env python3
"""
Author: Chris Dijkstra

Contains the class SkeletonNetwork, which can be utilized to read in skele-
tonized images and output an edge graph, where each intersection of lines is
considered a node and the connecting lines the edges.
"""

from skimage import morphology, io, measure
import numpy as np
import networkx as nx


class SkeletonNetwork:
    def __init__(self, skel_im):
        self.skel_im = np.pad(skel_im, (1, 1))
        self.labelled_edges = np.zeros_like(self.skel_im)
        self.surr_mods = self._neighbours()
        self.nodes = {}
        self.edges = []
        self.label_edge_dict = {}

        self._mark()
        self._parse_structure()
        self.network = self.construct_network()

    def _flat_to_coord(self, flat_index):
        """ Takes the index of 1D representation and returns 2D coordinates

        parameters:
        flat_index -- int, the position in the 1D representation of a 2D array
        return:
        tuple, the row and column in the 2D array
        """
        cols = self.skel_im.shape[1]
        x = flat_index % cols
        y = flat_index // cols
        return x, y

    def _neighbours(self):
        """ Returns index modifiers for _neighbours in 1D version of 2D array

        parameters:
        None
        return:
        np.array, 1D array with the index modifiers to find the _neighbours in
            flattened image
        """
        row_len = self.skel_im.shape[1]
        mod = np.array([-1, 0, 1])
        top = np.array([-row_len] * 3) + mod
        mid = np.array([-1, 1])
        bottom = (np.array([row_len] * 3) + mod)
        return np.concatenate([top, mid, bottom])

    def _mark(self):
        """ Reads a skeletonized image and marks the pixels based on _neighbours

        parameters:
        None
        return:
        np.array, a 2d array containing 0s for background, 1s for nodes in a
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

    def _construct_nodes(self):
        """ Reads the marked image output by self._mark and constructs the nodes

        parameters:
        None
        returns:
        None, appends all nodes with less or more than 2 edges to self.nodes
        """
        flat_im = self.skel_im.ravel()
        labelled_nodes = measure.label(self.skel_im == 2).ravel()
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

    def _construct_edges(self):
        """ Creates an edge graph of a skeletonized image

        parameters:
        None
        returns:
        None, appends edges as tuples to self.edges
        """
        labelled_edges = measure.label(self.skel_im == 1)
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

    def _parse_structure(self):
        """ Reads the marked skeleton output of self._mark and annotates it

        parameters:
        None
        return:
        None, fills self.edges and self.nodes with edges and nodes.
        """
        # Get nodes
        self._construct_nodes()
        self._construct_edges()

    def construct_network(self):
        """ Takes defined nodes and edges and adds them to self.network

        parameters:
        None
        return:
        networkx.Graph object, contains the edges and nodes as defined in
            self.edges and self.nodes
        """
        network = nx.Graph()
        network.add_nodes_from(self.nodes.keys())
        network.add_edges_from(self.edges)
        return network

    def node_dict(self):
        """ Outputs a dictionary of nodes with coordinates

        parameters:
        None
        returns:
        dict, keys are sequential integers for the nodes, values are tuples in
            the form of (x, y) with x and y being ints.
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
            x = sum(x) / len(x)
            y = sum(y) / len(y)
            out_dict[node] = x, y
        return out_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    im_dir = "test_images/dummy_veins.png"
    image = io.imread(im_dir)[:, :, 3].astype(int)
    image[image > 0] = 1

    skeleton = morphology.skeletonize(image).astype(int)
    skeleton = morphology.remove_small_objects(skeleton)
    skel_net = SkeletonNetwork(skeleton)
    network = skel_net.construct_network()
    betweenness = nx.betweenness_centrality(network)

    _, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes[0].imshow(io.imread(im_dir))
    axes[1].imshow(skel_net.skel_im > 0, cmap="gray_r")
    x = []
    y = []
    c = []
    for key, node in skel_net.node_dict().items():
        x_p, y_p = node
        x.append(x_p)
        y.append(y_p)
        c.append(betweenness[key] > 0)
    plt.scatter(x, y, c=c)
    plt.tight_layout()
    plt.show()
