import numpy as np
import copy

class Graph():

    vertices = {}
    edges = []
    radius = 0.0
    
    def __init__(self, data, radius):
        # Set vertices and edges with list of data points and radius
        self.radius = radius

        vertices = {}
        for i, point in enumerate(data):
            vertices[i] = point
        self.vertices = copy.deepcopy(vertices)

        edges = []
        vertices_temp = copy.deepcopy(vertices)
        for vertex1 in vertices:
            v1 = vertices[vertex1]            
            vertices_temp.pop(vertex1)
            for vertex2 in vertices_temp:
                v2 = vertices_temp[vertex2]
                if (np.linalg.norm(np.array(v1) - np.array(v2)) < self.radius):
                    edges.append((vertex1, vertex2))
        self.edges = copy.deepcopy(edges)
        
        return None
        
    def generatePowerSet(self):
        n = len(self.vertices)
        for i in range(2 ** n):  # There are 2^n subsets
            subset = []
            for j in range(n):
            # Check if j-th bit in i is set (1)
                if i & (1 << j):
                    subset.append(j)
            yield subset


    def checkAdjacency(self, vertex1, vertex2):
        # check adjacency of two vertices
        if ((vertex1, vertex2) in self.edges) or ((vertex2, vertex1) in self.edges):
            return True
        else:
            return False


    def checkIndependentSet(self, subset):
        for vertex1 in subset:
            adjacencies = [self.checkAdjacency(vertex1, vertex2) for vertex2 in subset]
            if True in adjacencies:
                return False
        return True


    def checkMaxIndependentSet(self, subset):
        if (self.checkIndependentSet(subset) == False):
            return False

        not_subset = [vertex for vertex in self.vertices.keys() if vertex not in subset]
        for not_vertex in not_subset:
            new_subset = copy.deepcopy(subset)
            new_subset.append(not_vertex)
            if new_subset and self.checkIndependentSet(new_subset):
                return False
        
        return True


