"""Spectral clustering based segmentation of STL mesh.

This script reads an ASCII STL file, constructs a mesh graph based on
vertex connectivity, performs spectral clustering to partition the
mesh into a user defined number of clusters and writes each cluster
into a separate STL file.

Requirements: numpy. Install with `pip install numpy`.
"""

import sys
import math
from collections import defaultdict
from typing import List, Tuple

try:
    import numpy as np
except ImportError:
    np = None


class STLMesh:
    def __init__(self, vertices: List[Tuple[float, float, float]],
                 faces: List[Tuple[int, int, int]]):
        self.vertices = vertices
        self.faces = faces

    @staticmethod
    def load_ascii(path: str) -> 'STLMesh':
        """Load a very simple ASCII STL file."""
        vertices: List[Tuple[float, float, float]] = []
        faces: List[Tuple[int, int, int]] = []
        vertex_map = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                if parts[0].lower() == 'vertex':
                    v = tuple(float(x) for x in parts[1:4])
                    if v not in vertex_map:
                        vertex_map[v] = len(vertices)
                        vertices.append(v)
                    # push vertex index on stack to form triangles
                    if 'pending' not in locals():
                        pending = []
                    pending.append(vertex_map[v])
                    if len(pending) == 3:
                        faces.append(tuple(pending))
                        pending = []
        return STLMesh(vertices, faces)

    def write_ascii(self, path: str, face_indices: List[int]):
        """Write a subset of faces to an ASCII STL file."""
        with open(path, 'w') as f:
            f.write('solid segment\n')
            for idx in face_indices:
                a, b, c = self.faces[idx]
                f.write('  facet normal 0 0 0\n')
                f.write('    outer loop\n')
                for v in (a, b, c):
                    x, y, z = self.vertices[v]
                    f.write(f'      vertex {x} {y} {z}\n')
                f.write('    endloop\n')
                f.write('  endfacet\n')
            f.write('endsolid segment\n')


def build_adjacency(mesh: STLMesh) -> np.ndarray:
    """Construct adjacency matrix based on edge connectivity."""
    n = len(mesh.vertices)
    adj = np.zeros((n, n), dtype=float)
    for a, b, c in mesh.faces:
        for u, v in ((a, b), (b, c), (c, a)):
            adj[u, v] = 1.0
            adj[v, u] = 1.0
    return adj


def spectral_cluster(adj: np.ndarray, k: int) -> np.ndarray:
    """Perform spectral clustering and return cluster labels."""
    if np is None:
        raise RuntimeError('numpy is required to run this script')
    degree = np.diag(adj.sum(axis=1))
    laplacian = degree - adj
    # Smallest k eigenvectors of Laplacian
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    idx = np.argsort(eigvals)[:k]
    embedding = eigvecs[:, idx]
    # normalize rows
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embedding = embedding / norms
    labels = kmeans(embedding, k)
    return labels


def kmeans(data: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """Simple k-means implementation."""
    n, d = data.shape
    rng = np.random.default_rng(0)
    centroids = data[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # assign step
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.all(labels == new_labels):
            break
        labels = new_labels
        # update step
        for i in range(k):
            points = data[labels == i]
            if len(points) > 0:
                centroids[i] = points.mean(axis=0)
    return labels


def segment_mesh(mesh: STLMesh, k: int) -> List[List[int]]:
    """Segment mesh into k clusters of faces."""
    adj = build_adjacency(mesh)
    labels = spectral_cluster(adj, k)
    face_labels = []
    for face in mesh.faces:
        # assign face label as majority label among its vertices
        vertex_labels = labels[list(face)]
        counts = np.bincount(vertex_labels, minlength=k)
        face_labels.append(counts.argmax())
    # gather face indices per cluster
    clusters = [[] for _ in range(k)]
    for idx, lbl in enumerate(face_labels):
        clusters[lbl].append(idx)
    return clusters


def main(argv: List[str]):
    if len(argv) < 3:
        print('Usage: python spectral_segmentation.py <input.stl> <k> [output_prefix]')
        return
    path = argv[0]
    k = int(argv[1])
    out_prefix = argv[2] if len(argv) > 2 else 'segment'
    mesh = STLMesh.load_ascii(path)
    clusters = segment_mesh(mesh, k)
    for i, face_indices in enumerate(clusters):
        mesh.write_ascii(f'{out_prefix}_{i}.stl', face_indices)
    print('Segmentation written to', ', '.join(f'{out_prefix}_{i}.stl' for i in range(k)))


if __name__ == '__main__':
    main(sys.argv[1:])
