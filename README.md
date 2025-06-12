# 3D Mesh Spectral Segmentation

This project implements a basic spectral clustering approach to segment
ASCII STL mesh files. The main script `spectral_segmentation.py` loads
an STL model, builds a graph of vertex connections and partitions the
mesh into a user specified number of clusters. Each cluster is written
as a separate STL file.

## Requirements
- Python 3.8+
- `numpy` (install via `pip install numpy`)

## Usage
```
python spectral_segmentation.py <input.stl> <num_clusters> [output_prefix]
```
Example:
```
python spectral_segmentation.py model.stl 3 segments/segment
```
This command creates files `segments/segment_0.stl`,
`segments/segment_1.stl`, etc.

## Testing
Run unit tests with:
```
python -m unittest test_spectral.py
```
