from scipy.ndimage import gaussian_filter

import numpy as np


def dfs(grid, x, y, z, value, visited, region):
    directions = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
    grid_shape = grid.shape
    stack = [(x, y, z)]
    while stack:
        cur_x, cur_y, cur_z = stack.pop()
        for dx, dy, dz in directions:
            nx, ny, nz = cur_x + dx, cur_y + dy, cur_z + dz
            if (0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1] and 0 <= nz < grid_shape[2] and
                grid[nx, ny, nz] == value and (nx, ny, nz) not in visited):
                visited.add((nx, ny, nz))
                region.append((nx, ny, nz))
                stack.append((nx, ny, nz))

def find_connected_regions(grid, value=1):
    connected_regions = []
    visited = set()
    X,Y,Z=grid.shape
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if grid[x, y, z] == value and (x, y, z) not in visited:
                    region = []
                    dfs(grid, x, y, z, value, visited, region)
                    connected_regions.append(region)
    return connected_regions


