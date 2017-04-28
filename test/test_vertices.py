import numpy as np
import minimask.io.vertices as vertices

def test():
    polys = [
            [[0,0],[2,0],[2,2],[0,2]],
            [[0.5,0.5],[1.5,0.5],[1.5,1.5],[0.5,1.5]],
            ]
    weights = [1, 0]

    M = vertices.vertices_to_mask(polys, weights=weights)

    x, y = M.sample(n=1000)

    inside, w = M.get_combined_weight(x, y, operation='and')
