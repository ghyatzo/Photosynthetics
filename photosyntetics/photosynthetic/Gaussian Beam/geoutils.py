# Imports
import numpy as np;

def ray_grid_propagation(enter, exit, N, c):
    """
    Calculates contribution of a ray entering at enter and leaving at exit,
    on an N x N grid.
    enter and exit must be normalized to [0,1] x [0,1].
    """
    steps = c*N;
    A_sub = np.zeros((N,N));
    for i in range(0, int(steps)):
        per = i/int(steps);
        current = enter+per*(exit-enter);
        x_ind = min(int(current[0]*N),N-1);
        y_ind = min(int(current[1]*N),N-1);
        A_sub[x_ind, y_ind] = A_sub[x_ind, y_ind]+1;

    return A_sub/(c);
 
'''
https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
'''
def ray_plane_intersection(ray_origin, ray_direction, plane_normal, plane_origin):
    '''
        Assumes that ray_direction and plane_normal are normalized
    '''
    denom = np.dot(plane_normal, ray_direction); 
    if (denom > 1e-6):
        p = plane_origin - ray_origin; 
        t = np.dot(p, plane_normal) / denom; 
        if t >= 0:
            return [t];
 
    return [];
    
def ray_box_intersection(ray_origin, ray_direction, box_min, box_max):
    norm_ray_direction = ray_direction / np.linalg.norm(ray_direction);

    ts = [];
    # left
    tl = ray_plane_intersection(ray_origin, norm_ray_direction, np.array([1,0]), box_min);
    # right
    tr = ray_plane_intersection(ray_origin, norm_ray_direction, np.array([1,0]), box_max);
    for t in tl+tr:
        intersect_y = ray_origin[1] + t*norm_ray_direction[1];
        if (intersect_y >= box_min[1]) and (intersect_y <= box_max[1]):
            ts += [t];
    
    # up
    tu = ray_plane_intersection(ray_origin, norm_ray_direction, np.array([0,1]), box_max);
    # down
    td = ray_plane_intersection(ray_origin, norm_ray_direction, np.array([0,1]), box_min);
    for t in tu+td:
        intersect_x = ray_origin[0] + t*norm_ray_direction[0];
        if (intersect_x >= box_min[0]) and (intersect_x <= box_max[0]):
            ts += [t];
    
    if(len(ts) == 0):
        return [];
    tmin = np.amin(ts);
    tmax = np.amax(ts);
    
    return [ray_origin + tmin*norm_ray_direction, ray_origin + tmax*norm_ray_direction];

def ray_circle_intersection(ray_origin, direction, circle_origin, radius):
    D = np.linalg.norm(direction)
    delta = np.linalg.norm(circle_origin - ray_origin)
    D_delta = np.dot(direction, (ray_origin - circle_origin))

    diff = D_delta**2 - D**2*(delta**2 - radius**2)
    if diff >= 0:
        diff = np.sqrt(diff)
        t_parameter = [(-D_delta + diff)/D**2, (-D_delta - diff)/D**2]
        t_parameter = [t for t in t_parameter if t >= 0];
        t_parameter = np.sort(t_parameter);
        intersection_points = [ray_origin + t * direction for t in t_parameter]
        return intersection_points
    else:
        return []