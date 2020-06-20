import taichi as ti 
import numpy as np
import math
from time import perf_counter

ti.init(arch=ti.cpu)

RES = 1024, 1024
NB_REF_POINTS = 1024


ref_pts = ti.var(ti.f32, shape=(NB_REF_POINTS, 2))

ref_mean = ti.var(ti.f32, shape=(2))

# dim1,2,3 = normalized distance for grayscale visuaization
visual_sdf = ti.Vector(3, dt=ti.f32)
ti.root.dense(ti.ij, RES).place(visual_sdf)

# dim1 = distance
# dim2,3 = direction vector
sdf = ti.Vector(3, dt=ti.f32)
ti.root.dense(ti.ij, RES).place(sdf)

# dim1,2,3 = normalized laplacian for grayscale visuaization
visual_laplacian = ti.Vector(3, dt=ti.f32)
ti.root.dense(ti.ij, RES).place(visual_laplacian)

@ti.kernel
def computeVisualSdf():
    """
    Create the signed distance field for visulaization
    """
    dmax = math.sqrt(RES[0]*RES[0] + RES[1]*RES[1]) / 4
    for i,j in visual_sdf:
        dmin = ti.cast(RES[0] * RES[1], ti.f32)
        for n in range(NB_REF_POINTS):
            a = ref_pts[n,0]
            b = ref_pts[n,1]
            d = (i-a)*(i-a) + (j-b)*(j-b) 
            if d < dmin:
                dmin = d
        dmin = ti.sqrt(dmin)
        visual_sdf[i,j][0] = dmin / dmax
        visual_sdf[i,j][1] = 0 #dmin / dmax
        visual_sdf[i,j][2] = 0 #dmin / dmax
        
@ti.kernel
def computeSdf():
    """
    Create the signed distance field.
    """
    ref_mean_x = ti.cast(0, ti.f32)
    ref_mean_y = ti.cast(0, ti.f32)
    for i,j in sdf:
        dmin = ti.cast(RES[0] * RES[1], ti.f32)
        closest_index = 0
        for n in range(NB_REF_POINTS):
            x = ref_pts[n,0]
            y = ref_pts[n,1]
            if i == 0 and j == 0: # TODO is it the right place?
                ref_mean_x += x
                ref_mean_y += y
            d = (i-x)*(i-x) + (j-y)*(j-y)
            if d < dmin:
                dmin = d
                closest_index = n
        sdf[i,j][0] = ti.sqrt(dmin)
        sdf[i,j][1] = ref_pts[closest_index,0] - i
        sdf[i,j][2] = ref_pts[closest_index,1] - j
    
    ref_mean[0] = ref_mean_x / NB_REF_POINTS
    ref_mean[1] = ref_mean_y / NB_REF_POINTS
    print("ref mean x = ", ref_mean[0])
    print("ref mean y = ", ref_mean[1])

@ti.func
def getBoundaryValue(x_index, y_index):
    dx =  sdf[x_index+1, y_index-1][0] - sdf[x_index-1, y_index-1][0] \
        + sdf[x_index+1, y_index][0]   - sdf[x_index-1, y_index][0]   \
        + sdf[x_index+1, y_index+1][0] - sdf[x_index-1, y_index+1][0]
    dy =  sdf[x_index-1, y_index+1][0] - sdf[x_index-1, y_index-1][0] \
        + sdf[x_index,   y_index+1][0] - sdf[x_index,   y_index-1][0] \
        + sdf[x_index+1, y_index+1][0] - sdf[x_index+1, y_index-1][0]
    return ti.abs(ti.atan2(dy, dx) / 3.14159)

@ti.kernel
def computeVisualLaplacian():
    """
    Create the laplacian at any point using the method described here:
    - paper: https://www.cs.cmu.edu/~kmcrane/Projects/MonteCarloGeometryProcessing/paper.pdf
    - project page: https://www.cs.cmu.edu/~kmcrane/Projects/MonteCarloGeometryProcessing/index.html
    Uses the SDF (SDF and Laplacian tensor must have the same shape).
    """
    epsilon = ti.cast(1, ti.f32)
    max_iter = 32
    n_walks = ti.cast(128, ti.i32)
    for i,j in sdf:
        sum_u_x = ti.cast(0, ti.f32)
        sum_u_y = ti.cast(0, ti.f32)
        for walk in range(n_walks):
            d = sdf[i,j][0]
            n = 0
            x_index = i
            y_index = j
            while d > epsilon and n < max_iter:
                 alpha = ti.random() * 2 * 3.14159
                 x = x_index + d * ti.cos(alpha)
                 y = y_index + d * ti.sin(alpha)
                 x_index = ti.cast(x, ti.i32)
                 y_index = ti.cast(y, ti.i32)
                 d = sdf[x_index, y_index][0]
                 n += 1
            #sum_u += getBoundaryValue(x_index, y_index) 
            dx =  sdf[x_index+1, y_index-1][0] - sdf[x_index-1, y_index-1][0] \
                + sdf[x_index+1, y_index][0]   - sdf[x_index-1, y_index][0]   \
                + sdf[x_index+1, y_index+1][0] - sdf[x_index-1, y_index+1][0]
            dy =  sdf[x_index-1, y_index+1][0] - sdf[x_index-1, y_index-1][0] \
                + sdf[x_index,   y_index+1][0] - sdf[x_index,   y_index-1][0] \
                + sdf[x_index+1, y_index+1][0] - sdf[x_index+1, y_index-1][0]
            sum_u_x += ti.abs(dx)
            sum_u_y += ti.abs(dy)
        visual_laplacian[i,j][0] = sum_u_x / n_walks / ti.sqrt(sdf[i,j][0])
        visual_laplacian[i,j][1] = sum_u_y / n_walks / ti.sqrt(sdf[i,j][0])
    
def createInputPointCloud(side_size:int, center_x:int, center_y:int, nb_pts:int):
    """
    Create the input point cloud as numpy array.
    """
    in_pts = np.zeros(shape=(nb_pts,2), dtype=np.float32)
    side_nb_pts = nb_pts / 4
    ds = side_size / side_nb_pts
    for i in range(nb_pts):
        if i < side_nb_pts:
            in_pts[i][0] = center_x + i * ds - side_size * 0.5
            in_pts[i][1] = center_y + side_size / 2
        elif i < 2 * side_nb_pts:
            in_pts[i][0] = center_x + side_size / 2
            in_pts[i][1] = center_y + (i - 1*side_nb_pts) * ds - side_size * 0.5
        elif i < 3 * side_nb_pts:
            in_pts[i][0] = center_x + (i - 2*side_nb_pts) * ds - side_size * 0.5
            in_pts[i][1] = center_y - side_size / 2
        else:
            in_pts[i][0] = center_x - side_size / 2
            in_pts[i][1] = center_y + (i - 3*side_nb_pts) * ds - side_size * 0.5
    return in_pts

def vec2_npf32(m):
    return np.array([m[0], m[1]], dtype=np.float32)

def applyTransform(center_x:float, center_y:float, trans:np.array, rot:np.array, in_pts:np.array):
    out_pts = np.zeros(shape=in_pts.shape, dtype=np.float32)
    for i in range(in_pts.shape[0]):
        out_pts[i][0] = rot[0,0] * (in_pts[i][0]-center_x) + rot[1,0] * (in_pts[i][1]-center_y) + center_x + trans[0]
        out_pts[i][1] = rot[0,1] * (in_pts[i][0]-center_x) + rot[1,1] * (in_pts[i][1]-center_y) + center_y + trans[1]
    return out_pts

#--------------------------------------------------------
# simulate existing incoming point cloud (ex: from LIDAR)
side_size = RES[0] / 2
center_x = RES[0] / 2
center_y = RES[0] / 2

size1 = int(NB_REF_POINTS * 3 / 4)
ref_pts_numpy = createInputPointCloud(side_size, center_x, center_y, size1)

alpha = math.radians(-45)
rot = np.array([[math.cos(alpha),math.sin(alpha)],[-math.sin(alpha),math.cos(alpha)]])
trans = np.array([0, 0])
ref_pts_numpy = applyTransform(center_x, center_y, trans, rot, ref_pts_numpy)

size2 = NB_REF_POINTS - size1
ref_pts_numpy2 = createInputPointCloud(side_size / 2, center_x-100, center_y, size2)
ref_pts_numpy = np.append(ref_pts_numpy, ref_pts_numpy2, axis=0)

ref_pts.from_numpy(ref_pts_numpy)

#--------------------------------------------------------
# actual computation

# compute the SDF for visualization
start_time = perf_counter() 
computeVisualSdf()
stop_time = perf_counter()
print("computeVisualSdf: %f [ms]" % ((stop_time - start_time)*1000))

# compute the actual SDF
start_time = perf_counter() 
computeSdf()
stop_time = perf_counter()
print("computeSdf: %f [ms]" % ((stop_time - start_time)*1000))

# compute the laplacian field
start_time = perf_counter() 
computeVisualLaplacian()
stop_time = perf_counter()
print("computeVisualLaplacian: %f [ms]" % ((stop_time - start_time)*1000))

#--------------------------------------------------------
# visualization
ti.imshow(visual_sdf, 'SDF')
ti.imshow(visual_laplacian, 'Laplacian field')
# gui = ti.GUI('Closest point', RES)
# while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
#     pos = gui.get_cursor_pos()
#     gui.set_image(visual_sdf)
#     gui.show()
