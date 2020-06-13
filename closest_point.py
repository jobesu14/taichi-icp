import taichi as ti 
import numpy as np
import math
from time import perf_counter

ti.init(arch=ti.cpu)

RES = 1024, 1024
NB_REF_POINTS = 1024
NB_DST_POINTS = NB_REF_POINTS

error = 0

ref_pts = ti.var(ti.f32, shape=(NB_REF_POINTS, 2))
dst_pts = ti.var(ti.f32, shape=(NB_DST_POINTS, 2))

ref_mean = ti.var(ti.f32, shape=(2))

A = ti.var(ti.f32, shape=(2, 2))
U = ti.var(ti.f32, shape=(2, 2))
S = ti.var(ti.f32, shape=(2, 2))
V = ti.var(ti.f32, shape=(2, 2))

Rot = ti.var(ti.f32, shape=(2, 2))
Trans = ti.var(ti.f32, shape=(2))

# dim1,2,3 = normalized distance for grayscale visuaization
visual_sdf = ti.Vector(3, dt=ti.f32)
ti.root.dense(ti.ij, RES).place(visual_sdf)

# dim1 = distance
# dim2,3 = direction vector
sdf = ti.Vector(3, dt=ti.f32)
ti.root.dense(ti.ij, RES).place(sdf)

@ti.kernel
def computeVisualSdf():
    """
    Create the signed distance field for visulaization
    """
    dmax = math.sqrt(RES[0]*RES[0] + RES[1]*RES[1])
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
def writeDstPtsOnVisualSdfGreen():
    """
    Write the dst point on the blue chanel of the SDF.
    """
    for n in range(NB_DST_POINTS):
        i = ti.cast(dst_pts[n,0], ti.i32)
        j = ti.cast(dst_pts[n,1], ti.i32)
        visual_sdf[i,j][1] = 1

@ti.kernel
def writeDstPtsOnVisualSdfBlue():
    """
    Write the dst point on the blue chanel of the SDF.
    """
    for n in range(NB_DST_POINTS):
        i = ti.cast(dst_pts[n,0], ti.i32)
        j = ti.cast(dst_pts[n,1], ti.i32)
        visual_sdf[i,j][2] = 1

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
def computeDstToRefTransform(dst_mean_x, dst_mean_y):
    """
    Compute the rotation matrix and translation vectors to move dst p.c. into ref p.c
    Beased on https://lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/
    """
    a = A[0,0]
    b = A[1,0]
    c = A[0,1]
    d = A[1,1]

    teta = 0.5 * ti.atan2(2*a*c + 2*b*d, a*a + b*b - c*c - d*d)

    U[0,0] =  ti.cos(teta)
    U[1,0] = -ti.sin(teta)
    U[0,1] =  ti.sin(teta)
    U[1,1] =  ti.cos(teta)

    # We don't need the Sigma matrix for ICP
    # s1 = a*a + b*b + c*c + d*d
    # s2 = ti.sqrt((a*a + b*b - c*c - d*d) * (a*a + b*b - c*c - d*d) + 4 * (a*c + b*d) * (a*c + b*d))
    # sigma1 = ti.sqrt((s1 + s2) * 0.5)
    # sigma2 = ti.sqrt((s1 - s2) * 0.5)
    # S[0,0] = sigma1
    # S[1,0] = 0
    # S[0,1] = 0
    # S[1,1] = sigma2

    phi = 0.5 * ti.atan2(2*a*b + 2*c*d, a*a - b*b + c*c - d*d)
    s11 = (a*ti.cos(teta) + c*ti.sin(teta))*ti.cos(phi) + ( b*ti.cos(teta) + d*ti.sin(teta))*ti.sin(phi)
    if s11 != 0:
        s11 = s11 / ti.abs(s11)
    s22 = (a*ti.sin(teta) - c*ti.cos(teta))*ti.sin(phi) + (-b*ti.sin(teta) + d*ti.cos(teta))*ti.cos(phi)
    if s22 != 0:
        s22 = s22 / ti.abs(s22)
    V[0,0] =  s11 * ti.cos(phi)
    V[1,0] = -s22 * ti.sin(phi)
    V[0,1] =  s11 * ti.sin(phi)
    V[1,1] =  s22 * ti.cos(phi)

    Rot[0,0] = V[0,0]*U[0,0] + V[1,0]*U[1,0]
    Rot[1,0] = V[0,0]*U[0,1] + V[1,0]*U[1,1]
    Rot[0,1] = V[0,1]*U[0,0] + V[1,1]*U[1,0]
    Rot[1,1] = V[0,1]*U[0,1] + V[1,1]*U[1,1]

    Trans[0] = ref_mean[0] - Rot[0,0]*dst_mean_x - Rot[1,0]*dst_mean_y
    Trans[1] = ref_mean[1] - Rot[0,1]*dst_mean_x - Rot[1,1]*dst_mean_y

@ti.func
def applyTransformToDst():
    for n in range(NB_DST_POINTS):
        dst_pts[n,0] = Rot[0,0] * dst_pts[n,0] + Rot[1,0] * dst_pts[n,1] + Trans[0]
        dst_pts[n,1] = Rot[0,1] * dst_pts[n,0] + Rot[1,1] * dst_pts[n,1] + Trans[1]

@ti.kernel
def compueRefDstError():
    error = ti.cast(0, ti.f32)
    for n in range(NB_DST_POINTS):
        dst_x = dst_pts[n, 0]
        dst_y = dst_pts[n, 1]
        dst_x_ind = ti.cast(dst_x, ti.i32)
        dst_y_ind = ti.cast(dst_y, ti.i32)
        error += sdf[dst_x_ind,dst_y_ind][0]
    error /= NB_DST_POINTS
    print("error = ", error)

@ti.kernel
def computeIcp():
    """
    Ues the SDF to compute the transform that matches best destination points into reference points.
    Based on custom modification of https://hal.archives-ouvertes.fr/hal-01178661/documentv page 43.
    """
    dst_mean_x = ti.cast(0, ti.f32)
    dst_mean_y = ti.cast(0, ti.f32)
    for n in range(NB_DST_POINTS):
        dst_mean_x += dst_pts[n, 0]
        dst_mean_y += dst_pts[n, 1]
    dst_mean_x /= NB_DST_POINTS
    dst_mean_y /= NB_DST_POINTS
    
    for n in range(NB_DST_POINTS):
        dst_x = dst_pts[n, 0]
        dst_y = dst_pts[n, 1]
        dst_x_ind = ti.cast(dst_x, ti.i32)
        dst_y_ind = ti.cast(dst_y, ti.i32)
        sdf_x = sdf[dst_x_ind,dst_y_ind][1]
        sdf_y = sdf[dst_x_ind,dst_y_ind][2]
    
        A[0,0] += (dst_x - dst_mean_x) * ((dst_x + sdf_x) - ref_mean[0])
        A[1,0] += (dst_x - dst_mean_x) * ((dst_y + sdf_y) - ref_mean[1])
        A[0,1] += (dst_y - dst_mean_y) * ((dst_x + sdf_x) - ref_mean[0])
        A[1,1] += (dst_y - dst_mean_y) * ((dst_y + sdf_y) - ref_mean[1])

    print("dst mean x = ", dst_mean_x)
    print("dst mean y = ", dst_mean_y)

    computeDstToRefTransform(dst_mean_x, dst_mean_y)
    print("Rot[0,0] = ", Rot[0,0])
    print("Rot[1,0] = ", Rot[1,0])
    print("Rot[0,1] = ", Rot[0,1])
    print("Rot[1,1] = ", Rot[1,1])
    print("Trans x = ", Trans[0])
    print("Trans y = ", Trans[1])

    applyTransformToDst()
    
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

def applyTransform(trans:np.array, rot:np.array, in_pts:np.array):
    out_pts = np.zeros(shape=in_pts.shape, dtype=np.float32)
    for i in range(in_pts.shape[0]):
        out_pts[i][0] = rot[0,0] * in_pts[i][0] + rot[1,0] * in_pts[i][1] + trans[0]
        out_pts[i][1] = rot[0,1] * in_pts[i][0] + rot[1,1] * in_pts[i][1] + trans[1]
    return out_pts
    

def vec2_npf32(m):
    return np.array([m[0], m[1]], dtype=np.float32)

#--------------------------------------------------------
# simulate existing incoming point cloud (ex: from LIDAR)
side_size = RES[0] / 2
center_x = RES[0] / 2
center_y = RES[0] / 2

ref_pts_numpy = createInputPointCloud(side_size, center_x, center_y, NB_REF_POINTS)
ref_pts.from_numpy(ref_pts_numpy)

alpha = math.radians(5)
rot = np.array([[math.cos(alpha),-math.sin(alpha)],[math.sin(alpha),math.cos(alpha)]])
trans = np.array([45, 21])
dst_pts_numpy = applyTransform(trans, rot, ref_pts_numpy)
dst_pts.from_numpy(dst_pts_numpy)

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

compueRefDstError()

# Show the intial state of the dst points
writeDstPtsOnVisualSdfGreen()

nb_passes = 0
while nb_passes < 10 and error < 1:
    # compute ICP
    start_time = perf_counter() 
    computeIcp()
    stop_time = perf_counter()
    print("computeIcp: %f [ms]" % ((stop_time - start_time)*1000))

    compueRefDstError()
    nb_passes += 1

# Show the final state of the dst points
writeDstPtsOnVisualSdfBlue()

#--------------------------------------------------------
# visualization
ti.imshow(visual_sdf, 'SDF ref')
# gui = ti.GUI('Closest point', RES)
# while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
#     pos = gui.get_cursor_pos()
#     gui.set_image(visual_sdf)
#     gui.show()
