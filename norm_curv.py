import taichi as ti 
import numpy as np
import math
from time import perf_counter

ti.init(arch=ti.cpu)

RES = 1024, 1024
NB_REF_POINTS = 1024

pts = ti.var(ti.f32)
block1 = ti.root.bitmasked(ti.ij, RES)
block1.place(pts)

# dim1 = normal
visual_norm = ti.Vector(3, dt=ti.f32)
ti.root.bitmasked(ti.ij, RES).place(visual_norm)

# dim1 = distance
visual_curv = ti.Vector(3, dt=ti.f32)
ti.root.bitmasked(ti.ij, RES).place(visual_curv)

@ti.kernel
def placePts(np_pts: ti.ext_arr()):
    for n in range(NB_REF_POINTS):
        x = ti.cast(np_pts[n,0], ti.i32)
        y = ti.cast(np_pts[n,1], ti.i32)
        pts[x,y] = 1

@ti.kernel
def computeNormalAndCurvature():
    """
    Compute the normal and the curvature at all points voxels.
    Based on the PC limplementation:
    https://pointclouds.org/documentation/group__features.html
    """
    radius = 50
    for i,j in pts:
        nb_pts = ti.cast(0, ti.f32)
        accu_0 = ti.cast(0, ti.f32)
        accu_1 = ti.cast(0, ti.f32)
        accu_2 = ti.cast(0, ti.f32)
        accu_3 = ti.cast(0, ti.f32)
        accu_4 = ti.cast(0, ti.f32)
        accu_5 = ti.cast(0, ti.f32)
        accu_6 = ti.cast(0, ti.f32)
        accu_7 = ti.cast(0, ti.f32)
        accu_8 = ti.cast(0, ti.f32)
        z = 0
        for x in range(i-radius, i+radius):
            for y in range(j-radius, j+radius):
                if ti.is_active(block1, [x,y]):
                    accu_0 += x * x
                    accu_1 += x * y
                    accu_2 += x * z
                    accu_3 += y * y
                    accu_4 += y * z
                    accu_5 += z * z
                    accu_6 += x
                    accu_7 += y
                    accu_8 += z
                    nb_pts += 1
        accu_0 /= nb_pts
        accu_1 /= nb_pts
        accu_2 /= nb_pts
        accu_3 /= nb_pts
        accu_4 /= nb_pts
        accu_5 /= nb_pts
        accu_6 /= nb_pts
        accu_7 /= nb_pts
        accu_8 /= nb_pts
        cov_mat_0 = accu_0 - accu_6 * accu_6
        cov_mat_1 = accu_1 - accu_6 * accu_7
        cov_mat_2 = accu_2 - accu_6 * accu_8
        cov_mat_4 = accu_3 - accu_7 * accu_7
        cov_mat_5 = accu_4 - accu_7 * accu_8
        cov_mat_8 = accu_5 - accu_8 * accu_8
        cov_mat_3 = cov_mat_1
        cov_mat_6 = cov_mat_2
        cov_mat_7 = cov_mat_5

        # Compute eigen value and eigen vector
        # Make sure in [-1, 1]
        scale = ti.max(1.0,   ti.abs(cov_mat_0))
        scale = ti.max(scale, ti.abs(cov_mat_1))
        scale = ti.max(scale, ti.abs(cov_mat_2))
        scale = ti.max(scale, ti.abs(cov_mat_3))
        scale = ti.max(scale, ti.abs(cov_mat_4))
        scale = ti.max(scale, ti.abs(cov_mat_5))
        scale = ti.max(scale, ti.abs(cov_mat_6))
        scale = ti.max(scale, ti.abs(cov_mat_7))
        scale = ti.max(scale, ti.abs(cov_mat_8))
        if scale > 1.0:
            cov_mat_0 /= scale
            cov_mat_1 /= scale
            cov_mat_2 /= scale
            cov_mat_3 /= scale
            cov_mat_4 /= scale
            cov_mat_5 /= scale
            cov_mat_6 /= scale
            cov_mat_7 /= scale
            cov_mat_8 /= scale
        
        # Compute roots
        eigen_val_0 = ti.cast(0, ti.f32)
        eigen_val_1 = ti.cast(0, ti.f32)
        eigen_val_2 = ti.cast(0, ti.f32)
        
        c0 = cov_mat_0 * cov_mat_4 * cov_mat_8 \
            + 2 * cov_mat_3 * cov_mat_6 * cov_mat_7 \
            - cov_mat_0 * cov_mat_7 * cov_mat_7 \
            - cov_mat_4 * cov_mat_6 * cov_mat_6 \
            - cov_mat_8 * cov_mat_3 * cov_mat_3
        c1 = cov_mat_0 * cov_mat_4 \
            - cov_mat_3 * cov_mat_3 \
            + cov_mat_0 * cov_mat_8 \
            - cov_mat_6 * cov_mat_6 \
            + cov_mat_4 * cov_mat_8 \
            - cov_mat_7 * cov_mat_7
        c2 = cov_mat_0 + cov_mat_4 + cov_mat_8
  
        if ti.abs(c0) < 0.00001:
            eigen_val_0 = 0
            d = c2 * c2 - 4.0 * c1
            if d < 0.0:  # no real roots ! THIS SHOULD NOT HAPPEN!
                d = 0.0
            sd = ti.sqrt(d)
            eigen_val_2 = 0.5 * (c2 + sd)
            eigen_val_1 = 0.5 * (c2 - sd)
        else:
            s_inv3 = ti.cast(1.0 / 3.0, ti.f32)
            s_sqrt3 = ti.sqrt(3.0)
            c2_over_3 = c2 * s_inv3
            a_over_3 = (c1 - c2 * c2_over_3) * s_inv3
            if a_over_3 > 0:
                a_over_3 = 0
        
            half_b = 0.5 * (c0 + c2_over_3 * (2 * c2_over_3 * c2_over_3 - c1))
            q = half_b * half_b + a_over_3 * a_over_3 * a_over_3
            if q > 0:
                q = 0
        
            rho = ti.sqrt(-a_over_3)
            theta = ti.atan2(ti.sqrt(-q), half_b) * s_inv3
            cos_theta = ti.cos(theta)
            sin_theta = ti.sin(theta)
            eigen_val_0 = c2_over_3 + 2 * rho * cos_theta
            eigen_val_1 = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta)
            eigen_val_2 = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta)
            temp_swap = ti.cast(0, ti.f32)
        
            # Sort in increasing order.
            if eigen_val_0 >= eigen_val_1:
                temp_swap = eigen_val_1
                eigen_val_1 = eigen_val_0
                eigen_val_0 = temp_swap
            if eigen_val_1 >= eigen_val_2:
                temp_swap = eigen_val_2
                eigen_val_2 = eigen_val_1
                eigen_val_1 = temp_swap
                if eigen_val_0 >= eigen_val_1:
                    temp_swap = eigen_val_1
                    eigen_val_1 = eigen_val_0
                    eigen_val_0 = temp_swap
        
            if eigen_val_0 <= 0:
                eigen_val_0 = 0
                d = c2 * c2 - 4.0 * c1
                if d < 0.0:  # no real roots ! THIS SHOULD NOT HAPPEN!
                    d = 0.0
                sd = ti.sqrt(d)
                eigen_val_2 = 0.5 * (c2 + sd)
                eigen_val_1 = 0.5 * (c2 - sd)
            # end of compute roots

        eigen_value = eigen_val_1 * scale # eigen value for 2D SDF
        # eigen value for 3D SDF
        #eigen_value = eigen_val_0 * scale

        #print("eigen_val_0 ", eigen_val_0)
        #print("eigen_val_1 ", eigen_val_1)
        #print("eigen_val_2 ", eigen_val_2)
    
        # TODO
        #scaledMat.diagonal ().array () -= eigenvalues (0)
        #eigenvector = detail::getLargest3x3Eigenvector<Vector> (scaledMat).vector;

        # Compute normal vector
        visual_norm[i,j][0] = eigen_val_0 #eigen_vector[0]
        visual_norm[i,j][1] = eigen_val_1 #eigen_vector[1]
        visual_norm[i,j][2] = eigen_val_2 #eigen_vector[2]

        # Compute the curvature surface change
        eig_sum = cov_mat_0 + cov_mat_1 + cov_mat_2
        visual_curv[i,j][0] = 0
        if eig_sum != 0:
            visual_curv[i,j][0] = eigen_val_1 # true curvature is: ti.abs(eigen_value / eig_sum)
    
    
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

#--------------------------------------------------------
# actual computation

# compute normals
start_time = perf_counter() 
placePts(ref_pts_numpy)
stop_time = perf_counter()
print("placePts: %f [ms]" % ((stop_time - start_time)*1000))

# compute normals
start_time = perf_counter() 
computeNormalAndCurvature()
stop_time = perf_counter()
print("computeNormalAndCurvature: %f [ms]" % ((stop_time - start_time)*1000))

# compute normals
start_time = perf_counter() 
computeNormalAndCurvature()
stop_time = perf_counter()
print("computeNormal: %f [ms]" % ((stop_time - start_time)*1000))


#--------------------------------------------------------
# visualization
ti.imshow(visual_norm, 'Normals')
ti.imshow(visual_curv, 'Curvature')