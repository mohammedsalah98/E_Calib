import numpy as np
from sklearn.cluster import DBSCAN
import h5py
from math import cos, sin, pi, tan
from scipy.optimize import least_squares
import warnings
import cv2
warnings.filterwarnings("ignore")

# Load h5 event sequence
file_dir = input('Specify h5 data sequence directory: ')
h5f = h5py.File(file_dir, 'r')
events = h5f['events_data'][:]
h5f.close()
H = int(input('Specify image height: '))
W = int(input('Specify image width: '))
pattern_height = int(input('Specify pattern height: '))
pattern_width = int(input('Specify pattern width: '))
diagonal_spacing = float(input('Specify pattern diagonal spacing: '))

# Initialize utility variables
t_sample = 1/30.0
events = np.delete(events, 2, axis=1)
events[:,2] = events[:,2] - events[0,2]
time_array = np.linspace(0.1, events[-1,2]-0.1, num=int((events[-1,2])/t_sample))
imgPts = []
worldPts = []

# Create calibration grid world points
def create_worldPts():
    count = 0
    world_pts = np.zeros((pattern_height*pattern_width, 3))
    vertical_spacing = 2 * diagonal_spacing * tan(pi/4.0)
    horizantal_spacing = diagonal_spacing * tan(pi/4.0)
    for i in range(pattern_width):
        for j in range(pattern_height):
            world_pts[count, :] = np.array([i*horizantal_spacing, j*vertical_spacing, 0.0])
            if (i % 2) != 0:
                world_pts[count,1] = world_pts[count,1] + horizantal_spacing
            count = count + 1
    return world_pts
    
# Jacobians
def jac_fcn(solution_vector, events_set, homo_col):
    r = solution_vector[0]
    h = solution_vector[1]
    k = solution_vector[2]
    z = solution_vector[5]
    theta = solution_vector[3]
    beta = solution_vector[4]
    x = events_set[:,0]
    y = events_set[:,1]
    t = events_set[:,2]
    jac = np.zeros((len(events_set), len(solution_vector)))
    jac[:,0] = (-2*r) * np.ones(len(events_set))
    jac[:,1] = (2*(cos(beta)**2*cos(theta) + sin(beta)**2*cos(theta))*(t*cos(beta)*sin(theta) - z*cos(beta)*sin(theta) + k*sin(beta)*sin(theta) - y*sin(beta)*sin(theta) + h*cos(beta)**2*cos(theta) + h*sin(beta)**2*cos(theta) - x*cos(beta)**2*cos(theta) - x*sin(beta)**2*cos(theta)))/((cos(beta)**2 + sin(beta)**2)**2*(cos(theta)**2 + sin(theta)**2)**2)
    jac[:,2] = (2*cos(beta)*(k*cos(beta) - y*cos(beta) - t*sin(beta) + z*sin(beta)))/(cos(beta)**2 + sin(beta)**2)**2 + (2*sin(beta)*sin(theta)*(t*cos(beta)*sin(theta) - z*cos(beta)*sin(theta) + k*sin(beta)*sin(theta) - y*sin(beta)*sin(theta) + h*cos(beta)**2*cos(theta) + h*sin(beta)**2*cos(theta) - x*cos(beta)**2*cos(theta) - x*sin(beta)**2*cos(theta)))/((cos(beta)**2 + sin(beta)**2)**2*(cos(theta)**2 + sin(theta)**2)**2)
    jac[:,3] = (2*(t*cos(beta)*cos(theta) - z*cos(beta)*cos(theta) + k*sin(beta)*cos(theta) - y*sin(beta)*cos(theta) - h*cos(beta)**2*sin(theta) - h*sin(beta)**2*sin(theta) + x*cos(beta)**2*sin(theta) + x*sin(beta)**2*sin(theta))*(t*cos(beta)*sin(theta) - z*cos(beta)*sin(theta) + k*sin(beta)*sin(theta) - y*sin(beta)*sin(theta) + h*cos(beta)**2*cos(theta) + h*sin(beta)**2*cos(theta) - x*cos(beta)**2*cos(theta) - x*sin(beta)**2*cos(theta)))/((cos(beta)**2 + sin(beta)**2)**2*(cos(theta)**2 + sin(theta)**2)**2)
    jac[:,4] = (2*(k*cos(beta)*sin(theta) - y*cos(beta)*sin(theta) - t*sin(beta)*sin(theta) + z*sin(beta)*sin(theta))*(t*cos(beta)*sin(theta) - z*cos(beta)*sin(theta) + k*sin(beta)*sin(theta) - y*sin(beta)*sin(theta) + h*cos(beta)**2*cos(theta) + h*sin(beta)**2*cos(theta) - x*cos(beta)**2*cos(theta) - x*sin(beta)**2*cos(theta)))/((cos(beta)**2 + sin(beta)**2)**2*(cos(theta)**2 + sin(theta)**2)**2) - (2*(k*cos(beta) - y*cos(beta) - t*sin(beta) + z*sin(beta))*(t*cos(beta) - z*cos(beta) + k*sin(beta) - y*sin(beta)))/(cos(beta)**2 + sin(beta)**2)**2
    jac[:,5] = (2*sin(beta)*(k*cos(beta) - y*cos(beta) - t*sin(beta) + z*sin(beta)))/(cos(beta)**2 + sin(beta)**2)**2 - (2*cos(beta)*sin(theta)*(t*cos(beta)*sin(theta) - z*cos(beta)*sin(theta) + k*sin(beta)*sin(theta) - y*sin(beta)*sin(theta) + h*cos(beta)**2*cos(theta) + h*sin(beta)**2*cos(theta) - x*cos(beta)**2*cos(theta) - x*sin(beta)**2*cos(theta)))/((cos(beta)**2 + sin(beta)**2)**2*(cos(theta)**2 + sin(theta)**2)**2)
    return jac

# Cost function
def cost_fcn(x, events_set, homo_col):
    events_set = np.append(events_set, homo_col, axis=1)
    T_sb = np.zeros((4,4))
    rx = np.array([[1, 0, 0], [0, cos(x[4]), -sin(x[4])], [0, sin(x[4]), cos(x[4])]])
    ry = np.array([[cos(x[3]), 0, sin(x[3])], [0, 1, 0], [-sin(x[3]), 0, cos(x[3])]])
    R_sb = np.matmul(rx, ry)
    T_sb[0:3,0:3] = R_sb
    T_sb[:,3] = np.array([x[1], x[2], x[5], 1.0])
    rotated_pts = np.matmul(np.linalg.inv(T_sb), events_set.transpose())
    residuals = np.square((rotated_pts[0,:])) + np.square(rotated_pts[1,:]) - np.square(x[0])
    sigma = np.std(residuals)
    mu = np.mean(residuals)
    power = -(np.square((residuals-mu)) / (2*np.square(sigma)))
    w = (1/(sigma*np.sqrt(2*pi))) * np.exp(power)
    return w*residuals

# DBSCAN clustering
def do_dbscan(events_normalized):
    db = DBSCAN(eps=0.015, min_samples=10, n_jobs=1, metric='euclidean').fit(events_normalized)
    n_clusters_ = len(np.unique(db.labels_))
    return db, n_clusters_

# Circle fitting for clusters
def do_optimization(events_array, db_clustering, n_clusters_):
    solution_all = np.zeros((n_clusters_, 6))
    for i in range(n_clusters_):
        events_set = events_array[db_clustering.labels_ == i]
        if len(events_set) != 0:
            events_set[:,2] = (events_set[:,2]-events_set[0,2]) * 1e4
            cluster_center = np.array([np.mean(events_set[:,0]), np.mean(events_set[:,1])])
            events_r = np.linalg.norm(np.transpose(np.array([events_set[:,0]-cluster_center[0], events_set[:,1]-cluster_center[1]])), ord=None, axis=1, keepdims=True)
            r_init = np.median(events_r)
            try:
                lb = np.array([np.min(events_r), cluster_center[0]-r_init, cluster_center[1]-r_init, -pi/8, -pi/8, np.mean(events_set[:,2])])
                ub = np.array([np.max(events_r), cluster_center[0]+r_init, cluster_center[1]+r_init, +pi/8, +pi/8, np.mean(events_set[:,2])+1e-12])
                x0 = np.concatenate(([r_init], cluster_center, [0.0], [0.0], [np.mean(events_set[:,2])]), axis=0)
                solution = least_squares(cost_fcn, x0, jac_fcn, args=(events_set, np.ones((len(events_set),1))), method='dogbox',loss='linear', bounds=(lb,ub))
                solution_all[i,:] = solution.x
            except:
                return solution_all
    return solution_all

# Extract calibration targets image points
def find_grid(solution, count):
    params = cv2.SimpleBlobDetector_Params()
    detector = cv2.SimpleBlobDetector_create(params)
    image = 255 * np.ones((H, W), np.uint8)
    radius = 8
    thickness = -1
    color = (0, 0)
    for i in range(len(solution)):
        center_coordinates = (int(solution[i,1]), int(solution[i,2]))
        image = cv2.circle(image, center_coordinates, radius, color, thickness)
    image = cv2.GaussianBlur(image, (3,3), 0)
    cv2.imshow('Reconstructed Circles', image)
    cv2.waitKey(1)
    ret, corners = cv2.findCirclesGrid(image, (pattern_height,pattern_width), None, flags=(cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING), blobDetector=detector)
    if ret:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.drawChessboardCorners(image, (pattern_height,pattern_width), corners, ret)
        cv2.imshow('Reconstructed Circles', image)
        cv2.waitKey(1)
        return corners
    else:
        return []

# Organize image with world points
def organize_pts(imPts_events, imPts_blob):
    organized_imgPts = np.zeros((len(imPts_blob), 2))
    for i in range(len(imPts_blob)):
        euc_pt = imPts_events[:,1:3] - imPts_blob[i,:]
        norm_imPts = np.linalg.norm(euc_pt, axis=1)
        closest_pt = np.argmin(norm_imPts)
        organized_imgPts[i,:] = imPts_events[closest_pt, 1:3]
    if organized_imgPts.shape[0] != 0:
        imgPts.append(organized_imgPts)
        worldPts.append(world_pts)
        
# Camera calibration function
def calibrateCam(imgPoints, worldPoints):
    ret, camMatrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        worldPoints, imgPoints, (H, W), None, None)
    error = np.zeros((len(imgPoints), 1))
    for i in range(len(error)):
        reprojected_pts, _ = cv2.projectPoints(worldPoints[i], rvecs[i], tvecs[i], camMatrix, dist_coeffs)
        error[i] = np.mean(np.linalg.norm(reprojected_pts[:,0,:] - imgPoints[i], axis=1))
    return error, camMatrix, dist_coeffs
     
# Camera calibration       
def do_calibration():
    imgPoints = np.array(imgPts, dtype='float32')
    worldPoints = np.array(worldPts, dtype='float32')
    error, initial_camMatrix, initial_dist_coeffs = calibrateCam(imgPoints, worldPoints)
    error_q3, error_q1 = np.percentile(error, [75, 25])
    calib_inliers = np.array([(error<(error_q3+1.5*(error_q3-error_q1))) & (error>(error_q1-1.5*(error_q3-error_q1)))])[0]
    imgPoints = imgPoints[calib_inliers[:,0]]
    worldPoints = worldPoints[calib_inliers[:,0]]
    repr_error, camMatrix, dist_coeffs = calibrateCam(imgPoints, worldPoints)
    return np.sum(repr_error)/len(imgPoints), camMatrix, dist_coeffs
    
# Main
if __name__ == "__main__":
    world_pts = create_worldPts()
    for i in range(len(time_array)):
        time_window_events_initial = np.where((events[:,2] > time_array[i]))
        events_array = events[time_window_events_initial[0][0]:time_window_events_initial[0][4000],:]
        events_normalized = np.array([(events_array[:,0])/W, (events_array[:,1])/H]).T
        db_clustering, n_clusters_ = do_dbscan(events_normalized)
        imPts_events = do_optimization(events_array, db_clustering, n_clusters_)
        imPts_blob = find_grid(imPts_events, count)
        count = count + 1
        organize_pts(imPts_events, imPts_blob)
    cv2.destroyAllWindows()
    repr_error, camMatrix, dist_coeffs = do_calibration()
    print('Mean Reprojection Error:', repr_error)
    print('Intrinsic Matrix:', camMatrix)
    print('Distortion Coefficients:', dist_coeffs)
