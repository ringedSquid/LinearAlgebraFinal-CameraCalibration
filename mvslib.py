import cv2 as cv
import numpy as np
import open3d as o3d

def get_matched_points(iml, imr):
    sift = cv.SIFT_create()
    kpl, desl = sift.detectAndCompute(iml, None)
    kpr, desr = sift.detectAndCompute(imr, None)

    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(desl, desr, k=2)

    ptsl, ptsr = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            ptsl.append(kpl[m.queryIdx].pt)
            ptsr.append(kpr[m.trainIdx].pt)

    return np.array(ptsl), np.array(ptsr)

def filter_matches_with_fundamental(F, pts1, pts2, threshold=1.0):
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])  # Nx3
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])  # Nx3

    # Compute epipolar lines for each point
    lines1 = (F.T @ pts2_h.T).T  # epilines in img1 for pts2
    lines2 = (F @ pts1_h.T).T    # epilines in img2 for pts1

    # Compute symmetric epipolar distance
    def point_line_distance(line, point):
        a, b, c = line
        x, y = point
        return abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)

    distances = np.array([
        point_line_distance(lines1[i], pts1[i]) +
        point_line_distance(lines2[i], pts2[i])
        for i in range(len(pts1))
    ])

    mask = distances < threshold
    return pts1[mask], pts2[mask]

def rectify_uncalibrated(img1, img2, pts1, pts2, F):
    h, w = img1.shape[:2]
    retval, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, imgSize=(w, h))
    if not retval:
        raise ValueError("Stereo rectification failed")
    img1_rect = cv.warpPerspective(img1, H1, (w, h))
    img2_rect = cv.warpPerspective(img2, H2, (w, h))
    return img1_rect, img2_rect, H1, H2

def compute_disparity_map(imgL, imgR, num_disparities=96, block_size=5):
    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disparity

def get_dense_corr_from_disp(disp):
    h, w = disp.shape
    correspondences = []
    for y in range(h):
        for x in range(w):
            d = disp[y, x]
            if d > 0 and (x - d) >= 0:
                xR = x - d
                correspondences.append(((x, y), (xR, y)))
    
    return np.array(correspondences)

def triangulate_dense_points(Pl, Pr, correspondences):
    ptsL = np.array([pt1 for pt1, _ in correspondences], dtype=np.float32).T 
    ptsR = np.array([pt2 for _, pt2 in correspondences], dtype=np.float32).T

    points_4d = cv.triangulatePoints(Pl, Pr, ptsL, ptsR) 
    points_3d = (points_4d[:3] / points_4d[3]).T       
    return points_3d

def get_projection_matrices(F, K, ptsl, ptsr):
    E = K.T @ F @ K

    ptsl_norm = cv.undistortPoints(ptsl.reshape(-1, 1, 2), K, None)
    ptsr_norm = cv.undistortPoints(ptsr.reshape(-1, 1, 2), K, None)

    _, R, t, _ = cv.recoverPose(E, ptsl_norm, ptsr_norm)

    Pl = K @ np.hstack((np.eye(3), np.zeros((3, 1))))   
    Pr = K @ np.hstack((R, t))

    return [Pl, Pr]     

def get_extrinsics(F, K, ptsl, ptsr):
    E = K.T @ F @ K

    ptsl_norm = cv.undistortPoints(ptsl.reshape(-1, 1, 2), K, None)
    ptsr_norm = cv.undistortPoints(ptsr.reshape(-1, 1, 2), K, None)

    _, R, t, _ = cv.recoverPose(E, ptsl_norm, ptsr_norm)

    return [R, t]

def draw_matches(img1, img2, pts1, pts2, max_display=50):
    # Convert to color if needed
    if len(img1.shape) == 2:
        img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

    # Stack images side-by-side
    h = max(img1.shape[0], img2.shape[0])
    w = img1.shape[1] + img2.shape[1]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:img1.shape[0], :img1.shape[1]] = img1
    vis[:img2.shape[0], img1.shape[1]:] = img2

    offset = img1.shape[1]

    # Shuffle and select a subset for display
    indices = np.random.choice(len(pts1), min(max_display, len(pts1)), replace=False)

    for i in indices:
        pt1 = tuple(np.round(pts1[i]).astype(int))
        pt2 = tuple(np.round(pts2[i]).astype(int))
        pt2_offset = (pt2[0] + offset, pt2[1])

        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv.circle(vis, pt1, 4, color, -1)
        cv.circle(vis, pt2_offset, 4, color, -1)
        cv.line(vis, pt1, pt2_offset, color, 1)

    return vis

def compute_fundamental(ptsl, ptsr):
    F, _ = cv.findFundamentalMat(ptsl, ptsr, cv.FM_RANSAC)
    return F

def show_pointcloud(points3D, points2D, image):
    colors = []
    h, w = image.shape[:2]
    for x, y in points2D:
        x, y = int(round(x)), int(round(y))
        if 0 <= x < w and 0 <= y < h:
            colors.append(image[y, x][::-1] / 255.0)  # BGR to RGB
        else:
            colors.append([1.0, 1.0, 1.0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    o3d.visualization.draw_geometries([pcd])
    
def remove_outliers(points3D, points2D, threshold=2.0):
    centroid = np.mean(points3D, axis=0)
    distances = np.linalg.norm(points3D - centroid, axis=1)
    std_dev = np.std(distances)
    mask = distances < threshold * std_dev
    return [points3D[mask], points2D[mask]]

if __name__ == "__main__":
    iml = cv.imread("images/im_0040.JPG", cv.IMREAD_COLOR)
    iml = cv.resize(iml, (0,0), fx=0.5, fy=0.5)
    imr = cv.imread("images/im_0041.JPG", cv.IMREAD_COLOR)
    imr = cv.resize(imr, (0,0), fx=0.5, fy=0.5)

    IMG_WIDTH = 1704
    IMG_HEIGHT = 2272
    FOCAL_LEN = 34 #in mm
    SENSOR_WIDTH = 13
    SENSOR_HEIGHT = 17.3

    #Intrinsic matrix K
    K = np.matrix([
            [FOCAL_LEN*(IMG_WIDTH/SENSOR_WIDTH), 0, IMG_WIDTH/2],
            [0, FOCAL_LEN*(IMG_HEIGHT/SENSOR_HEIGHT), IMG_HEIGHT/2],
            [0, 0, 1]
            ], dtype=np.float64)

    pl, pr = get_matched_points(iml, imr)
    F = compute_fundamental(pl, pr)
    pl, pr = filter_matches_with_fundamental(F, pl, pr)

    Prl, Prr = get_projection_matrices(F, K, pl, pr)
    imgL, imgR, _, _ = rectify_uncalibrated(iml, imr, pl, pr, F)
    disp = compute_disparity_map(imgL, imgR)
    dense = get_dense_corr_from_disp(disp)
    point3d = triangulate_dense_points(Prl, Prr, dense)
    R, t = get_extrinsics(F, K, pl, pr)
    print(R)
    print(t)
    show_pointcloud(point3d, np.array([pt1 for pt1, _ in dense], dtype=np.float32), iml)

    cv.imshow("Matches", draw_matches(iml, imr, pl, pr))
    cv.waitKey(0)
    cv.destroyAllWindows()
