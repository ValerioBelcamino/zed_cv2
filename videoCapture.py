import pyzed.sl as sl
import cv2
import numpy as np
import quaternionic

def get_transform_matrix(r_mat, t_vec):
     tf_mat = np.concatenate((r_mat, t_vec.reshape(3,1)), axis=1)
     tf_mat = np.concatenate((tf_mat, np.array([[0, 0, 0, 1]])), axis=0)
     return tf_mat

def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''

    

    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    i = 0
    for i in range(len(corners)):
        nada, R, t = cv2.solvePnP(marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return np.asarray(rvecs), np.asarray(tvecs), trash

def aruco_display(corners, ids, rejected, image):
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()
		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			# compute and draw the center (x, y)-coordinates of the ArUco
			# marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			# draw the ArUco marker ID on the image
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))
			# show the output image
	return image

def init_zed(resolution):
    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()

    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = resolution
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD
    image_size = zed.get_camera_information().camera_resolution
    print(image_size.width, image_size.height)
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    return zed, image_size, image_zed

def grab_zed_frame(zed, image_size, image_zed):
    if zed.grab() == sl.ERROR_CODE.SUCCESS :
        # Retrieve the left image in sl.Mat
        zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        # Use get_data() to get the numpy array
        image_ocv = image_zed.get_data()
        return image_ocv

def main():

    calibration_matrix_path = "zed2_calibration_matrix.npy"
    calibration_matrix = np.load(calibration_matrix_path)    
    distortion_path = "zed2_distortion.npy"
    distortion = np.load(distortion_path)
    print(distortion)
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters()
    arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    zed, image_size, image_zed = init_zed(sl.RESOLUTION.HD1080)
    # print(zed.get_camera_information())
    while True:
        image_ocv = grab_zed_frame(zed, image_size, image_zed)
        # print(image_ocv.dtype)
        # Display the left image from the numpy array
        image_ocv_grey = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = arucoDetector.detectMarkers(image_ocv_grey)

        rvec, tvec, _ = estimatePoseSingleMarkers(corners, 0.058, calibration_matrix, distortion)
        rod = [cv2.Rodrigues(r)[0] for r in rvec]
        # print(np.where(ids == 100))
        ref_row = np.where(ids == 100)
        pos_row = np.where(ids == 42)


        if len(ref_row[0]) != 0 and len(pos_row[0]) != 0:
            ref_row = ref_row[0][0]
            pos_row = pos_row[0][0]
            # ref_col = np.where(ids == 100)[1][0]
            # print(ref_row)

            ref_rot = rod[ref_row]
            ref_trasl = tvec[ref_row]

            # print(ref_rot)
            # print(ref_trasl)

            pos_rot = rod[pos_row]
            pos_trasl = tvec[pos_row]

            # print(pos_rot)
            # print(pos_trasl)

            ref_tf = get_transform_matrix(ref_rot, ref_trasl)
            pos_tf = get_transform_matrix(pos_rot, pos_trasl)

            # np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
            ref_to_pos_tf = np.dot(np.linalg.inv(ref_tf), pos_tf)

            print(ref_to_pos_tf)

        # for c in rod:
        #     print(c)
        # # print(rvec)
        # print(tvec)
        # print('--------------')
        quats = []
        # for i in range(len(rvec)):
        #     # q = quaternionic.array.from_euler_angles(rvec[i][0], rvec[i][1], rvec[i][2])
        #     # quats.append(q)
        #     # print(f'w: {quats[i].w}, x: {quats[i].x}, y: {quats[i].y}, z: {quats[i].z}')
        #     print(f'w: {rvec[i][0]}, x: {rvec[i][1]}, y: {rvec[i][2]}, z: {rvec[i][3]}')
        # print(quats)
        if ids is not None: print(len(ids))
        image_ocv = aruco_display(corners, ids, rejected, image_ocv)
        # print(rvec.shape)
        # print(tvec.shape)

        if tvec.shape[0] > 0 and rvec.shape[0] > 0:
            for j in range(tvec.shape[0]):
                cv2.drawFrameAxes(image_ocv, calibration_matrix, distortion, rvec[j], tvec[j], 0.05) 
                
        image_resize = cv2.resize(image_ocv, (1280, 720))
        cv2.imshow("Image", image_resize)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()