import pyzed.sl as sl
import cv2
import numpy as np
import quaternionic
import rospy
from std_msgs.msg import String, Bool



class ArucoDetection():
    def __init__(self):
        rospy.init_node('camera_listener', anonymous=True)
        self.image_sub = rospy.Subscriber("/aruco_detection_activation", Bool, self.loop)
        self.obj_pub = rospy.Publisher("/aruco_detection", String, queue_size=10)
        self.trasl_list = []


    def tf2quat_tr(self, tf):
        quat = quaternionic.array.from_rotation_matrix(tf[:3, :3])
        trasl = tf[:3, 3]
        return trasl, quat

    def get_transform_matrix(self, r_mat, t_vec):
        tf_mat = np.concatenate((r_mat, t_vec.reshape(3,1)), axis=1)
        tf_mat = np.concatenate((tf_mat, np.array([[0, 0, 0, 1]])), axis=0)
        return tf_mat
    
    def moving_average_filter(self, sample, sequence, window_size):
            if len(sequence) > window_size:
                sequence.append(sample)
                sequence=sequence[1:]
            if len(sequence) == window_size:
                yield sum(sequence) / len(sequence)

    def estimatePoseSingleMarkers(self, corners, marker_size, mtx, distortion):
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

    def aruco_display(self, corners, ids, rejected, image):
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



    def loop(self):

        static_rot = np.asarray([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0]
        ])

        baxter2ref = np.asarray([   [ 1.0,  0.0,  0.0,  0.668],
                                    [ 0.0, -1.0, -0.0, -0.245],
                                    [-0.0,  0.0, -1.0, -0.324],
                                    [ 0.0,          0.0,          0.0,          1.0        ]])
        baxter2ref[2,3] += 0.04



        logi2ref = np.load("/home/index1/index_ws/src/zed_cv2/logi2ref.npy")
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        calibration_matrix_path = "/home/index1/index_ws/src/zed_cv2/logitech_calibration_matrix.npy"
        calibration_matrix = np.load(calibration_matrix_path)   
        # calibration_matrix = np.asarray([   [ 1463.4,  0.0,  964.2],
        #                                     [ 0.0, 1464.9, 572.5],
        #                                     [0.0,  0.0, 1.0]])
        print(calibration_matrix)
        distortion_path = "/home/index1/index_ws/src/zed_cv2/logitech_distortion.npy"
        distortion = np.load(distortion_path)
        # distortion = np.asarray([[0.0522,  -0.1169,  -0.003,  0.005, -0.01]])
        print('\n',distortion)
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250) #cv2.aruco.DICT_ARUCO_ORIGINAL
        arucoParams = cv2.aruco.DetectorParameters()
        arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

        cap = cv2.VideoCapture(0)

        #resolution stuff
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(width, height)

        # print(zed.get_camera_information())
        real_ids = []
        iterations = 0

        while iterations<20:
            ret, image_ocv = cap.read()

            # Display the left image from the numpy array
            image_ocv_grey = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = arucoDetector.detectMarkers(image_ocv_grey)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0005)


            if ids is not None:
                corners = cv2.cornerSubPix(image_ocv_grey, corners[0], (3, 3), (-1, -1), criteria)
                for id in ids:
                    if id not in real_ids:
                        real_ids.append(id)

            # rvec, tvec, _ = self.estimatePoseSingleMarkers(corners, 0.251, calibration_matrix, distortion)
            rvec, tvec, _ = self.estimatePoseSingleMarkers(corners, 0.037, calibration_matrix, distortion)

            # self.trasl_list.append(tvec)
            # tvec = self.moving_average_filter(tvec, self.trasl_list, 10)

            rod = [cv2.Rodrigues(r)[0] for r in rvec]
            # # print(np.where(ids == 100))
            ref_row = np.where(ids == 5)
            # pos_row = np.where(ids == 5)


            if len(ref_row[0]) != 0:
                ref_row = ref_row[0][0]


                ref_rot = rod[ref_row]
                ref_rot = np.dot(ref_rot, static_rot)
                rvec[ref_row] = cv2.Rodrigues(ref_rot)[0]
                ref_trasl = tvec[ref_row]


                ref_tf = self.get_transform_matrix(ref_rot, ref_trasl)
                # print(ref_tf)
                # bax2obj = np.dot(baxter2camera, ref_tf)
                # print(bax2obj)
                trasl, r = self.tf2quat_tr(np.dot(baxter2ref, np.dot(np.linalg.inv(logi2ref), ref_tf)))
                print(trasl)
                print(r)
                return trasl, r
                # print(np.dot(baxter2ref, np.linalg.inv(logi2ref)))
                # print(np.dot(np.linalg.inv(logi2ref), ref_tf))
                # np.save('/home/index1/index_ws/src/zed_cv2/logi2ref.npy', ref_tf)
                # pos_tf = get_transform_matrix(pos_rot, pos_trasl)
                # np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
                # ref_to_pos_tf = np.dot(np.linalg.inv(ref_tf), pos_tf)

                # print(ref_to_pos_tf)

            # # for c in rod:
            # #     print(c)
            # # # print(rvec)
            # # print(tvec)
            # # print('--------------')
            # quats = []
            # # for i in range(len(rvec)):
            # #     # q = quaternionic.array.from_euler_angles(rvec[i][0], rvec[i][1], rvec[i][2])
            # #     # quats.append(q)
            # #     # print(f'w: {quats[i].w}, x: {quats[i].x}, y: {quats[i].y}, z: {quats[i].z}')
            # #     print(f'w: {rvec[i][0]}, x: {rvec[i][1]}, y: {rvec[i][2]}, z: {rvec[i][3]}')
            # # print(quats)
            # if ids is not None: print(len(ids))
            image_ocv = self.aruco_display(corners, ids, rejected, image_ocv)
            # # print(rvec.shape)
            # # print(tvec.shape)

            if tvec.shape[0] > 0 and rvec.shape[0] > 0:
                for j in range(tvec.shape[0]):
                    cv2.drawFrameAxes(image_ocv, calibration_matrix, distortion, rvec[j], tvec[j], 0.1) 
                    
            image_resize = cv2.resize(image_ocv, (1280, 720))
            cv2.imshow("Image", image_resize)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.obj_pub.publish('_'.join([str(id) for id in real_ids]))
        cv2.destroyAllWindows()

    def listener(self):
        rospy.loginfo("I am listening to the camera")
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

if __name__ == '__main__':
    HD = ArucoDetection()
    HD.loop()