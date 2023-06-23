import quaternionic
import numpy as np


#baxter to qr
baxter_to_qr_t = np.array([0.506, -0.465, -0.311])
baxter_to_qr_r = np.array([0.0, 1.0, 0.0, 0.0])
baxter_to_qr_r = quaternionic.array(baxter_to_qr_r).normalized

baxter_to_qr_rt = np.concatenate((baxter_to_qr_r.to_rotation_matrix, baxter_to_qr_t.reshape(3,1)), axis=1)
baxter_to_qr_rt = np.concatenate((baxter_to_qr_rt, np.array([[0, 0, 0, 1]])), axis=0)


#camera to qr
camera_to_qr_t = np.array([-0.54013101, 0.06972169, 0.67694433])
camera_to_qr_r = np.array([[ 0.99873941,  0.02822804,  0.04150619],
                            [ 0.04969951, -0.67205766, -0.73882912],
                            [ 0.00703886,  0.73996059, -0.67261339]])

camera_to_qr_rt = np.concatenate((camera_to_qr_r, camera_to_qr_t.reshape(3,1)), axis=1)
camera_to_qr_rt = np.concatenate((camera_to_qr_rt, np.array([[0, 0, 0, 1]])), axis=0)



#camera to pos
camera_to_pos_t = np.array([-0.37569162, 0.01151293, 0.81112785])
camera_to_pos_r = np.array([[-0.99644861,  0.03139953,  0.07812956],
                            [-0.03453131,  0.69387491, -0.71926713],
                            [-0.07679679, -0.71941065, -0.69032642]])

camera_to_pos_rt = np.concatenate((camera_to_pos_r, camera_to_pos_t.reshape(3,1)), axis=1)
camera_to_pos_rt = np.concatenate((camera_to_pos_rt, np.array([[0, 0, 0, 1]])), axis=0)


#shared ref to pos
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
shared_ref_to_pos = np.dot(np.linalg.inv(camera_to_qr_rt), camera_to_pos_rt)

print(shared_ref_to_pos)

#baxter to pos
baxter_to_pos = np.dot( baxter_to_qr_rt, shared_ref_to_pos)
print(baxter_to_pos)

