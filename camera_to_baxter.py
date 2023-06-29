import quaternionic
import numpy as np


# #baxter to qr
# baxter_to_qr_t = np.array([0.506, -0.465, -0.311])
# baxter_to_qr_r = np.array([0.0, 1.0, 0.0, 0.0])
# baxter_to_qr_r = quaternionic.array(baxter_to_qr_r).normalized

# baxter_to_qr_rt = np.concatenate((baxter_to_qr_r.to_rotation_matrix, baxter_to_qr_t.reshape(3,1)), axis=1)
# baxter_to_qr_rt = np.concatenate((baxter_to_qr_rt, np.array([[0, 0, 0, 1]])), axis=0)


# #camera to qr
# camera_to_qr_t = np.array([-0.54013101, 0.06972169, 0.67694433])
# camera_to_qr_r = np.array([[ 0.99873941,  0.02822804,  0.04150619],
#                             [ 0.04969951, -0.67205766, -0.73882912],
#                             [ 0.00703886,  0.73996059, -0.67261339]])

# camera_to_qr_rt = np.concatenate((camera_to_qr_r, camera_to_qr_t.reshape(3,1)), axis=1)
# camera_to_qr_rt = np.concatenate((camera_to_qr_rt, np.array([[0, 0, 0, 1]])), axis=0)



# #camera to pos
# camera_to_pos_t = np.array([-0.37569162, 0.01151293, 0.81112785])
# camera_to_pos_r = np.array([[-0.99644861,  0.03139953,  0.07812956],
#                             [-0.03453131,  0.69387491, -0.71926713],
#                             [-0.07679679, -0.71941065, -0.69032642]])

# camera_to_pos_rt = np.concatenate((camera_to_pos_r, camera_to_pos_t.reshape(3,1)), axis=1)
# camera_to_pos_rt = np.concatenate((camera_to_pos_rt, np.array([[0, 0, 0, 1]])), axis=0)


# #shared ref to pos
# np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
# shared_ref_to_pos = np.dot(np.linalg.inv(camera_to_qr_rt), camera_to_pos_rt)

# print(shared_ref_to_pos)

# #baxter to pos
# baxter_to_pos = np.dot( baxter_to_qr_rt, shared_ref_to_pos)
# print(baxter_to_pos)

# camera2ref = np.asarray([[ 0.98569868,  0.1132282,   0.12480978, -0.12885113],
#                         [ 0.15999065, -0.86139501, -0.48208052, -0.01364867],
#                         [ 0.05292542,  0.49515453, -0.86719138,  1.22745267],
#                         [ 0.0,          0.0,          0.0,          1.0        ]])


camera2ref = np.asarray([[ 0.98350527,  -0.08457559,   -0.15988858, -0.13392051],
                        [ 0.16451668, 0.78565578, 0.59638852, -0.01896198],
                        [ 0.07517748,  -0.61285558, 0.78661069,  1.17743348],
                        [ 0.0,          0.0,          0.0,          1.0        ]])


# baxter2ref = np.asarray([[ 1.0,  0.0,  0.0, 0.568],
#                         [ 0.0, -1.0, 0.0, -0.216],
#                         [0.0,  0.0, -1.0,  -0.291],
#                         [ 0.0,          0.0,          0.0,          1.0       ]])

baxter2ref = np.asarray([[-0.06007087, -0.99615504,  0.06377016,  0.04445781],
                        [ 0.98871109, -0.05059348,  0.14103431, -0.01248386],
                        [-0.13726568,  0.07152232,  0.98794873,  0.49506678],
                        [ 0.0,          0.0,          0.0,          1.0        ]])

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

print(np.dot(baxter2ref, np.linalg.inv(camera2ref)))

bax2cam = np.dot(baxter2ref, np.linalg.inv(camera2ref))

cam2obj = np.asarray(
    [[ 0.9906019,  -0.08974525, -0.10321661,  0.05078958],
    [ 0.13280122,  0.81170506, 0.56876949, -0.14579158],
    [ 0.03273708, -0.57713143, 0.81599485,  1.36610245],
    [ 0.0,          0.0,          0.0,          1.0        ]]
)

bax2obj = np.dot(bax2cam, cam2obj)

print('\n------------------\n')
print(bax2obj)