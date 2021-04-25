import numpy as np
import os
import sys
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
import cv2
import time
import dlib
from imutils import face_utils


def makeVector(vecbase, basenumber, vecsmile, smilenumber):
    vec_smile = np.concatenate([vecbase[basenumber][[0, 1, 2, 3]], vecsmile[smilenumber][[
        4, 5]], vecbase[basenumber][[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]])
    vec_smile = vec_smile.reshape(1, 18, 512)
    return vec_smile


def morphing(vec_syn, idx, speed):
    for j in range(len(idx)-1):
        for i in range(speed):
            vec = vec_syn[idx[j]] + \
                (vec_syn[idx[j+1]]-vec_syn[idx[j]])*i/(speed - 1)
            vec = vec.reshape(1, 18, 512)
            images = Gs.components.synthesis.run(
                vec, **Gs_syn_kwargs)
            img_cv = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
            cv2.imshow('images', img_cv)

            for i in range(5):
                img = cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.03)


def receivegaze(frame, detector, predictor):
    face_rects = detector(frame, 0)

    if len(face_rects) > 0:
        shape = predictor(frame, face_rects[0])
        shape = face_utils.shape_to_np(shape)

        reprojectdst, euler_angle = get_head_pose(shape)
    else:
        euler_angle = [[90],
                       [90],
                       [90]]
    return euler_angle


class State():
    def selectface(self):
        raise NotImplementedError("selectface is abstractmethod")

    def editting(self):
        raise NotImplementedError("editting is abstractmethod")

    def vecDecide(self):
        raise NotImplementedError("vecDecide is abstractmethod")

    def showPortraits(self):
        raise NotImplementedError("showPortraits is abstractmethod")

    def timemanagement(self):
        raise NotImplementedError("timemanagement is abstractmethod")


class Laughing(State):
    def selectface(self, k, idxs):
        idx = idxs[k]
        return k, idx

    def editting(self, i, speed, vec_syn):
        if i < speed:
            i += 1
        else:
            pass
        return i

    def vecDecide(self, vec_syn, idx, speed, i):
        j = 0
        vec = vec_syn[idx[j]] + \
            (vec_syn[idx[j+1]]-vec_syn[idx[j]])*(i)/(speed - 1)
        vec = vec.reshape(1, 18, 512)
        return vec

    def showPortraits(self, vec):
        images = Gs.components.synthesis.run(
            vec, **Gs_syn_kwargs)
        img_cv = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
        cv2.imshow('images', img_cv)
        time.sleep(0.06)

    def timemanagement(self, t0):
        return 0, time.time()


class ToNeutral(State):
    def selectface(self, k, idxs):
        idx = idxs[k]
        return k, idx

    def editting(self, i, speed, vec_syn):
        if i > 0:
            i -= 1
        else:
            pass
        return i

    def vecDecide(self, vec_syn, idx, speed, i):
        j = 0
        vec = vec_syn[idx[j]] + \
            (vec_syn[idx[j+1]]-vec_syn[idx[j]])*(i)/(speed - 1)
        vec = vec.reshape(1, 18, 512)
        return vec

    def showPortraits(self, vec):
        images = Gs.components.synthesis.run(
            vec, **Gs_syn_kwargs)
        img_cv = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
        cv2.imshow('images', img_cv)
        time.sleep(0.06)

    def timemanagement(self, t0):
        return 0, time.time()


class Neutral(State):
    def selectface(self, k, idxs):
        idx = idxs[k]
        return k, idx

    def editting(self, i, speed, vec_syn):
        return 0

    def vecDecide(self, vec_syn, idx, speed, i):
        j = 0
        vec = vec_syn[idx[j]] + \
            (vec_syn[idx[j+1]]-vec_syn[idx[j]])*(i)/(speed - 1)
        vec = vec.reshape(1, 18, 512)
        return vec

    def showPortraits(self, vec):
        images = Gs.components.synthesis.run(
            vec, **Gs_syn_kwargs)
        img_cv = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
        cv2.imshow('images', img_cv)
        time.sleep(0.06)

    def timemanagement(self, t0):
        if state.getlaststate() != state.getstate():
            t0 = time.time()
        t1 = time.time()
        t = t1 - t0
        return t, t0


class Morphing(State):
    def selectface(self, k, idxs):
        k += 1
        if k == 5:
            k = 0
        if k == 0:
            idx = [idxs[4][0], idxs[k][0]]
        else:
            idx = [idxs[k - 1][0], idxs[k][0]]
        return k, idx

    def editting(self, i, speed, vec_syn):
        morphing(vec_syn, idx, speed)
        return 0

    def vecDecide(self, vec_syn, idx, speed, i):
        j = 0
        vec = vec_syn[idx[j]] + \
            (vec_syn[idx[j+1]]-vec_syn[idx[j]])*(i)/(speed - 1)
        vec = vec.reshape(1, 18, 512)
        return vec

    def showPortraits(self, vec):
        pass

    def timemanagement(self, t0):
        return 0, time.time()


class Context:
    def __init__(self):
        self.laughing = Laughing()
        self.toneutral = ToNeutral()
        self.neutral = Neutral()
        self.morphing = Morphing()
        self.state = self.neutral
        self.laststate = self.toneutral

    def change_state(self, event):
        self.laststate = self.state
        if event == "gazing":
            self.state = self.laughing
        elif event == "no gaze":
            self.state = self.toneutral
        elif event == "the time is about to be neutral":
            self.state = self.neutral
        elif event == "time passed":
            self.state = self.morphing
        else:
            raise ValueError("change_state method must be in {}".format(
                ["gazing", "no gaze", "the time is about to be neutral", "time passed"]))

    def selectface(self, k, idxs):
        return self.state.selectface(k, idxs)

    def timemanagement(self, t0):
        return self.state.timemanagement(t0)

    def editting(self, i, speed, vec_syn):
        return self.state.editting(i, speed, vec_syn)

    def vecDecide(self, vec_syn, idx, speed, i):
        return self.state.vecDecide(vec_syn, idx, speed, i)

    def showPortraits(self, vec):
        return self.state.showPortraits(vec)

    def getstate(self):
        return self.state

    def getlaststate(self):
        return self.laststate


# dlib init
face_landmark_path = 'stylegan2encoder/shape_predictor_68_face_landmarks.dat'

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002,
     0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(
        object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


cap = cv2.VideoCapture(0)

if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)
    euler_angle = [[90],
                   [90],
                   [90]]

    # stylegan2 init
    vec_smile = np.load('stylegan2encoder/samples/smile.npy')
    vec_syn = np.load('stylegan2encoder/samples/base.npy')
    vec_smile_special = makeVector(vec_syn, 60, vec_smile, 5)
    vec_smile_special2 = makeVector(vec_syn, 50, vec_smile, 2)
    vec_smile_special3 = makeVector(vec_syn, 20, vec_smile, 0)
    vec_smile_special4 = makeVector(vec_syn, 1, vec_smile, 2)
    vec_smile_special6 = makeVector(vec_syn, 54, vec_syn, 6)
    vec_smile_0 = vec_syn
    vec_syn = np.concatenate(
        [vec_smile_0, vec_smile_special, vec_smile_special2, vec_smile_special3, vec_smile_special4, vec_smile_special6])
    idxs = [[60, 124], [50, 125], [20, 126], [1, 127], [128, 54]]
    idx = idxs[0]

    network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'

    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items(
    ) if name.startswith('noise')]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(
        func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = True
    Gs_syn_kwargs.truncation_psi = 0.5
    i = 0
    k = 0
    t = 0
    laughingspeed = 20
    morphingspeed = 40
    t0 = time.time()
    state = Context()

    while cap.isOpened():
        ret, frame = cap.read()
        euler_angle = receivegaze(frame, detector, predictor)
        t, t0 = state.timemanagement(t0)

        if np.abs(euler_angle[1][0]) < 10:
            state.change_state("gazing")
        else:
            if i > 0:
                state.change_state("no gaze")
            else:
                if t > 100:
                    state.change_state("time passed")
                else:
                    state.change_state("the time is about to be neutral")

        k, idx = state.selectface(k, idxs)
        i = state.editting(i, laughingspeed, vec_syn)
        vec = state.vecDecide(vec_syn, idx, laughingspeed, i)
        state.showPortraits(vec)
        for z in range(5):
            img = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
