#%% md
# ## 1. Setup
#%%
import cv2
import math
import numpy as np
from rtmlib import Body
from tqdm import tqdm

import matplotlib.pyplot as plt
# from pose_tools import Sort
#%% md
# ## 2. Funcs
#%%
POSE_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

metricNames = [
    "shl_dist", "lshl_lpalm_dist", "rshl_rPalm_dist", "lshl_rpalm_dist",
    "rShl_lpalm_dist", "lshl_lHip_dist", "rshl_rhip_dist", "lknee_lhip_dist",
    "rknee_rhip_dist", "lknee_lfeet_dist", "rknee_rfeet_dist", "lhip_lfeet_dist",
    "rhip_rfeet_dist", "lpalm_lhip_dist", "rpalm_rhip_dist", "lpalm_lfeet_dist",
    "rpalm_rfeet_dist", "lrpalm_dist"
]
#%%
class ComputeMetric:
    @staticmethod
    def angle(p1, p2, p3):
        # Angle enclosed at p2
        angle_23 = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]))
        angle_21 = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
        angle_23 = angle_23 + 360 if angle_23 < 0 else angle_23
        angle_21 = angle_21 + 360 if angle_21 < 0 else angle_21
        a = abs(angle_23 - angle_21)
        return min(a, 360 - a)

    @staticmethod
    def distance(p1, p2, kps, as_ratio=None):
        if not as_ratio:
            as_ratio = (1, 1)
        l1x = (kps[p1][0] - kps[p2][0]) * as_ratio[0]
        l1y = (kps[p1][1] - kps[p2][1]) * as_ratio[1]
        l22   = l1x ** 2 + l1y ** 2
        l2 = l22 ** 0.5
        return l2

    @staticmethod
    def normalize_kps(keypoints, shape):
        for kp in keypoints:
            kp[0] /= shape[1]
            kp[1] /= shape[0]
        return keypoints


class SmoothFilter:
    def __init__(self):
        self.kps = None

    def update(self, kps, alpha=0.4):
        kps = np.asarray(kps, dtype=np.float32)
        if self.kps is None:
            self.kps = kps
        else:
            self.kps = alpha * self.kps + (1.0 - alpha) * kps

    def __call__(self, *args, **kwargs):
        return self.kps


class ZeroCrossing:
    def __init__(self, lag, reference):
        self.y = []
        self.lag = lag
        self.reference = reference

    def update(self, new_value):
        self.y.append(new_value)
        self.window = self.y[-self.lag:]

    def checkCross(self):
        rl = self.window[-1]
        ru = self.window[0]
        if rl < self.reference and ru > self.reference:
            return True
        return False
#%%
class Metrics:
    def __init__(self, params=None):
        self.state = {el:0 for el in metricNames}
        self.lenMetrics = len(self.state)
        self.as_ratio = params

    def update(self, kps):
        self.state["shl_dist"] = ComputeMetric.distance(
            POSE_DICT["left_shoulder"],
            POSE_DICT["right_shoulder"],
            kps,
            self.as_ratio
        )

        self.state["lshl_lpalm_dist"] = kps[POSE_DICT["left_shoulder"]][1] - kps[POSE_DICT["left_wrist"]][1]
        self.state["rshl_rPalm_dist"] = kps[POSE_DICT["right_shoulder"]][1] - kps[POSE_DICT["right_wrist"]][1]
        self.state["lshl_rpalm_dist"] = kps[POSE_DICT["left_shoulder"]][1] - kps[POSE_DICT["right_wrist"]][1]
        self.state["rShl_lpalm_dist"] = kps[POSE_DICT["right_shoulder"]][1] - kps[POSE_DICT["left_wrist"]][1]

        self.state["lshl_lHip_dist"] = kps[POSE_DICT["left_hip"]][1] - kps[POSE_DICT["left_shoulder"]][1]
        self.state["rshl_rhip_dist"] = kps[POSE_DICT["right_hip"]][1] - kps[POSE_DICT["right_shoulder"]][1]

        self.state["lknee_lhip_dist"] = kps[POSE_DICT["left_knee"]][1] - kps[POSE_DICT["left_hip"]][1]
        self.state["rknee_rhip_dist"] = kps[POSE_DICT["right_knee"]][1] - kps[POSE_DICT["right_hip"]][1]

        self.state["lknee_lfeet_dist"] = kps[POSE_DICT["left_ankle"]][1] - kps[POSE_DICT["left_knee"]][1]
        self.state["rknee_rfeet_dist"] = kps[POSE_DICT["right_ankle"]][1] - kps[POSE_DICT["right_knee"]][1]

        self.state["lhip_lfeet_dist"] = kps[POSE_DICT["left_ankle"]][1] - kps[POSE_DICT["left_hip"]][1]
        self.state["rhip_rfeet_dist"] = kps[POSE_DICT["right_ankle"]][1] - kps[POSE_DICT["right_hip"]][1]

        self.state["lpalm_lhip_dist"] = kps[POSE_DICT["left_wrist"]][1] - kps[POSE_DICT["left_hip"]][1]
        self.state["rpalm_rhip_dist"] = kps[POSE_DICT["right_wrist"]][1] - kps[POSE_DICT["right_hip"]][1]

        self.state["lpalm_lfeet_dist"] = kps[POSE_DICT["left_ankle"]][1] - kps[POSE_DICT["left_wrist"]][1]
        self.state["rpalm_rfeet_dist"] = kps[POSE_DICT["right_ankle"]][1] - kps[POSE_DICT["right_wrist"]][1]

        self.state["lrpalm_dist"] = kps[POSE_DICT["left_wrist"]][0] - kps[POSE_DICT["right_wrist"]][0]

        self.state["lshl_angle"] = ComputeMetric.angle(
            kps[POSE_DICT["left_elbow"]],
            kps[POSE_DICT["left_shoulder"]],
            kps[POSE_DICT["left_hip"]]
        )

        self.state["rshl_angle"] = ComputeMetric.angle(
            kps[POSE_DICT["right_elbow"]],
            kps[POSE_DICT["right_shoulder"]],
            kps[POSE_DICT["right_hip"]]
        )

    def getMetrics(self):
        return self.state
#%%
def inference(model, image, id=1):
    bboxes = model.det_model(image)
    # bbox_id, bbox = sort.choose_athlete(bboxes)#, id)
    # cv2.putText(image, f'ID: {id}', (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (0, 255, 0), 2)

    # bbox = bbox[None, ...]
    kps, scores = model.pose_model(image, bboxes=bboxes)
    kps, scores = kps[0], scores[0][:, None]
    kps = np.concatenate([kps, scores], axis=1)
    return kps


def normalize_keypoints(kps):
    kps = kps.copy()
    visible = kps[:, 2] > 0.1

    centre_of_gravity = np.mean(kps[visible, :2], axis=0, keepdims=True)
    kps[:, :2] -= centre_of_gravity

    # max_norm = np.max(np.linalg.norm(kps[visible, :2], axis=1))
    # kps[:, :2] /= max_norm
    
    flatten = kps[:, :2].reshape(-1)
    flatten /= np.linalg.norm(flatten)
    kps[:, :2] = flatten.reshape(-1, 2)

    # shoulder_distance = np.linalg.norm(kps[5, :2] - kps[6, :2])
    # kps[:, :2] /= shoulder_distance
    return kps
#%%

if __name__ == '__main__':
    model = Body(
        mode='lightweight',
        backend='onnxruntime',
        device='cpu',
        to_openpose=False,
    )
    #%% md
    # ## 2. Reference
    #%%
    config_dict = {}
    #%%
    # ref_path = '../data/rounds/241_003_dumbbell-snatch-left_009.mp4'
    ref_path = '/home/user/gdrive/crossfit/data/workout/dumbbell-snatch-left/oyn3r70PzQ0_21.mp4'
    # ref_path = '../gmj/GMJ_24.1_burpee_21.mp4'
    cap = cv2.VideoCapture(ref_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # sort = Sort()
    metrics = Metrics()
    track = [[] for _ in range(metrics.lenMetrics)]
    lpftrack = [SmoothFilter() for _ in range(metrics.lenMetrics)]
    #%%
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kps = inference(model, frame)##,0)
        kps = normalize_keypoints(kps)
        kps = kps[:, :2]

        metrics.update(kps)
        for i in range(metrics.lenMetrics):
            x = metrics.state[metricNames[i]] / metrics.state["shl_dist"]
            lpftrack[i].update([x], alpha=0.5)
            track[i].append(lpftrack[i]()[0])

    cap.release()
    #%%
    SDthreshold = 0.4
    std_array = np.std(track, axis=1)
    nonStation = [i for i, s in enumerate(std_array) if s >= SDthreshold]
    motionMetric = list(np.array(metricNames)[nonStation])

    statistics = {}
    overall_signal = np.sum(np.array(track)[nonStation], axis=0)
    # plt.plot(overall_signal)#[:1000])

    statistics["mean"] = np.mean(overall_signal) * 1.0
    # plt.plot([statistics["mean"]]*len(overall_signal[:1000]))
    # plt.show()
    # plt.plot(statistics["mean"])
    #%%

    #%% md
    # ## 4. Inference
    #%%

    inference_path = "/home/user/gdrive/crossfit/data/rounds/241_002_dumbbell-snatch-left_009.mp4"
    #%%
    cap = cv2.VideoCapture(inference_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #%%
    windowSize = 10
    # sort = Sort()
    metrics = Metrics()
    track = [[] for _ in range(len(motionMetric))]
    lpftrack = [SmoothFilter() for _ in range(len(motionMetric))]
    zc = ZeroCrossing(windowSize, statistics['mean'])

    overall_signal = []
    checkzc = []
    prev = reps = 0
    #%%
    for i in tqdm(range(num_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        kps = inference(model, frame)#, 4)
        kps = normalize_keypoints(kps)
        kps = kps[:, :2]

        metrics.update(kps)
        sum_ = 0
        for i in range(len(motionMetric)):
            x = (metrics.state[motionMetric[i]]) / metrics.state["shl_dist"]
            lpftrack[i].update([x], alpha=0.5)
            track[i].append(lpftrack[i]()[0])
            sum_ += lpftrack[i]()[0]

        overall_signal.append(sum_)
        zc.update(sum_)

        current = zc.checkCross()
        checkzc.append(current)

        if prev == 0 and current == 1:
            reps += 1
            print(reps)
        prev = current

        cv2.putText(frame, f'Count: {reps} / Phase: {prev}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    plt.plot(zc.y)
    plt.show()
    #%%
