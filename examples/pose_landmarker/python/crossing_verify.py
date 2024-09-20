import cv2
import numpy as np
from pose_3d import PoseEstimator
from sigma_crossing import Keypoints, FeatureExtractor, COCO_KEYPOINT_NAMES

KEYPOINT_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
    'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index',
    'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
]

MEDIAPIPE_TO_COCO = {name: KEYPOINT_NAMES.index(name) for name in COCO_KEYPOINT_NAMES}

WORKOUT = {
    'lateral-burpee-over-dumbbell': {'reference': 'oyn3r70PzQ0_21', 'feature': 'shoulder_to_ankle_height'},
    'dumbbell-snatch-left': {'reference': 'oyn3r70PzQ0_21', 'feature': 'left_wrist_to_left_ankle_height'},
    'thruster': {'reference': 'MqJFbmvcKJE_21', 'feature': 'left_knee_angle'}
}


def mp2coco_keypoints(results):
    coco_keypoints = []
    for name in COCO_KEYPOINT_NAMES:
        idx = MEDIAPIPE_TO_COCO[name]
        landmark = results.landmark[idx]
        coco_keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.array(coco_keypoints)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # data = np.load('/home/user/Study/crossfit/mediapipe-samples/examples/pose_landmarker/python/Deu-_GZLE9w_09.npy')
    # video_path = "/home/user/gdrive/crossfit/data/rounds/241_001_dumbbell-snatch-left_009.mp4"
    video_path = "/home/user/gdrive/crossfit/data/rounds/241_002_dumbbell-snatch-left_009.mp4"
    # video_path = "/home/user/gdrive/crossfit/data/reps/241_003_dumbbell-snatch-left_009_001.mp4"
    workout = 'dumbbell-snatch-left'

    ## reference
    feature_name = WORKOUT[workout]['feature']
    reference_name = WORKOUT[workout]['reference']
    reference_sequence = np.load(f'/home/user/gdrive/crossfit/data/sequence/{workout}/{reference_name}.npy')

    reference_feature = FeatureExtractor()
    for keypoints in reference_sequence:
        keypoints = Keypoints(keypoints)
        reference_feature.update(keypoints)
    mean, std = reference_feature.stat(feature_name)



    ## call atheletes
    mp = PoseEstimator()
    cap = cv2.VideoCapture(video_path)

    inference_feature = FeatureExtractor()

    ##############################
    # 플롯 초기화
    plt.ion()  # 대화형 모드 켜기
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    img_plot = ax1.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    line, = ax2.plot([], [], 'r.')
    ax2.set_xlim(0, 500)
    ax2.set_ylim(-5, 5)
    ax2.set_title('Z-Score')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Z-Score')
    ax2.grid(True)
    frame_count = 0
    text_counts = ax1.text(10, 30, '', color='white', fontsize=12, fontweight='bold')
    
    ################################
    counts = 0
    cross_flag = 0
    direction = 0
    delta_zscore, pre_zscore = 0, 0
    zsdelta_list, zscore_list = [], []  ## remove in the future
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret: break

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = mp.pose.process(frame_bgr)
        if results.pose_landmarks is None:
            continue

        coco_keypoints = mp2coco_keypoints(results.pose_landmarks)
        keypoints = Keypoints(coco_keypoints)
        inference_feature.update(keypoints, smoothen=True)

        feat = inference_feature.get(feature_name)[-1]
        zscore = (feat - mean) / std

        if pre_zscore == 0: pre_zscore = zscore
        delta_zscore = zscore - pre_zscore
        pre_zscore = zscore

        # if abs(zscore) < 1:
        #     continue

        ####################################
        zsdelta_list.append(delta_zscore)
        zscore_list.append(zscore)
        frame_count += 1

        img_plot.set_array(frame)
        line.set_data(range(len(zscore_list)), zscore_list)
        ax2.set_xlim(max(0, frame_count - 100), frame_count)

        text_counts.set_text(f'Counts: {counts}')

        plt.draw()
        plt.pause(0.05)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ####################################

        ## counting
        if direction == 0:
            direction = 1 if zscore < 0 else -1
        elif (zscore * direction) > 0:
            cross_flag += 1
            direction *= -1
            if cross_flag % 2 != 0:
                counts += 1
                # counts.append(i)

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()  # 대화형 모드 끄기
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(zsdelta_list)
    plt.title('Z-Score Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Z-Score')
    plt.grid(True)
    plt.show()




    ## counting
    # inference_feature = FeatureExtractor()



# 0 - nose
# 1 - left eye (inner)
# 2 - left eye
# 3 - left eye (outer)
# 4 - right eye (inner)
# 5 - right eye
# 6 - right eye (outer)
# 7 - left ear
# 8 - right ear
# 9 - mouth (left)
# 10 - mouth (right)
# 11 - left shoulder
# 12 - right shoulder
# 13 - left elbow
# 14 - right elbow
# 15 - left wrist
# 16 - right wrist
# 17 - left pinky
# 18 - right pinky
# 19 - left index
# 20 - right index
# 21 - left thumb
# 22 - right thumb
# 23 - left hip
# 24 - right hip
# 25 - left knee
# 26 - right knee
# 27 - left ankle
# 28 - right ankle
# 29 - left heel
# 30 - right heel
# 31 - left foot index
# 32 - right foot index
