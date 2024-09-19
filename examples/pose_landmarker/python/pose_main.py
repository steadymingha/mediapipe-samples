import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def _draw_landmarks(self, image, landmarks):
        if landmarks is not None:
            annotated_image = image.copy()
            self.mp_drawing.draw_landmarks(
                annotated_image,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS)
            return annotated_image
        else:
            return image
    def process_video(self, video_path):
        fig = plt.figure()
        ax = fig.add_subplot(111 , projection = '3d')
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            annotated_frame = self._draw_landmarks(frame, results.pose_landmarks)

            # cv2.imshow('Pose Estimation', annotated_frame)

            ax.clear()

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                x = [landmark.x for landmark in landmarks]
                y = [landmark.y for landmark in landmarks]
                z = [landmark.z for landmark in landmarks]

                ax.scatter(x, y, z)

                # Draw connections
                for connection in self.mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    ax.plot(
                        [x[start_idx], x[end_idx]],
                        [y[start_idx], y[end_idx]],
                        [z[start_idx], z[end_idx]],
                        'r-'
                    )

            ax.set_xlim(0, 1)
            ax.set_ylim(0.1, 1)
            ax.set_zlim(0, 1)
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            ax.view_init(elev=20, azim=-60)  # Set the initial view
            
            # Show plot
            plt.pause(0.1)
            plt.show(block=False)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        cap.release()
        # cv2.destroyAllWindows()
    
    







    def process_image(self, image_path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        input_image = cv2.imread(image_path)
        input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(input_image_rgb)
        annotated_image = self._draw_landmarks(input_image, results.pose_landmarks)
        cv2.imshow('Pose Estimation - Image', annotated_image)
        ax.clear()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            x = [landmark.x for landmark in landmarks]
            y = [landmark.y for landmark in landmarks]
            z = [landmark.z for landmark in landmarks]

            # Plot 3D landmarks
            ax.scatter(x, y, z)

            # Draw connections
            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                ax.plot(
                    [x[start_idx], x[end_idx]],
                    [y[start_idx], y[end_idx]],
                    [z[start_idx], z[end_idx]],
                    'r-'
                )

        # Set plot limits and labels
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # Set view
        ax.view_init(elev=-90, azim=-90)

        plt.show()

if __name__ == '__main__':
    import os
    scirpt_dir = (os.path.dirname(os.path.abspath(__file__)))
    video_path = "241_001_dumbbell-snatch-left_009.mp4"
    pose = PoseEstimator()
    pose.process_video(os.path.join(scirpt_dir, video_path))
    # pose.process_image('image.jpg')
