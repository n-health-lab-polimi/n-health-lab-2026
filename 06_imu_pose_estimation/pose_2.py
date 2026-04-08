import cv2
from ultralytics import YOLO

# COCO skeleton connections
SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

model = YOLO("yolo26n-pose.pt")

results = model.track(
    source="video.mp4",
    stream=True, # Low memnory usage
    persist=True, # Keep object identities between frames
    show=False
)

for result in results:
    frame = result.orig_img.copy()

    if result.keypoints is not None:
        kpts = result.keypoints.xy
        confs = result.keypoints.conf

        for p_idx, person in enumerate(kpts):

            # Draw skeleton lines
            for i, j in SKELETON:
                if confs[p_idx][i] > 0.3 and confs[p_idx][j] > 0.3:
                    pt1 = tuple(person[i].int().tolist())
                    pt2 = tuple(person[j].int().tolist())

                    cv2.line(
                        frame,
                        pt1,
                        pt2,
                        color=(255, 0, 0),
                        thickness=2
                    )

            # Draw joints as circles
            for kp_idx, (x, y) in enumerate(person):
                if confs[p_idx][kp_idx] < 0.3:
                    continue

                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    radius=5,
                    color=(0, 255, 0),
                    thickness=-1
                )

    cv2.imshow("Pose with custom skeleton", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
