import cv2

EXIT_KEY = "q"


def play_video(video: cv2.VideoCapture) -> None:
    # video.read returns (bool, frame)
    # the bool indicating whether the frame is valid

    while (read := video.read())[0]:
        (_, frame) = read
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord(EXIT_KEY):
            break
    video.release()
