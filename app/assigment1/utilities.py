import cv2
from pydantic import BaseModel


class Metadata(BaseModel):
    frame_count: int
    width: int
    height: int
    fps: float


def extract_metadata(video: cv2.VideoCapture) -> Metadata:
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    return Metadata(frame_count=length, width=width, height=height, fps=fps)
