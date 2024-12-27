import sys
import time
from collections.abc import Callable
from typing import TextIO

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


def progressbar(
    count: int,
    prefix: str = "",
    size: int = 60,
    out: TextIO = sys.stdout,
) -> Callable:
    start = time.time()  # time estimate start
    progress = 0.1

    def advance() -> None:
        nonlocal progress
        x = int(size * progress / count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / progress) * (count - progress)
        mins, sec = divmod(remaining, 60)  # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(
            f"{prefix}[{'█' * x}{('.' * (size - x))}] "
            f"{int(progress)}/{count} Est wait {time_str}",
            end="\r",
            file=out,
            flush=True,
        )
        progress += 1

    return advance
