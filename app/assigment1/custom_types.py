from typing import Annotated, Literal, TypeVar

import numpy as np
from numpy import typing as npt

DType = TypeVar("DType", bound=np.generic)
ThreeChannelArray = Annotated[npt.NDArray[DType], Literal["H", "W", 3]]
RBGFrame = Annotated[npt.NDArray[DType], Literal["H", "W", 3]]
GrayScaleArray = Annotated[npt.NDArray[DType], Literal["H", "W"]]
