from pathlib import Path

from app.assigment1.pipeline.pipeline import Pipeline

PATH = "/Users/gstrauss/Reichman_University/computer-vision"
INPUT_VIDEO = Path(PATH) / "input.mp4"
INPUT_VIDEO_CROSSWALKS = Path(PATH) / "input_crosswalks.mp4"
OUTPUT_VIDEO = Path(PATH) / "output.mp4"
OUTPUT_VIDEO_CROSSWALKS = Path(PATH) / "output_crosswalks.mp4"


def main() -> None:
    pipeline = Pipeline(INPUT_VIDEO_CROSSWALKS)
    pipeline.process(OUTPUT_VIDEO_CROSSWALKS)


if __name__ == "__main__":
    main()
