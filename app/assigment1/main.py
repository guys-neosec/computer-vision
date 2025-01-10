from pathlib import Path

from app.assigment1.pipeline.pipeline import Pipeline

PATH = "/home/aweinsto/projects/computer-vision/"
GUY_PATH = "/Users/gstrauss/Reichman_University/computer-vision"
INPUT_VIDEO = Path(GUY_PATH) / "input.mp4"
OUTPUT_VIDEO = Path(GUY_PATH) / "output.mp4"
INPUT_NIGHT = Path(GUY_PATH) / "input_night.mp4"
OUTPUT_NIGHT = Path(GUY_PATH) / "output_night.mp4"


def main() -> None:
    pipeline = Pipeline(INPUT_NIGHT)
    pipeline.process(OUTPUT_NIGHT)


if __name__ == "__main__":
    main()
