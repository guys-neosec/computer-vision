from pathlib import Path

from app.assigment1.pipeline.pipeline import Pipeline

PATH = "/home/aweinsto/projects/computer-vision/"
INPUT_VIDEO = Path(PATH) / "input.mp4"
OUTPUT_VIDEO = Path(PATH) / "output.mp4"


def main() -> None:
    pipeline = Pipeline(INPUT_VIDEO)
    pipeline.process(OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
