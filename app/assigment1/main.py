from pathlib import Path


from app.assigment1.pipeline.pipeline import  Pipeline

PATH = "/Users/gstrauss/Reichman_University/computer-vision"
INPUT_VIDEO = Path(PATH) / "short_input.mp4"
OUTPUT_VIDEO = Path(PATH) / "output.mp4"


def main() -> None:
   pipeline = Pipeline(INPUT_VIDEO)
   pipeline.process(OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
