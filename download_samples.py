from pathlib import Path
from urllib.request import urlretrieve

SAMPLES = {
    "dog.jpg": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "jfk.flac": "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
    "vtest.avi": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi",
}


def main():
    Path("samples").mkdir(exist_ok=True)
    for name, url in SAMPLES.items():
        path = Path("samples") / name
        if not path.exists():
            urlretrieve(url, path)
        print(path)


if __name__ == "__main__":
    main()
