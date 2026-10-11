import argparse
import json
from pathlib import Path
from metrics import word_error_rate
from common import save_json


def main():
    parser = argparse.ArgumentParser(description="WER для готовой расшифровки")
    parser.add_argument("--reference", default="samples/jfk_reference.txt")
    parser.add_argument("--prediction", default="results/audio.json")
    parser.add_argument("--output", default="results/audio_evaluation.json")
    args = parser.parse_args()
    reference = Path(args.reference).read_text(encoding="utf-8").strip()
    prediction = json.loads(Path(args.prediction).read_text(encoding="utf-8"))
    result = word_error_rate(reference, prediction["text"])
    result.update({"reference": reference, "hypothesis": prediction["text"]})
    save_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()
