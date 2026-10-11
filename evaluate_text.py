import csv
from text_task import SentimentAnalyzer
from metrics import classification_metrics
from common import save_json


def main():
    with open("samples/texts.csv", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    analyzer = SentimentAnalyzer()
    predictions = [analyzer.predict(row["text"]) for row in rows]
    report = classification_metrics([row["label"] for row in rows],
                                    [item["label"] for item in predictions],
                                    ["negative", "neutral", "positive"])
    report["predictions"] = predictions
    save_json(report, "results/text_evaluation.json")
    print({k: v for k, v in report.items() if k != "predictions"})


if __name__ == "__main__":
    main()
