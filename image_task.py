import argparse
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.utils import img_to_array, load_img

from common import save_json


def classify_image(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    model = MobileNetV2(weights="imagenet")
    started = perf_counter()
    image = load_img(path, target_size=(224, 224), color_mode="rgb")
    array = img_to_array(image)
    batch = tf.expand_dims(array, axis=0)
    batch = preprocess_input(batch)
    predictions = model(batch, training=False).numpy()
    elapsed = perf_counter() - started
    top = decode_predictions(predictions, top=5)[0]
    return {"file": str(path), "top5": [
        {"class_id": class_id, "label": label, "score": float(score)}
        for class_id, label, score in top], "processing_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Классификация изображения")
    parser.add_argument("file")
    parser.add_argument("--output", default="results/image.json")
    args = parser.parse_args()
    result = classify_image(args.file)
    save_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()
