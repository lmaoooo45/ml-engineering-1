import argparse
import math
from pathlib import Path
from time import perf_counter

import cv2
import torch
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms.functional import to_tensor

from common import save_json


def detect_video(path, output="results/video.avi", threshold=0.5, max_frames=150):
    if not 0 <= threshold <= 1:
        raise ValueError("Порог должен быть от 0 до 1")
    if max_frames < 1:
        raise ValueError("Число кадров должно быть положительным")
    if Path(path).resolve() == Path(output).resolve():
        raise ValueError("Входной и выходной видеофайлы должны отличаться")
    if Path(output).suffix.lower() != ".avi":
        raise ValueError("Выходное видео должно иметь расширение .avi")
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
    categories = weights.meta["categories"]
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Не удалось открыть видео")
    writer = None
    frames = []
    inference_seconds = 0.0
    started = perf_counter()
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("В видео нет корректной частоты кадров")
        for frame_index in range(max_frames):
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
                if not writer.isOpened():
                    raise ValueError("Не удалось создать AVI с кодеком MJPG")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = to_tensor(rgb)
            inference_started = perf_counter()
            with torch.inference_mode():
                prediction = model([tensor])[0]
            inference_seconds += perf_counter() - inference_started
            detections = []
            for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
                if float(score) < threshold:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                name = categories[int(label)]
                detections.append({"label": name, "score": float(score), "box": [x1, y1, x2, y2]})
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {float(score):.2f}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frames.append({"frame": frame_index, "seconds": frame_index / fps, "detections": detections})
            writer.write(frame)
        if not frames:
            raise ValueError("Видео не содержит читаемых кадров")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    elapsed = perf_counter() - started
    return {"file": str(path), "output": str(output), "threshold": threshold,
            "processed_frames": len(frames), "source_fps": fps,
            "processing_seconds": elapsed, "processing_fps": len(frames) / elapsed,
            "inference_seconds": inference_seconds, "frames": frames}


def main():
    parser = argparse.ArgumentParser(description="Детекция объектов по кадрам видео")
    parser.add_argument("file")
    parser.add_argument("--output", default="results/video.avi")
    parser.add_argument("--json", default="results/video.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=150)
    args = parser.parse_args()
    result = detect_video(args.file, args.output, args.threshold, args.max_frames)
    save_json(result, args.json)
    print({k: v for k, v in result.items() if k != "frames"})


if __name__ == "__main__":
    main()
