# Программная инженерия

Серия заданий 1. Четыре программы с готовыми моделями машинного обучения. Используются PyTorch и TensorFlow, дополнительное обучение не проводится.

| Задача | Модель | Результат |
| --- | --- | --- |
| Тональность текста | ruBERT tiny | Класс и оценки в JSON |
| Распознавание речи | Whisper tiny | Расшифровка в JSON |
| Классификация изображения | MobileNetV2 | Пять классов в JSON |
| Объекты на видео | SSDLite320 MobileNetV3 | Видео с рамками и JSON |

Установка на Windows с Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Запуск из папки проекта:

```powershell
.\.venv\Scripts\python.exe download_samples.py
.\.venv\Scripts\python.exe text_task.py "Мне понравился этот фильм"
.\.venv\Scripts\python.exe evaluate_text.py
.\.venv\Scripts\python.exe audio_task.py samples/jfk.flac --language en
.\.venv\Scripts\python.exe evaluate_audio.py
.\.venv\Scripts\python.exe image_task.py samples/dog.jpg
.\.venv\Scripts\python.exe video_task.py samples/vtest.avi --threshold 0.25 --max-frames 30
```

Первый запуск требует интернета для скачивания примеров и весов. Результаты сохраняются в `results`. Для своей аудиозаписи до 30 секунд нужно заменить путь; язык по умолчанию русский. Видео обрабатывается по кадрам, без отслеживания объектов и переноса звука.

На 15 текстах получено 14 правильных ответов: accuracy 0.9333, macro-F1 0.9327. Для английской записи WER равен 0. На фотографии первым определён Samoyed. На видео проверены 30 кадров. Это небольшие демонстрационные проверки.

Постановка задач: [ТЗ](docs/specification.md). Модели, данные, метрики и ограничения: [отчёт](report.md).

Продолжение: [API анализа тональности и тесты](https://github.com/lmaoooo45/ml-engineering-api).
