# Программная инженерия

Серия заданий 1. Добавлена работа с текстом и аудио. Используются готовые модели, дополнительное обучение не проводится.

Установка на Windows с Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Запуск из папки проекта:

```powershell
.\.venv\Scripts\python.exe text_task.py "Мне понравился этот фильм"
.\.venv\Scripts\python.exe evaluate_text.py
.\.venv\Scripts\python.exe download_samples.py
.\.venv\Scripts\python.exe audio_task.py samples/jfk.flac --language en
.\.venv\Scripts\python.exe evaluate_audio.py
```

При первом запуске нужен интернет. Результаты сохраняются в `results`. Аудиозапись ограничена 30 секундами, язык по умолчанию русский.

На 15 текстах accuracy 0.9333, macro-F1 0.9327. На английской записи WER равен 0. Это небольшие демонстрационные проверки.
