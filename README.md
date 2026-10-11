# Программная инженерия

Серия заданий 1. Пока в репозитории добавлена работа с текстом.

Программа принимает русскую фразу и определяет её тональность: negative, neutral или positive. Результат выводится в терминал и сохраняется в JSON.

Используется готовая модель [rubert-tiny-sentiment-balanced](https://huggingface.co/cointegrated/rubert-tiny-sentiment-balanced), библиотека Transformers и PyTorch. Обучение модели в программе не проводится.

Запуск на Windows с Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe text_task.py "Мне понравился этот фильм"
.\.venv\Scripts\python.exe evaluate_text.py
```

При первом запуске нужен интернет для скачивания модели. Для другого пути сохранения можно указать `--output results/test.json`.

Проверка на 15 фразах из `samples/texts.csv`: 14 правильных ответов, accuracy 0.9333, macro-F1 0.9327. Подробные результаты находятся в `results/text_evaluation.json`. Это небольшой учебный набор, модель может ошибаться, например на сарказме.
