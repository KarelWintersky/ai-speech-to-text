## Пакетная расшифровка речи в текст

**Вход**: аудиофайл(ы)

**Выход**: текст + текст с таймкодами

### Зависимости

- [Python](https://python.org) 3.10+
- Библиотека [Whisper](https://github.com/openai/whisper)
- Скомпилированный [FFMPEG](https://ffmpeg.org/download.html) + путь к нему в переменной окружения path
- Желательно GPU, но на CPU тоже работает (медленно)

### Подготовка

1. Установите зависимости:
```
pip install openai-whisper
```

2. Задайте параметры распознавания в файле `settings.ini`:
   - sources_dir - папка с аудиофайлами для распознавания в текст
   - whisper_model - размер модели, от которого зависит скоость и качество распознавания. Подробности о моделях есть в [readme](https://github.com/openai/whisper#available-models-and-languages) Whisper
   - force_transcribe_language - код языка для распознавания. Поддерживаются все популярные языки, включая русский

### Использование (windows)

1. Запустите batch-speech-to-text.bat. Если все зависимости установлены, загрузится модель и начнется распознавание найденных в указанной папке аудиофайлов.
2. По ходу работы программы отображаются фрагменты распознанного текста.
3. По завершении работы программы рядом с исходным файлом появятся два текстовых файла:
   - filename.txt - распознанный текст, по одному предложению в строке.
   - filename_timecode.txt - распознанный текст с таймкодами, по ~3-5 секунд в строке.

### Использование (другие OC)

```
python3 batch-speech-to-text.py
```

### Благодарности

https://github.com/dimonier/batch-speech-to-text.git 
