# 📚 Lecture Notes Pipeline

Автоматическая обработка аудио лекций в Markdown-конспекты с использованием MacWhisper и локальной LLM (LM Studio).

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/temperatura-40/LecturesAnalize.git
cd LecturesAnalize
```

### 2. Создание виртуального окружения

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка проекта

```bash
pip install -e .
```

После установки команда `lectures` будет доступна глобально.

### 4. Настройка конфигурации

```bash
cp config.example.json config.json
```

Открой `config.json` и укажи свои пути:

```json
{
  "folders": {
    "audio": "~/Documents/Lectures",
    "obsidian": "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian/Конспекты (Auto)"
  },
  "macwhisper": {
    "model": "parakeet-pro:nvidia_parakeet-v3",
    "language": "ru"
  },
  "llm": {
    "model": "google/gemma-4-e4b",
    "max_context": 65536,
    "temperature": 0.2
  }
}
```

### 5. Запуск

```bash
lectures
```

---

## 📖 Использование

### Интерактивное меню

Просто запусти `lectures` без аргументов:

```bash
lectures
```

Появится меню с навигацией стрелочками и подтверждением по Enter:

```
📚 Lecture Notes Pipeline
Use ↑↓ to navigate, Enter to select

▶ Process all lectures
  Transcribe only
  Show status
  Exit
```

### Команды без меню

```bash
lectures process      # Обработать все лекции (аудио → транскрипт → конспект)
lectures transcribe    # Только транскрипция (без LLM)
lectures status       # Показать статус всех лекций
```

---

## 📁 Структура проекта

```
LecturesAnalize/
├── src/
│   ├── cli.py            # Интерактивный CLI (curses)
│   ├── pipeline.py       # Основной pipeline обработки
│   ├── scanner.py        # Сканирование аудио и артефактов
│   ├── transcriber.py    # Обёртка над MacWhisper CLI
│   ├── transcript.py     # Работа с транскриптами
│   ├── llm.py            # Клиент для LM Studio API
│   ├── config.py         # Загрузка конфигурации
│   └── util.py           # Атомарные операции с файлами
├── prompts/
│   └── lecture.md        # Шаблон промпта для LLM
├── config.example.json   # Шаблон конфигурации
├── pyproject.toml        # Упаковка для pip
└── README.md             # Этот файл
```

---

## 🔧 Требования

- **Python 3.10+**
- **MacWhisper** (`mw` CLI) — для транскрипции
- **LM Studio** — для локальной LLM (запускается автоматически)

---

## 📝 Как это работает

1. **Сканирование** — поиск аудио файлов в папке `audio`
2. **Транскрипция** — преобразование аудио в текст через MacWhisper
3. **Анализ** — отправка транскрипта в LM Studio для генерации конспекта
4. **Сохранение** — запись `.md` напрямую в папку Obsidian

---

## ⚠️ Примечания

- Markdown-конспекты хранятся **только в Obsidian** (iCloud-синхронизация)
- Аудио и транскрипты — в локальной папке `audio`
- Существующие `.md` в Obsidian не перезаписываются (пропускаются)
- Для работы LLM нужен запущенный LM Studio с загруженной моделью

---

## 📄 Лицензия

MIT
