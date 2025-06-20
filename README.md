# Закон-Science Classifier

Цей проект надає інструмент для офлайн-класифікації українських законів за їхньою науковою спрямованістю.

## Встановлення

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Швидкий старт

```bash
# перший прогін без LLM
python classify.py run data --out output --threshold 0.01

# з перевіркою через LLM
python classify.py run data --out output --llm verify --model mistral:7b-instruct
```

Результати зберігаються в каталозі `output`.
