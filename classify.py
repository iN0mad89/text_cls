import hashlib
import json
import os
import csv
import pandas as pd
import chardet
import typer
import regex as re
from tqdm import tqdm

try:
    import ollama
except Exception:  # noqa: E722
    ollama = None

app = typer.Typer(add_completion=False)


def detect_encoding(path: str) -> str:
    with open(path, 'rb') as f:
        raw = f.read()
    result = chardet.detect(raw)
    return result['encoding'] or 'utf-8'


def read_text(path: str) -> str:
    enc = detect_encoding(path)
    with open(path, 'rb') as f:
        data = f.read()
    text = data.decode(enc, errors='ignore')
    return text, enc, hashlib.sha256(data).hexdigest()


def tokenize(text: str) -> list[str]:
    return re.findall(r'\p{L}+', text.lower())


def science_score(tokens: list[str], stems: list[str]) -> float:
    if not tokens:
        return 0.0
    count = sum(any(tok.startswith(s) for s in stems) for tok in tokens)
    return count / len(tokens)


def classify_regex(text: str, patterns: dict[str, str], priority: list[str]):
    for cat in priority:
        pat = patterns.get(cat, '')
        if pat and re.search(pat, text, flags=re.IGNORECASE):
            return cat, pat
    return None, None


def llm_verify(chunk: str, category: str, model: str) -> str:
    if ollama is None:
        return 'ollama_not_available'
    prompt = (
        'Ти експерт з українського права. Оціни, чи документ належить до категорії '
        f'"{category}". Якщо так, напиши "OK" і чому, якщо ні — "NO" і чому.'
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": chunk[:1000]},
    ]
    try:
        response = ollama.chat(model=model, messages=messages)
        return response['message']['content'].strip()[:1024]
    except Exception as e:  # noqa: E722
        return f'error: {e}'


@app.command()
def run(
    data_path: str,
    out: str = 'output',
    threshold: float = 0.01,
    llm: str = typer.Option('none', help='verify|none'),
    model: str = 'mistral:7b-instruct',
    chunk_size: int = 1000,
):
    with open('criteria.json', 'r', encoding='utf-8') as f:
        criteria = json.load(f)
    stems = criteria['science_stems']
    patterns = criteria['categories']
    priority = criteria['priority']

    os.makedirs(out, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    results = []
    unmatched = []

    files = [
        os.path.join(root, fn)
        for root, _, fns in os.walk(data_path)
        for fn in fns if fn.lower().endswith('.txt')
    ]

    for path in tqdm(files, desc='Processing'):
        text, enc, sha = read_text(path)
        tokens = tokenize(text)
        score = science_score(tokens, stems)
        cat = None
        reason_regex = None
        reason_llm = ''
        if score >= threshold:
            cat, reason_regex = classify_regex(text, patterns, priority)
        else:
            cat = None
        if cat and llm == 'verify':
            chunk = ' '.join(tokens[:chunk_size])
            reason_llm = llm_verify(chunk, cat, model)
        if not cat:
            unmatched.append(path)
        results.append({
            'filename': path,
            'category': cat,
            'score': round(score, 4),
            'reason_regex': reason_regex,
            'reason_llm': reason_llm,
            'encoding': enc,
            'sha256': sha,
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out, 'classified_docs.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)

    if not df.empty:
        df['year'] = df['filename'].apply(lambda x: re.search(r'(\d{4})', x or '') and re.search(r'(\d{4})', x).group(1))
        summary = df.groupby('year')['score'].agg(['count', 'mean', 'median']).reset_index()
        summary.to_csv(os.path.join('summary', 'science_impact.csv'), index=False)
    os.makedirs('summary', exist_ok=True)

    with open('logs/unmatched.txt', 'w', encoding='utf-8') as f:
        for path in unmatched:
            f.write(path + '\n')

    typer.echo(f'Processed {len(files)} files')


@app.command()
def inspect(file: str):
    df = pd.read_csv(file)
    typer.echo(df.head())


if __name__ == '__main__':
    app()
