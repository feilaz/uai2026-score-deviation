"""Generate homogeneous model experiments or evaluate saved numerical snapshots."""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from uai_risk.core import ClaimTuple
from uai_risk.data import load_data
from uai_risk.experiments import run_panel, summarize, write_csv

ROOT = Path(__file__).resolve().parent

# These are five independently prompted calls to the same model. There is no
# debate, shared conversation, model training, or agent-to-agent communication.
DEFAULT_AGENT_PROMPTS = ['You are a skeptical scientist. Answer the question using ONLY information from the provided '
 'context. If the context lacks evidence, say so explicitly.',
 'You are a thorough researcher. Answer the question using ONLY the provided context '
 'document. Explore multiple angles present in the text.',
 'You are a precise assistant focused on accuracy. Answer using ONLY the provided context. '
 'Quote or paraphrase specific passages to support your answer.',
 'You are a critical analyst. Answer the question using ONLY the provided context. Question '
 'any assumptions not directly supported by the text.',
 'You are a balanced reporter. Answer the question using ONLY the provided context document. '
 'Present facts objectively without adding external knowledge.']

DECOMPOSITION_PROMPT = ('Extract atomic facts from the text. Each fact must be:\n'
 '1. Self-contained (understandable without context)\n'
 '2. Verifiable (can be checked as true/false)\n'
 '3. Atomic (contains exactly one claim)\n'
 '\n'
 'Text: {text}\n'
 '\n'
 'Output format (JSON array):\n'
 '["fact 1", "fact 2", ...]')

NLI_PROMPT = ('You are a factual entailment judge. Estimate the probability (0-100) that the CLAIM is '
 'supported by the CONTEXT.\n'
 '\n'
 'CONTEXT:\n'
 '{context}\n'
 '\n'
 'CLAIM: {claim}\n'
 '\n'
 'Guidelines:\n'
 '- 0 = directly contradicted or completely absent\n'
 '- 100 = explicitly and fully stated in context\n'
 '- Use ANY integer from 0 to 100\n'
 '\n'
 'Examples:\n'
 '- Claim restates a context sentence verbatim → 95\n'
 '- Claim is a reasonable paraphrase of context → 78\n'
 '- Claim is partially supported, key detail missing → 42\n'
 '- Claim is vaguely related but not confirmed → 18\n'
 '- Claim contradicts the context → 3\n'
 '\n'
 'Respond with ONLY a single integer (e.g. 72).')

VERIFICATION_PROMPT = ('You are a strict factual verification judge.\n'
 '\n'
 'CONTEXT:\n'
 '{context}\n'
 '\n'
 'CLAIM: {claim}\n'
 '\n'
 'Rules:\n'
 '- Use ONLY information explicitly stated in the context.\n'
 '- Do NOT use external knowledge or reasoning beyond what the context states.\n'
 '- If the context lacks sufficient information to confirm the claim, answer NOT_SUPPORTED.\n'
 '- Paraphrases and logical entailments count as supported.\n'
 '\n'
 'Respond with EXACTLY one word: SUPPORTED or NOT_SUPPORTED')


def read_json(path):
    # Used for both compact .json.gz snapshots and readable per-call records.
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def save_json(path, value):
    # Replace a completed temporary file; this is not a multi-process lock.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def pack(path, value):
    # Canonical key order and zero gzip timestamp make identical exports stable.
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw, mtime=0))
    return {'file': path.name, 'json_sha256': hashlib.sha256(raw).hexdigest()}


def number(name, default, integer=False):
    # Positive numeric settings only. Temperature is validated separately so it
    # may be zero, unlike timeouts, token caps, prices and the spend ceiling.
    value = (int if integer else float)(os.getenv(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return value


def parse_claims(text):
    """Read a complete JSON claim array, including Markdown/prose wrappers."""
    # Like the original bracket-aware parser, respect quotes/escapes inside an
    # array. raw_decode does that without changing or repairing claim content.
    cleaned = text.strip()
    try:
        claims = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find('[')
        if start < 0:
            raise ValueError('Expected a complete JSON array of claims') from None
        try:
            claims, end = json.JSONDecoder().raw_decode(cleaned, start)
        except json.JSONDecodeError:
            raise ValueError('Incomplete or malformed JSON claim array') from None
        # Ambiguous multiple arrays should not silently discard annotations.
        if '[' in cleaned[end:]:
            raise ValueError('Multiple JSON claim arrays are ambiguous')
    if (not isinstance(claims, list) or not claims
            or any(not isinstance(f, str) or not f.strip() for f in claims)):
        raise ValueError('Expected a nonempty JSON list of claim strings')
    return claims


class ModelClient:
    """Cached requests with a cumulative estimated spend ceiling per output folder."""

    def __init__(self, output):
        self.provider = os.getenv('MODEL_PROVIDER', 'gemini')
        if self.provider not in ('gemini', 'openrouter'):
            raise ValueError('MODEL_PROVIDER must be gemini or openrouter')
        key_name = 'OPENROUTER_API_KEY' if self.provider == 'openrouter' else 'GEMINI_API_KEY'
        self.key = os.getenv(key_name, '')
        if not self.key:
            raise ValueError(f'Set {key_name} in .env or the environment')
        self.cache = output / 'calls'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.ledger_path = output / 'spend.json'
        self.spent = read_json(self.ledger_path)['estimated_usd'] if self.ledger_path.exists() else 0
        # A resumed run reads the same ledger; a new output folder starts anew.
        self.budget = number('MAX_COST_USD', 3)
        # Conservative defaults for the configured Flash models, including thinking tokens.
        self.input_price = number('INPUT_USD_PER_MILLION', 1)
        self.output_price = number('OUTPUT_USD_PER_MILLION', 5)
        self.interval = number('MIN_REQUEST_INTERVAL_SECONDS', 13)
        self.timeout = number('REQUEST_TIMEOUT_SECONDS', 120)
        self.last_call = 0

    def charge(self, amount):
        self.spent += amount
        save_json(self.ledger_path, {'estimated_usd': self.spent, 'limit_usd': self.budget})

    def call(self, model, prompt, temperature=0, system=None, json_mode=False, max_tokens=None):
        # Both providers receive equivalent tasks, but their wire formats differ.
        # Secrets go in HTTP headers, never in the cached request identity.
        limit = number('MAX_OUTPUT_TOKENS', 8192, integer=True) if max_tokens is None else max_tokens
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('Output token limit must be a positive integer')
        body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
                'generationConfig': {'temperature': temperature, 'maxOutputTokens': limit}}
        if system:
            body['systemInstruction'] = {'parts': [{'text': system}]}
        if json_mode:
            body['generationConfig']['responseMimeType'] = 'application/json'
        if self.provider == 'openrouter':
            model = model if '/' in model else 'google/' + model
            messages = ([{'role': 'system', 'content': system}] if system else [])
            messages.append({'role': 'user', 'content': prompt})
            body = {'model': model, 'messages': messages, 'temperature': temperature,
                    'max_tokens': limit}
            if json_mode:
                body['response_format'] = {'type': 'json_schema', 'json_schema': {
                    'name': 'claims', 'strict': True,
                    'schema': {'type': 'array', 'items': {'type': 'string'}}}}
        # Gemini retains its original cache identity; OpenRouter gets a separate namespace.
        request = {'model': model, 'body': body}
        if self.provider == 'openrouter':
            request['provider'] = 'openrouter'
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        target = self.cache / f'{digest}.json'
        if target.exists():
            # Even temperature-0.7 calls replay the saved answer. To request a
            # fresh stochastic sample, use a separate output directory.
            record = read_json(target)
            if record['request'] != request:
                raise ValueError('Cached request mismatch')
            response = record['response']
        else:
            reserve = ((len((prompt + (system or '')).encode()) + 512) * self.input_price
                       + limit * self.output_price) / 1e6
            # Bytes approximate an input-token upper estimate. This is a local
            # spend estimate, not a provider-enforced cap or a billing guarantee.
            if self.spent + reserve > self.budget:
                raise RuntimeError('Estimated budget reached; cached progress is preserved')
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_call)))
            self.charge(reserve)
            self.last_call = time.monotonic()
            url = ('https://generativelanguage.googleapis.com/v1beta/models/'
                   + urllib.parse.quote(model, safe='') + ':generateContent')
            headers = {'Content-Type': 'application/json', 'x-goog-api-key': self.key}
            if self.provider == 'openrouter':
                url = 'https://openrouter.ai/api/v1/chat/completions'
                headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key}
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as handle:
                    response = json.load(handle)
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 401, 402, 403, 404, 429):
                    self.charge(-reserve)
                # No raw exception/request output: credentials must never enter logs.
                raise RuntimeError(f'{self.provider} HTTP {exc.code} for {model}; progress saved. '
                                   'Check model access/quota, then rerun the same command.') from None
            except (urllib.error.URLError, TimeoutError):
                raise RuntimeError('Model connection failed; reservation retained in spend.json '
                                   'because billing is uncertain. Progress saved.') from None
            usage = response.get('usageMetadata', {})
            cost = ((usage.get('promptTokenCount', 0) * self.input_price
                     + (usage.get('candidatesTokenCount', 0) + usage.get('thoughtsTokenCount', 0))
                     * self.output_price) / 1e6) if usage else reserve
            if self.provider == 'openrouter':
                usage = response.get('usage') or {}
                actual = usage.get('cost')
                cost = (float(actual) if isinstance(actual, (int, float))
                        and math.isfinite(actual) and actual >= 0 else reserve)
            save_json(target, {'request': request, 'response': response, 'estimated_usd': cost})
            # Settle the reserved estimate using reported usage/cost when present.
            self.charge(cost - reserve)
        if self.provider == 'openrouter':
            # Reject silent model substitution and incomplete content. Invalid
            # responses remain cached, so rerunning alone may repeat the error.
            choices = response.get('choices', [])
            returned_model = response.get('model', '')
            if returned_model != model:
                raise RuntimeError(f'OpenRouter returned unexpected model {returned_model!r}')
            if not choices or choices[0].get('finish_reason') != 'stop':
                raise RuntimeError('Incomplete or blocked model response; no placeholder will be exported')
            text = choices[0].get('message', {}).get('content')
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError('Empty model response; no placeholder will be exported')
            return text.strip()
        candidates = response.get('candidates', [])
        if not candidates or candidates[0].get('finishReason') != 'STOP':
            raise RuntimeError('Incomplete or blocked model response; no placeholder will be exported')
        text = ''.join(p.get('text', '') for p in candidates[0].get('content', {}).get('parts', [])
                       if not p.get('thought')).strip()
        if not text:
            raise RuntimeError('Empty model response; no placeholder will be exported')
        return text


def generate_one(item, api, encoder, embedding_info):
    # This is an offline experiment: strong labels are generated for every claim.
    # It is not the paper's deployment-only path, which omits the strong judge.
    context, query = item['context'], item['query']
    prompt = f'Context:\n{context}\n\nQuestion: {query}\n\nAnswer:'
    generator = os.getenv('GENERATOR_MODEL', 'gemini-2.5-flash')
    decomposer = os.getenv('DECOMPOSER_MODEL', 'gemini-2.5-flash')
    weak_judge = os.getenv('WEAK_JUDGE_MODEL', 'gemini-2.5-flash')
    strong_judge = os.getenv('STRONG_JUDGE_MODEL', 'gemini-3-flash-preview')
    temperature = float(os.getenv('GENERATION_TEMPERATURE', '0.7'))
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError('GENERATION_TEMPERATURE must be in [0, 2]')
    cap = number('MAX_FACTS_PER_RESPONSE', 20, integer=True)
    # Per-stage overrides support the original pipeline's separate budgets.
    # With no overrides, preserve the current thinking-model allowance.
    default_limit = number('MAX_OUTPUT_TOKENS', 8192, integer=True)
    limits = {stage: number(stage + '_MAX_OUTPUT_TOKENS', default_limit, integer=True)
              for stage in ('GENERATOR', 'DECOMPOSER', 'WEAK_JUDGE', 'STRONG_JUDGE')}
    responses = [api.call(generator, prompt, temperature, system=p,
                          max_tokens=limits['GENERATOR']) for p in DEFAULT_AGENT_PROMPTS]
    # Calls run sequentially. With C retained claims, an uncached query normally
    # needs 5 generation + 5 extraction + 2*C annotation calls (at most 210).
    facts = []
    for index, response in enumerate(responses):
        extracted = parse_claims(api.call(decomposer, DECOMPOSITION_PROMPT.format(text=response),
            system='You are a precise fact extractor. Extract atomic facts as a JSON array.',
            json_mode=True, max_tokens=limits['DECOMPOSER']))
        for claim in extracted[:cap]:
            # Take at most 20 claims per answer by default. Duplicates across
            # answers are preserved, so repeated claims still count separately.
            weak = api.call(weak_judge, NLI_PROMPT.format(context=context, claim=claim),
                            max_tokens=limits['WEAK_JUDGE'])
            if not re.fullmatch(r'\d{1,3}', weak) or not 0 <= int(weak) <= 100:
                raise ValueError('Weak judge must return an integer from 0 to 100')
            label = api.call(strong_judge, VERIFICATION_PROMPT.format(context=context, claim=claim),
                             max_tokens=limits['STRONG_JUDGE'])
            if label not in ('SUPPORTED', 'NOT_SUPPORTED'):
                raise ValueError('Strong judge must return SUPPORTED or NOT_SUPPORTED')
            facts.append({'text': claim, 'source_response_idx': index,
                          'weak_score': int(weak) / 100, 'strong_label': int(label == 'SUPPORTED')})
        print(f"{item['query_id']}: completed response {index + 1}/5, {len(facts)} claims", flush=True)
    embeddings = encoder.encode(responses, convert_to_numpy=True, normalize_embeddings=True)
    # Entropy describes the five full responses, not the extracted claims.
    # Normalized embeddings turn dot products into cosine similarities.
    similarity = np.clip(embeddings @ embeddings.T, 0, 1)
    similarity = np.clip((similarity + similarity.T) / 2, 0, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(similarity), 1e-10)
    # Entrywise clipping can affect the spectrum; flooring avoids negative/zero
    # eigenvalues before normalization. This follows the archived computation.
    probabilities = eigenvalues / eigenvalues.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-10)))
    return {**item, 'ensemble_responses': responses, 'ensemble_entropy': entropy, 'facts': facts,
            'embedding': embedding_info}


def export_snapshot(records, destination, base=None):
    """Merge only fully annotated records, preserving existing numerical rows."""
    destination.mkdir(parents=True, exist_ok=True)
    if base and destination.resolve() == base.resolve():
        raise ValueError('Export to a new directory so the original snapshot stays recoverable')
    rows = []
    manifest = {'datasets': {}}
    if base:
        # Extension mode requires both base datasets, but only homogeneous rows
        # are added. Existing heterogeneous data are copied unchanged.
        load_data(base, 'homogeneous')
        load_data(base, 'heterogeneous')
        rows = read_json(base / 'homogeneous.json.gz')
        previous = read_json(base / 'manifest.json')
        shutil.copy2(base / 'heterogeneous.json.gz', destination / 'heterogeneous.json.gz')
        manifest['datasets']['heterogeneous'] = previous['datasets']['heterogeneous']
    ids = {r['id'] for r in rows}
    for record in records:
        if record['query_id'] in ids:
            raise ValueError(f"Refusing to replace existing tuple {record['query_id']}")
        facts = record['facts']
        row = {'id': record['query_id'], 'weak': [f['weak_score'] for f in facts],
               'entropy': [record['ensemble_entropy']] * len(facts),
               'labels': [f['strong_label'] for f in facts]}
        # One entropy scalar is repeated per claim to fit ClaimTuple's shape.
        # Text, agent provenance and embedding metadata remain in records/ only.
        ClaimTuple(row['id'], row['weak'], row['entropy'], row['labels'])
        rows.append(row)
        ids.add(row['id'])
    rows.sort(key=lambda row: row['id'])
    info = pack(destination / 'homogeneous.json.gz', rows)
    manifest['datasets']['homogeneous'] = {**info, 'tuples': len(rows),
                                           'claims': sum(len(r['labels']) for r in rows)}
    orders = {}
    # Rebuild permutations for the exported size. Adding a tuple changes the
    # partitions even when the seeds stay the same.
    for size, seeds in ((len(rows), [*range(42, 62), *range(1000, 1020)]), (100, range(5000, 5050))):
        for seed in seeds:
            orders[f'{size}:{seed}'] = np.random.default_rng(seed).permutation(size).tolist()
    manifest['splits'] = pack(destination / 'splits.json.gz', orders)
    save_json(destination / 'manifest.json', manifest)
    load_data(destination, 'homogeneous')
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    evaluate = commands.add_parser('evaluate', help='Run saved-data experiments without API calls')
    evaluate.add_argument('--data', type=Path, default=ROOT / 'data')
    evaluate.add_argument('--output', type=Path, default=ROOT / 'outputs')
    evaluate.add_argument('--panel', choices=('main', 'gamma', 'heterogeneous', 'all'), default='all')
    evaluate.add_argument('--include-legacy', action='store_true')
    evaluate.add_argument('--thresholds', type=int, default=200,
                          help='Candidate count: 200 (original scripts), 100 (paper specification)')
    generate = commands.add_parser('generate', help='Generate, decompose and annotate five responses per input')
    generate.add_argument('--inputs', type=Path, default=ROOT / 'data/inputs.json.gz')
    generate.add_argument('--output', type=Path, default=ROOT / 'outputs/generation')
    generate.add_argument('--env-file', type=Path, default=ROOT / '.env')
    generate.add_argument('--limit', type=int, default=1, help='Number of inputs (default 1, full set 300)')
    generate.add_argument('--ids', nargs='+', help='Select specific query IDs before applying --limit')
    generate.add_argument('--merge-with', type=Path, help='Existing snapshot to extend, written under output/data')
    args = parser.parse_args()
    if args.command == 'evaluate':
        if args.thresholds < 2:
            parser.error('--thresholds must be at least 2')
        # Return before importing embedding/.env packages: offline evaluation
        # needs only the core dependencies and makes no model/API calls.
        args.output.mkdir(parents=True, exist_ok=True)
        panels = ('main', 'gamma', 'heterogeneous') if args.panel == 'all' else (args.panel,)
        for panel in panels:
            rows = run_panel(panel, args.data, args.include_legacy, args.thresholds)
            write_csv(args.output / f'{panel}_splits.csv', rows)
            write_csv(args.output / f'{panel}_summary.csv', summarize(rows))
            print(f'{panel}: {len(rows)} split results written to {args.output}')
        return
    try:
        from dotenv import load_dotenv
        from huggingface_hub import HfApi
        from sentence_transformers import SentenceTransformer
    except ImportError:
        parser.error("Install generation dependencies: python -m pip install -e '.[generation]'")
    load_dotenv(args.env_file, override=False)
    # Existing shell environment variables take precedence over the .env file.
    items = read_json(args.inputs)
    if not isinstance(items, list) or args.limit < 1:
        parser.error('Inputs must be a JSON list and --limit must be positive')
    if args.ids:
        missing = set(args.ids) - {x['query_id'] for x in items}
        if missing:
            parser.error(f'Unknown query IDs: {sorted(missing)}')
        items = [x for x in items if x['query_id'] in args.ids]
    items = items[:args.limit]
    ids = [x['query_id'] for x in items]
    if not items or len(set(ids)) != len(ids):
        parser.error('Select at least one input; query IDs must be unique')
    for item in items:
        if any(not isinstance(item.get(k), str) or not item[k].strip() for k in ('query_id', 'context', 'query')):
            parser.error('Each input requires nonempty query_id, query and context strings')
    if args.merge_with and set(ids) & {t.query_id for t in load_data(args.merge_with, 'homogeneous')}:
        parser.error('Selected query IDs already exist in --merge-with data')
    api = ModelClient(args.output)
    model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    revision = os.getenv('EMBEDDING_REVISION') or HfApi().model_info(model).sha
    encoder = SentenceTransformer(model, revision=revision, device='cpu')
    records = []
    # Resume is via calls/, not by loading records/. Reuse the same ID selection
    # to rebuild the same export; a later run exports only its selected records
    # plus --merge-with, not all records found in the output directory.
    for item in items:
        record = generate_one(item, api, encoder, {'model': model, 'revision': revision})
        # Hash the ID for a safe filename; the original ID remains in the record.
        filename = hashlib.sha256(item['query_id'].encode()).hexdigest() + '.json'
        save_json(args.output / 'records' / filename, record)
        records.append(record)
    count = export_snapshot(records, args.output / 'data', args.merge_with)
    print(f'Exported {count} tuples to {args.output / "data"}; estimated spend ${api.spent:.4f}')
    if count <= 200:
        print('Main/gamma evaluation requires more than 200 tuples (200 are used for calibration).')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
