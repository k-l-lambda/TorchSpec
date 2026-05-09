#!/usr/bin/env python3
"""Convert Jiuzhang Kimi CSV exports to TorchSpec conversation JSONL."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


VALID_JSON_ESCAPES = set('"\\/bfnrtu')
csv.field_size_limit(sys.maxsize)


@dataclass
class Stats:
    input_archives: int = 0
    input_rows: int = 0
    samples: int = 0
    conversations: int = 0
    messages: int = 0
    skipped_rows: int = 0
    malformed_request_body: int = 0
    empty_output: int = 0
    token_count: int | None = None


def fix_invalid_json_backslashes(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != '\\':
            out.append(ch)
            i += 1
            continue
        if i + 1 >= len(text):
            out.append('\\\\')
            i += 1
            continue
        nxt = text[i + 1]
        if nxt in VALID_JSON_ESCAPES:
            out.append(ch)
        else:
            out.append('\\\\')
        i += 1
    return ''.join(out)


def decode_export_text(raw: str) -> str:
    if not raw:
        return ''
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw


def get_field(row: dict[str, str], name: str) -> str:
    return row.get(name) or row.get(name.lstrip('﻿')) or row.get(f'﻿{name}') or ''


def parse_request_body(raw: str) -> dict[str, Any]:
    candidates = [raw]
    try:
        candidates.append(json.loads(f'"{raw}"'))
    except json.JSONDecodeError:
        pass
    candidates.append(raw.replace('\\"', '"'))

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            try:
                return json.loads(fix_invalid_json_backslashes(candidate))
            except json.JSONDecodeError as fixed_exc:
                last_error = fixed_exc
    raise ValueError(f'cannot parse request_body: {last_error}')


def content_to_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get('text'), str):
                    parts.append(item['text'])
                elif isinstance(item.get('content'), str):
                    parts.append(item['content'])
                elif item.get('type'):
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return '\n'.join(part for part in parts if part)
    return str(content)


def normalize_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = message.get('role') or message.get('from')
    if role == 'human':
        role = 'user'
    elif role == 'gpt':
        role = 'assistant'
    if role not in {'system', 'user', 'assistant', 'tool'}:
        return None

    normalized: dict[str, Any] = {'role': role, 'content': content_to_text(message.get('content', message.get('value', '')))}
    for key in ('reasoning_content', 'thinking_content', 'thinking', 'reasoning'):
        if message.get(key):
            normalized['reasoning_content'] = content_to_text(message[key])
            break
    if message.get('tool_calls') is not None:
        normalized['tool_calls'] = message['tool_calls']
    if message.get('tool_call_id') is not None:
        normalized['tool_call_id'] = message['tool_call_id']
    if message.get('name') is not None:
        normalized['name'] = message['name']
    return normalized


def archive_csv_members(archive: Path) -> Iterable[tarfile.TarInfo]:
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf.getmembers():
            if member.isfile() and member.name.endswith('.csv'):
                yield member


def iter_rows(archive: Path) -> Iterable[dict[str, str]]:
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf.getmembers():
            if not (member.isfile() and member.name.endswith('.csv')):
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            text_file = io.TextIOWrapper(extracted, encoding='utf-8', newline='')
            yield from csv.DictReader(text_file)


def build_sample(row: dict[str, str], row_index: int) -> dict[str, Any] | None:
    body = parse_request_body(get_field(row, 'request_body'))
    messages = body.get('messages') or []
    if not isinstance(messages, list):
        raise ValueError('request_body.messages is not a list')

    conversations: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        normalized = normalize_message(message)
        if normalized is not None:
            conversations.append(normalized)

    output = decode_export_text(get_field(row, 'output_tokens'))
    if output.strip():
        conversations.append({'role': 'assistant', 'content': output})

    if not conversations:
        return None

    trace_id = get_field(row, 'trace_id').strip()
    sample_id = trace_id or f'jiuzhang_{row_index:09d}'
    return {'id': sample_id, 'conversations': conversations}


def count_tokens_with_transformers(jsonl_path: Path, tokenizer_path: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    total = 0
    batch: list[str] = []

    def flush() -> int:
        if not batch:
            return 0
        encoded = tokenizer(batch, add_special_tokens=False, return_attention_mask=False)
        count = sum(len(ids) for ids in encoded['input_ids'])
        batch.clear()
        return count

    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            batch.append(tokenizer.apply_chat_template(sample['conversations'], tokenize=False, add_generation_prompt=False))
            if len(batch) >= 64:
                total += flush()
    total += flush()
    return total


def count_tokens_approx(jsonl_path: Path) -> int:
    total = 0
    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            for message in sample['conversations']:
                total += len(pattern.findall(message.get('content', '')))
    return total


def count_existing_jsonl(jsonl_path: Path, stats_json: Path, tokenizer_path: str | None) -> None:
    stats = Stats()
    roles = Counter()
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            stats.samples += 1
            stats.conversations += 1
            conversations = sample.get('conversations') or []
            stats.messages += len(conversations)
            for message in conversations:
                roles[message.get('role', '')] += 1
    if tokenizer_path:
        stats.token_count = count_tokens_with_transformers(jsonl_path, tokenizer_path)
        token_method = 'transformers_chat_template'
    else:
        stats.token_count = count_tokens_approx(jsonl_path)
        token_method = 'approx_regex'
    payload = asdict(stats)
    payload['roles'] = dict(roles)
    payload['request_types'] = {}
    payload['output_jsonl'] = str(jsonl_path)
    payload['stats_json'] = str(stats_json)
    payload['token_count_method'] = token_method
    stats_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=Path('/data2/datasets'))
    parser.add_argument('--output-jsonl', type=Path, default=Path('/data2/datasets/processed/jiuzhang_kimi_k26_torchspec.jsonl'))
    parser.add_argument('--stats-json', type=Path, default=Path('/data2/datasets/processed/jiuzhang_kimi_k26_torchspec_stats.json'))
    parser.add_argument('--tokenizer-path', default=None)
    parser.add_argument('--no-token-count', action='store_true')
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--count-existing', action='store_true')
    args = parser.parse_args()

    if args.count_existing:
        count_existing_jsonl(args.output_jsonl, args.stats_json, args.tokenizer_path)
        return

    archives = sorted(args.input_dir.glob('export-*.tar.gz'))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)

    stats = Stats(input_archives=len(archives))
    roles = Counter()
    request_types = Counter()
    row_index = 0

    with args.output_jsonl.open('w', encoding='utf-8') as out:
        for archive in archives:
            for row in iter_rows(archive):
                row_index += 1
                stats.input_rows += 1
                request_types[get_field(row, 'request_type')] += 1
                if args.max_rows and row_index > args.max_rows:
                    break
                try:
                    sample = build_sample(row, row_index)
                except Exception:
                    stats.skipped_rows += 1
                    stats.malformed_request_body += 1
                    continue
                if sample is None:
                    stats.skipped_rows += 1
                    continue
                if not decode_export_text(get_field(row, 'output_tokens')).strip():
                    stats.empty_output += 1
                for message in sample['conversations']:
                    roles[message.get('role', '')] += 1
                stats.samples += 1
                stats.conversations += 1
                stats.messages += len(sample['conversations'])
                out.write(json.dumps(sample, ensure_ascii=False) + '\n')
            if args.max_rows and row_index > args.max_rows:
                break

    if not args.no_token_count:
        if args.tokenizer_path:
            stats.token_count = count_tokens_with_transformers(args.output_jsonl, args.tokenizer_path)
        else:
            stats.token_count = count_tokens_approx(args.output_jsonl)

    payload = asdict(stats)
    payload['roles'] = dict(roles)
    payload['request_types'] = dict(request_types)
    payload['output_jsonl'] = str(args.output_jsonl)
    payload['stats_json'] = str(args.stats_json)
    payload['token_count_method'] = 'transformers_chat_template' if args.tokenizer_path else ('disabled' if args.no_token_count else 'approx_regex')
    args.stats_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
