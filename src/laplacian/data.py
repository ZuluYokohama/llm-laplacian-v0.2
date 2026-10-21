"""TinyStories memmaps on D:, plus the fixed prompt files used by evaluation."""

from __future__ import annotations

import json

import numpy as np

from laplacian.runtime import CACHE, ROOT, USER_HUB, apply_env, write_metrics

DATA = ROOT / "data"


def snapshot_dir(repo_id):
    folder = "models--" + repo_id.replace("/", "--")
    for root in (USER_HUB, CACHE / "hf" / "hub"):
        pointer = root / folder / "refs" / "main"
        if not pointer.exists():
            continue
        snap = root / folder / "snapshots" / pointer.read_text(encoding="utf-8").strip()
        if snap.exists():
            return snap
    return None


def load_tokenizer(repo_id):
    from transformers import AutoTokenizer

    local = snapshot_dir(repo_id)
    if local is not None:
        return AutoTokenizer.from_pretrained(str(local), local_files_only=True)
    return AutoTokenizer.from_pretrained(repo_id)


def resolve_eos(tokenizer):
    """Return an end-of-sequence id.

    The cached GPT-2 tokenizer leaves eos_token_id empty. Its vocabulary still
    contains <|endoftext|> at 50256, which is the GPT-2 end marker.
    """
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id), "eos_token_id"
    unknown = getattr(tokenizer, "unk_token_id", None)
    for token in ("<|endoftext|>", "<|im_end|>"):
        converted = tokenizer.convert_tokens_to_ids(token)
        if isinstance(converted, int) and converted >= 0 and converted != unknown:
            return converted, token
    raise RuntimeError("tokenizer has no eos id")


def _write_split(dataset, tokenizer, path):
    eos, _source = resolve_eos(tokenizer)
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in dataset:
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            ids.append(int(eos))
            if max(ids) > 65535:
                raise RuntimeError("token id does not fit uint16")
            array = np.asarray(ids, dtype=np.uint16)
            handle.write(array.tobytes())
            count += int(array.shape[0])
            if count // 200_000 != (count - array.shape[0]) // 200_000:
                print(f"PROGRESS tokenize {path.name} tokens={count}", flush=True)
    print(f"PROGRESS tokenize {path.name} tokens={count} done", flush=True)
    return count


def _load_split(name):
    from datasets import load_dataset

    return load_dataset("roneneldan/TinyStories", split=name)


def write_prompt_files():
    DATA.mkdir(parents=True, exist_ok=True)
    circuit = [
        "A small fox found a red boat and pushed it into the pond.",
        "Mina counted three blue stones and put them in a tin box.",
        "The robot watered one plant and left the other plant dry.",
        "Leo hid the key under the mat and told nobody.",
        "A duck asked the frog where the warm mud was.",
        "The baker gave the last bun to the child in the yellow hat.",
        "Two kites tangled, and only the green kite came down.",
        "Nia opened the round door and closed the square door.",
        "The clock stopped at noon, then the bell rang once.",
        "A cat watched the fish but did not touch the bowl.",
        "The train took the short tunnel and skipped the long bridge.",
        "Sam wrote the word moon and erased the word sun.",
        "The painter used red on the roof and blue on the door.",
        "A seed grew in the cup and not in the shoe.",
        "The pilot waved left, and the crew waved right.",
        "The library lamp was on, and the hallway lamp was off.",
    ]
    (DATA / "circuit_prompts.json").write_text(json.dumps(circuit, indent=2) + "\n", encoding="utf-8")
    facts = [
        ("What color is the sky on a clear day?", "blue"),
        ("How many legs does a typical dog have?", "4"),
        ("What do bees make?", "honey"),
        ("Which season comes after winter?", "spring"),
        ("What is frozen water called?", "ice"),
        ("How many days are in a week?", "7"),
        ("What gas do humans breathe in to live?", "oxygen"),
        ("What is the opposite of hot?", "cold"),
        ("How many hours are in a day?", "24"),
        ("What do plants need from the sun?", "light"),
        ("Which ocean is the largest?", "Pacific"),
        ("What is H2O?", "water"),
        ("How many months are in a year?", "12"),
        ("What tool cuts paper?", "scissors"),
        ("Where do fish live?", "water"),
        ("What is 10 minus 3?", "7"),
        ("What planet is known as the red planet?", "Mars"),
        ("How many wheels does a bicycle have?", "2"),
        ("What do cows drink?", "water"),
        ("Which direction does the sun rise?", "east"),
    ]
    capitals = [
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Canada", "Ottawa"),
        ("Egypt", "Cairo"),
        ("Kenya", "Nairobi"),
        ("Peru", "Lima"),
        ("Norway", "Oslo"),
        ("Greece", "Athens"),
        ("Thailand", "Bangkok"),
        ("Portugal", "Lisbon"),
        ("Austria", "Vienna"),
        ("Poland", "Warsaw"),
        ("Ireland", "Dublin"),
        ("Chile", "Santiago"),
        ("Finland", "Helsinki"),
        ("Morocco", "Rabat"),
        ("Vietnam", "Hanoi"),
        ("New Zealand", "Wellington"),
    ]
    rows = [{"prompt": question, "answer": answer} for question, answer in facts]
    rows.extend(
        {"prompt": f"What is the capital of {country}?", "answer": capital} for country, capital in capitals
    )
    for left in range(1, 31):
        rows.append({"prompt": f"What is {left} plus {left}?", "answer": str(left + left)})
    for left in range(1, 31):
        rows.append({"prompt": f"What is {left} times 2?", "answer": str(left * 2)})
    rows = rows[:100]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 prompts, built {len(rows)}")
    (DATA / "commonsense.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    ifeval = [
        {"prompt": "Reply with at most eight words about a river.", "checks": [{"type": "max_words", "value": 8}]},
        {"prompt": "Reply with at least twelve words about a mountain.", "checks": [{"type": "min_words", "value": 12}]},
        {"prompt": "Mention the word lantern in a sentence about night.", "checks": [{"type": "must_include", "value": "lantern"}]},
        {"prompt": "Describe a garden without using the word flower.", "checks": [{"type": "must_exclude", "value": "flower"}]},
        {"prompt": "Answer in lowercase about the moon.", "checks": [{"type": "lowercase", "value": True}]},
        {"prompt": "Answer in uppercase about the sun.", "checks": [{"type": "uppercase", "value": True}]},
        {"prompt": "Reply with a JSON object containing the key color.", "checks": [{"type": "json_key", "value": "color"}]},
        {"prompt": "Write exactly two sentences about a bridge.", "checks": [{"type": "sentences", "value": 2}]},
        {"prompt": "Use at most six words and include the word maple.", "checks": [{"type": "max_words", "value": 6}, {"type": "must_include", "value": "maple"}]},
        {"prompt": "Write at least ten lowercase words about rain.", "checks": [{"type": "min_words", "value": 10}, {"type": "lowercase", "value": True}]},
        {"prompt": "Do not use the word dragon. Mention a castle.", "checks": [{"type": "must_exclude", "value": "dragon"}, {"type": "must_include", "value": "castle"}]},
        {"prompt": "Reply with JSON containing the key count and use at most twenty words.", "checks": [{"type": "json_key", "value": "count"}, {"type": "max_words", "value": 20}]},
    ]
    with (DATA / "ifeval_local.jsonl").open("w", encoding="utf-8") as handle:
        for row in ifeval:
            handle.write(json.dumps(row) + "\n")


class TokenStore:
    def __init__(self, path, context=256):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.context = context
        if len(self.tokens) <= context + 2:
            raise RuntimeError(f"{path} is too short to sample")

    def batch(self, count, generator):
        import torch

        limit = len(self.tokens) - self.context - 1
        starts = torch.randint(0, limit, (count,), generator=generator)
        window = self.context + 1
        rows = []
        for start in starts.tolist():
            piece = np.asarray(self.tokens[start : start + window], dtype=np.int64)
            rows.append(torch.from_numpy(piece))
        packed = torch.stack(rows)
        return packed[:, :-1], packed[:, 1:]


def prepare():
    apply_env()
    write_prompt_files()
    print("PROGRESS download roneneldan/TinyStories", flush=True)
    train = _load_split("train")
    try:
        valid = _load_split("validation")
        valid_name = "validation"
    except Exception as exc:
        valid = train.select(range(min(2000, len(train))))
        valid_name = f"train_head_2000 ({exc.__class__.__name__})"
    tokenizer = load_tokenizer("gpt2")
    eos_id, eos_source = resolve_eos(tokenizer)
    train_count = _write_split(train, tokenizer, DATA / "tinystories_train.u16")
    valid_count = _write_split(valid, tokenizer, DATA / "tinystories_val.u16")
    meta = {
        "tokenizer": "gpt2",
        "vocab": int(tokenizer.vocab_size),
        "eos": eos_id,
        "eos_source": eos_source,
        "train_tokens": train_count,
        "val_tokens": valid_count,
        "val_split": valid_name,
        "dtype": "uint16",
    }
    try:
        smol = load_tokenizer("HuggingFaceTB/SmolLM2-135M-Instruct")
        meta["smol_tokenizer"] = "HuggingFaceTB/SmolLM2-135M-Instruct"
        meta["smol_vocab"] = int(smol.vocab_size)
        meta["smol_train_tokens"] = _write_split(train, smol, DATA / "tinystories_smol_train.u16")
        meta["smol_val_tokens"] = _write_split(valid, smol, DATA / "tinystories_smol_val.u16")
    except Exception as exc:
        meta["smol_error"] = str(exc)[:500]
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    digest = write_metrics(
        ROOT / "runs" / "data" / "metrics.json",
        {
            "run_id": "data",
            "complete": True,
            "host_gpu": False,
            "tokenizer": meta["tokenizer"],
            "vocab": meta["vocab"],
            "train_tokens": train_count,
            "val_tokens": valid_count,
            "val_split": valid_name,
            "smol_tokenizer": meta.get("smol_tokenizer"),
            "smol_train_tokens": meta.get("smol_train_tokens"),
            "smol_error": meta.get("smol_error"),
        },
    )
    print(f"DONE data train_tokens={train_count} val_tokens={valid_count} sha256={digest}", flush=True)
    return meta


if __name__ == "__main__":
    prepare()
