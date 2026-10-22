# G1 study partitions and token bundles

`study_data.py` creates four story-level partitions from the local TinyStories
Arrow IPC streams. Unicode NFKC, case folding, and whitespace collapse define
an exact normalized-content group. The first occurrence is retained and the
group hash determines train/development/calibration/test assignment. The
partition manifest binds each source's raw byte hash and size, all four JSONL
indexes, the seed/policy, and source/unique/duplicate/empty-row counts.

This prevents normalized exact duplicates from crossing these new splits. It
does not establish semantic deduplication, historical non-exposure, or an
independent external test corpus. Historical models may have seen the source
corpus. Confirmatory training must consume these new splits explicitly.

## Offline immutable tokenization

`laplacian.study_tokens.prepare(partitions, output)` creates a new directory;
it refuses to reuse either a successful or a failed attempt. Its production
loader uses `tokenizers.Tokenizer.from_file` on the local GPT-2 `tokenizer.json`
from snapshot `607a30d783dfa663caf39e06633721c8d4cfcd7e`. The exact file hash is
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.
No Hub lookup, network fallback, model loading, `datasets` import, or GPU work
is part of this stage. A copied local tokenizer is acceptable only if its bytes
match this pin. The vocabulary size must be 50,257 and `<|endoftext|>` must map
to 50,256. Padding and truncation are disabled, automatic special-token
insertion is disabled, and one EOS is appended to every selected story.
The prospective `gpt2_story_single_eos_u16le_v2` policy rejects any EOS already
encoded from story content before appending the terminal EOS. See
[`EOS_POLICY_REVIEW.md`](EOS_POLICY_REVIEW.md) for the historical bundle audit
and the requirement to create a new bundle and configurations after this change.

The four sorted JSONL row-reference streams are heap-merged in source-index,
row-index order. Arrow IPC batches are read sequentially through PyArrow. Each
selected source/row pair must exist, have the recorded original-text SHA-256,
and reproduce the recorded normalized story ID and split assignment before it
is encoded. Unsorted/repeated row references, missing rows, mismatched counts,
invalid text, and token IDs outside the pinned vocabulary fail the attempt.
The merge retains only the next reference from each split; it does not load
the corpus or all partition references into memory. One Arrow batch and one
story's encoded token sequence are materialized at a time.

For each split the output contains:

- `<split>.u16`: concatenated **little-endian unsigned 16-bit** token IDs.
- `<split>.stories.jsonl`: original `story_id`, `source_index`, `row_index`, and
  `text_sha256`, plus zero-based `offset` and `length` in **tokens**, including
  the appended EOS. These offsets identify story boundaries for later sampling
  and evaluation policy; tokenization does not decide whether a training window
  may span two stories within the same split.

`manifest.json` records raw output hashes/sizes and story/token counts. It also
binds the partition manifest and indexes, original Arrow sources, tokenizer
bytes/revision/options, both tokenization/partition source modules, Python
version/executable, and installed PyArrow/tokenizers distribution inventories.
Dependency identities hash actual installed files, including Python source,
native libraries, and distribution metadata/RECORD. Generated `__pycache__`
and `.pyc` files are excluded. This is a provenance record for the direct
dependencies and interpreter, not an operating-system or complete transitive
environment image.

Input, code, runtime, and dependency files are rechecked after encoding.
Output files are flushed and hashed before a final atomic manifest replacement
sets `complete: true`. A failure keeps partial outputs, an incomplete manifest,
and `failure.json`; it never promotes that attempt or overwrites another
attempt. A killed process may leave only the incomplete manifest/partial files.
Injected tiny tokenizers are an API-only testing facility, always marked
`production_ready: false` with `tokenizer.mode: injected_test_double`.

## Consumer contract

`validate_bundle(path_to_manifest_json)` is read-only and returns a validated
manifest. It requires a complete production bundle with the pinned GPT-2
policy; verifies all eight output files' raw hashes/sizes; revalidates the
bound partition, JSONL, and Arrow inputs; checks counts and token-byte lengths;
and checks bound tokenizer, code, dependency, and interpreter records. A
consumer must also bind the raw token-manifest hash into its own run identity.

The provenance uses absolute local paths. The original files must remain
available and intact for validation; relocation or environment upgrades need
an explicit audited migration, not an edited hash masquerading as the old
identity. Hashes detect changes relative to the manifest but do not authenticate
the person who created or edited it. A complete token bundle does not itself
authorize a training run or establish scientific claims.

The trainer must receive `manifest_path.parent / "train.u16"` explicitly,
`manifest["tokenizer"]["vocab_size"]`, and the bound data identity. It must not
silently fall back to historical memmaps. Development, calibration, and test
use their respective files. This document describes the data artifact; trainer
integration and confirmatory evaluation remain separate gates.

## CPU verification and later execution

Tests use tiny real Arrow IPC streams and an injected tokenizer. They check
global source-order selection across all splits, token offsets/EOS, raw hashes,
overflow/invalid IDs, missing rows, wrong content and story identities,
unsorted references, immutable outputs, mutations during encoding, and
consumer rejection of modified files and nonproduction fixtures.
When the pinned local snapshot is present, a further smoke test tokenizes eight
fixture stories with the real GPT-2 tokenizer, captures the actual dependency
inventory, and validates the resulting tiny production-format bundle. This is
not a run over the full corpus.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_study_tokens.py tests/test_study_data.py -q --tb=short --basetemp cache/g1_tokens_checks_unique
```

Use a new dedicated `--basetemp` because pytest may clean that directory.
The following is the production command shape, **not a record that full
tokenization has run**; the output directory must not already exist:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
.\.venv\Scripts\python.exe -m laplacian.study_tokens --partitions data/publication_v1/partitions --output data/publication_v1/tokens_v2
```

API usage was checked through Context7 against the official
[Tokenizers Python documentation](https://github.com/huggingface/tokenizers/blob/main/bindings/python/README.md)
and [Arrow IPC documentation](https://github.com/apache/arrow/blob/main/docs/source/python/ipc.rst).
Installed versions during implementation were tokenizers 0.23.3 and PyArrow
26.0.0; each real bundle records its actual installed-file identities.
