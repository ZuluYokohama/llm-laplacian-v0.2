# Single-terminal-EOS policy review

The prospective token policy is now `gpt2_story_single_eos_u16le_v2`.
Encoding still uses the same pinned GPT-2 tokenizer with automatic special-token
insertion disabled. Any EOS ID already present in the encoded story causes a
failed, immutable preparation attempt; one EOS is appended only after that
check. Literal `<|endoftext|>` content can encode to EOS even with
`add_special_tokens=False`. The implementation rejects that story instead of
silently stripping content or changing tokenizer behavior. The manifest declares
`tokenizer.interior_eos = "reject"`, and the structural auditor rejects EOS IDs
before a story's final token.

This is a prospective admission-policy clarification, not evidence that the
existing bundle had corrupt boundaries. The historical v1 producer promised
one appended EOS per story and did not explicitly reject special-token content.
Story boundaries and train/test partition identity were recorded separately
in the story indexes; an internal EOS was not itself a split leak.

The immutable [CPU audit](../../runs/publication_study/review_eos_fork_001/eos_audit.json)
and its [standalone source](../../runs/publication_study/review_eos_fork_001/audit_eos_counts.py)
verified every recorded extent and terminal EOS in the historical token bundle,
counted every EOS ID, and checked the raw hashes of all eight token/index files
before and after the audit. It found:

| Split | Stories / terminal EOS IDs | Interior EOS IDs |
|---|---:|---:|
| Training | 1,763,066 | 0 |
| Development | 18,039 | 0 |
| Calibration | 9,085 | 0 |
| Test | 9,058 | 0 |

The bound historical manifest SHA-256 is
`13f6c1493127edf48ba6a2b0310b14d5e8c45f1ee7338830ba038ef968cafb88`.
This host CPU audit is not a Jovin-chain receipt, independent retokenization,
semantic deduplication, or a model-quality evaluation. It supplements the
historical structural audit without rewriting it. The local audit artifacts
must be explicitly included in a later evidence release before public links
are presented as available to readers.

The raw pre-review bytes of the token producer, study trainer, structural
auditor and their tests are preserved in
[source_snapshot.json](../../runs/publication_study/review_eos_fork_001/source_snapshot.json).
The historical token manifest binds the old producer's original live path and
hash. Therefore it is intentionally rejected by the updated source/policy
validation, despite the successful EOS audit. **Do not edit the old manifest's
source hashes, policy or paths to make it pass.**

Before further GPU training or evaluation using these study data, prepare a new
immutable versioned bundle under the
updated producer (or perform a separately reviewed explicit migration), rerun
its structural audit, and create new source-bound study configurations. Existing
pilot configurations/checkpoints remain historical and must not be silently
resumed or relabeled as v2. No bundle regeneration or GPU experiment was
performed as part of this review fix.

The paired-fork admission path now verifies the `last.pt` name, complete
JSON/sidecar receipt, expected run directory and exact checkpoint bytes before
deserializing its full state. It retains configuration/state checks after
loading and rechecks bound checkpoint/receipt bytes before creating the new
branch. These hashes detect drift relative to a trusted local receipt; they do
not make arbitrary untrusted pickle content safe to load.
