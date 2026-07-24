#!/usr/bin/env python3
"""Tokenize a bundled public-domain text with a model's tokenizer into a train/held-out token
split, for held-out perplexity evaluation of SALT-distillation (plan 0042).

Usage: python3 tools/gen_corpus.py <tokenizer.json> <out.json> [--frac 0.75]
Writes {"train_ids": [...], "eval_ids": [...]} — eval is a HELD-OUT tail slice the distillation
never trains on, so its ppl measures generalization (not the in-sample memorization we had before).
"""
import json
import sys

from tokenizers import Tokenizer

# ~900 words of Alice's Adventures in Wonderland, ch.1 (public domain) — coherent English so
# perplexity is meaningful. Train on the head, evaluate on the held-out tail.
TEXT = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having
nothing to do: once or twice she had peeped into the book her sister was reading, but it had no
pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or
conversations?" So she was considering in her own mind (as well as she could, for the hot day made
her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the
trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close
by her. There was nothing so very remarkable in that; nor did Alice think it so very much out of the
way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over
afterwards, it occurred to her that she ought to have wondered at this, but at the time it all
seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and
looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she
had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and
burning with curiosity, she ran across the field after it, and fortunately was just in time to see it
pop down a large rabbit-hole under the hedge. In another moment down went Alice after it, never once
considering how in the world she was to get out again. The rabbit-hole went straight on like a
tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think
about stopping herself before she found herself falling down a very deep well. Either the well was
very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and
to wonder what was going to happen next. First, she tried to look down and make out what she was
coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed
that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung
upon pegs. She took down a jar from one of the shelves as she passed; it was labelled "ORANGE
MARMALADE", but to her great disappointment it was empty: she did not like to drop the jar for fear
of killing somebody underneath, so managed to put it into one of the cupboards as she fell past it.
"Well!" thought Alice to herself, "after such a fall as this, I shall think nothing of tumbling down
stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell
off the top of the house!" (Which was very likely true.) Down, down, down. Would the fall never come
to an end? "I wonder how many miles I've fallen by this time?" she said aloud. "I must be getting
somewhere near the centre of the earth. Let me see: that would be four thousand miles down, I think"
(for, you see, Alice had learnt several things of this sort in her lessons in the schoolroom, and
though this was not a very good opportunity for showing off her knowledge, as there was no one to
listen to her, still it was good practice to say it over) "yes, that's about the right distance, but
then I wonder what Latitude or Longitude I've got to?" (Alice had no idea what Latitude was, or
Longitude either, but thought they were nice grand words to say.) Presently she began again. "I
wonder if I shall fall right through the earth! How funny it'll seem to come out among the people
that walk with their heads downward! The Antipathies, I think." She was rather glad there was no one
listening, this time, as it didn't sound at all the right word. "But I shall have to ask them what
the name of the country is, you know. Please, Ma'am, is this New Zealand or Australia?" and she tried
to curtsey as she spoke, fancy curtseying as you're falling through the air! Do you think you could
manage it?"""


def arg(flag: str, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def load_text(path: str) -> str:
    """Read a corpus source into one string. `.parquet` reads the `text` column (WikiText/C4 layout);
    any other file is read raw with Project Gutenberg START/END boilerplate stripped."""
    if path.endswith(".parquet"):
        import pandas as pd  # local: only needed for the parquet path

        # fillna("") guards exporters (some C4 shards) that store empty lines as NaN, which would
        # otherwise make tolist() yield floats and "".join raise TypeError.
        return "".join(pd.read_parquet(path)["text"].fillna("").tolist())
    raw = open(path, encoding="utf-8", errors="ignore").read()
    s = raw.find("*** START")
    if s != -1:
        raw = raw[raw.find("\n", s) + 1 :]
    e = raw.find("*** END")
    if e != -1:
        raw = raw[:e]
    return raw


def main() -> None:
    # Usage: gen_corpus.py <tokenizer.json> <out.json>
    #          [--file TRAIN] [--eval-file EVAL] [--pool N] [--heldout N]
    # --file / --eval-file may be raw text or .parquet (WikiText/C4). With --eval-file the held-out set
    # is a SEPARATE corpus (e.g. WikiText test split) — a stronger disjointness than the tail split.
    tok_path = sys.argv[1]
    out_path = sys.argv[2]
    pool = int(arg("--pool", 8192))       # train pool = first `pool` train tokens
    heldout = int(arg("--heldout", 256))  # held-out = `heldout` tokens (disjoint from train)
    tok = Tokenizer.from_file(tok_path)
    encode = lambda t: tok.encode(" ".join(t.split())).ids  # noqa: E731

    train_text = load_text(arg("--file", "")) if "--file" in sys.argv else TEXT
    train_ids = encode(train_text)

    if "--eval-file" in sys.argv:
        # Held-out from a distinct corpus (fully disjoint from train).
        eval_ids_all = encode(load_text(arg("--eval-file", "")))
        assert len(train_ids) >= pool, f"train too short: {len(train_ids)} < {pool}"
        assert len(eval_ids_all) >= heldout, f"eval too short: {len(eval_ids_all)} < {heldout}"
        out = {"train_ids": train_ids[:pool], "eval_ids": eval_ids_all[:heldout]}
        src = f"train {len(train_ids)} / eval {len(eval_ids_all)} tokens (separate corpora)"
    else:
        # Single corpus: held-out is the tail slice just past the train pool (disjoint).
        assert len(train_ids) >= pool + heldout, (
            f"corpus too short: {len(train_ids)} < {pool + heldout}"
        )
        out = {"train_ids": train_ids[:pool], "eval_ids": train_ids[pool : pool + heldout]}
        src = f"{len(train_ids)} tokens (tail-split)"

    with open(out_path, "w") as f:
        json.dump(out, f)
    print(
        f"{src} → train pool {len(out['train_ids'])}, "
        f"held-out {len(out['eval_ids'])} (disjoint) → {out_path}"
    )


if __name__ == "__main__":
    main()
