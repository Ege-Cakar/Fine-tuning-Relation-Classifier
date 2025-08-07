#!/usr/bin/env python
"""dataset_generator.py

Generate **train/val/test JSONL files** for ModernBERT fine‑tuning from XML‑annotated EDU corpora.

Pipeline
========
1. **Read every XML** in `--xml_dir`.
2. Extract *all* EDU pairs + relation labels.
   * Original labels (`sup`, `add`, `und`) → `support`; `reb` → `rebuttal`; others ignored.
   * Generate `none` pairs for EU pairs *not* explicitly linked.
3. **Balance the `none` class** globally using `--none_ratio` (share of total examples).
4. **Randomly shuffle** and **split** into train/val/test according to `--train_ratio` & `--val_ratio`.
5. Write three JSONL files (`train.jsonl`, `val.jsonl`, `test.jsonl`) under `--output_dir`.

Example
-------
```bash
python dataset_generator.py \
    --xml_dir ./xml_corpus \
    --output_dir ./datasets \
    --none_ratio 0.5 \
    --train_ratio 0.7 --val_ratio 0.15 --seed 123
```

Requirements
------------
```bash
pip install tqdm
```
"""
import argparse
import itertools
import json
import logging
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

# --------------------------- logging ----------------------------
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("dataset_generator")

# ------------------------- CLI parsing --------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate JSONL splits from XML EDU corpora")
    p.add_argument("--xml_dir", required=True, type=str, help="Directory with XML files")
    p.add_argument("--output_dir", required=True, type=str, help="Where to write train/val/test JSONL files")
    p.add_argument("--train_ratio", type=float, default=0.8, help="Fraction of data for training split")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of data for validation split (rest is test)")
    p.add_argument("--none_ratio", type=float, default=0.4, help="Target share of 'none' examples in final dataset")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling")
    return p.parse_args()


# --------------------- Data‑extraction logic --------------------
class XMLDataExtractor:
    """Extract EDU relationship data from XML files (pair generation & label mapping).

    Updated to:
      * Accumulate pairs across *all files* first.
      * Balance 'none' label globally.
    """

    LABEL_MAPPING = {"sup": "support", "reb": "rebuttal", "add": "support", "und": "support"}

    def __init__(self, xml_dir: str):
        self.xml_dir = Path(xml_dir)
        self.xml_files = list(self.xml_dir.glob("*.xml"))
        if not self.xml_files:
            raise FileNotFoundError(f"No .xml files found in {self.xml_dir}")
        logger.info(f"Found {len(self.xml_files)} XML files in {self.xml_dir}")

    # ----------------------- core methods ----------------------
    def extract_all_pairs(self) -> List[Dict]:
        """Return a list with every (edu1, edu2, label) dict across the corpus."""
        all_pairs: List[Dict] = []
        for xml_file in tqdm(self.xml_files, desc="Parsing XML"):
            all_pairs.extend(self._extract_from_single_file(xml_file))
        logger.info(f"Total pairs before balancing: {len(all_pairs)}")
        self._log_class_distribution(all_pairs)
        return all_pairs

    # -------------------- single‑file helper -------------------
    def _extract_from_single_file(self, xml_file: Path) -> List[Dict]:
        """Extract labelled and implicit 'none' pairs from one XML file."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # ---- pull EDUs ----
            edus: Dict[str, str] = {}
            for edu in root.findall("edu"):
                edu_id = edu.get("id")
                # Text may be inside CDATA or direct
                edu_text = (edu.text or "").strip() or "".join(edu.itertext()).strip()
                edus[edu_id] = edu_text

            # ---- pull ADUs ---- (only need IDs)
            adus = {adu.get("id") for adu in root.findall("adu")}

            # ---- map EDU ↔ ADU via seg edges ----
            edu_to_adu, adu_to_edu = {}, {}
            for edge in root.findall("edge"):
                if edge.get("type") == "seg":
                    edu_id = edge.get("src")
                    adu_id = edge.get("trg")
                    if edu_id in edus and adu_id in adus:
                        edu_to_adu[edu_id] = adu_id
                        adu_to_edu[adu_id] = edu_id

            # ---- explicit relations (support / rebuttal) ----
            labelled_pairs = []
            explicit_pairs_set = set()
            for edge in root.findall("edge"):
                edge_type = edge.get("type")
                if edge_type in self.LABEL_MAPPING:
                    src_adu, trg_adu = edge.get("src"), edge.get("trg")
                    src_edu, trg_edu = adu_to_edu.get(src_adu), adu_to_edu.get(trg_adu)
                    if src_edu and trg_edu and src_edu in edus and trg_edu in edus:
                        label = self.LABEL_MAPPING[edge_type]
                        labelled_pairs.append({
                            "edu1": edus[src_edu],
                            "edu2": edus[trg_edu],
                            "label": label,
                            "source_file": xml_file.name,
                            "src_edu": src_edu,
                            "trg_edu": trg_edu,
                        })
                        explicit_pairs_set.add((src_edu, trg_edu))
                        explicit_pairs_set.add((trg_edu, src_edu))  # avoid symmetric duplicate

            # ---- generate 'none' pairs ----
            mapped_edu_ids = list(edu_to_adu.keys())
            none_pairs = []
            for edu1_id, edu2_id in itertools.combinations(mapped_edu_ids, 2):
                if (edu1_id, edu2_id) not in explicit_pairs_set:
                    none_pairs.append({
                        "edu1": edus[edu1_id],
                        "edu2": edus[edu2_id],
                        "label": "none",
                        "source_file": xml_file.name,
                        "src_edu": edu1_id,
                        "trg_edu": edu2_id,
                    })
            return labelled_pairs + none_pairs

        except Exception as e:
            logger.error(f"Failed parsing {xml_file.name}: {e}")
            return []

    # ----------------- utils: class balance & log --------------
    @staticmethod
    def balance_none_class(data: List[Dict], none_ratio: float, seed: int) -> List[Dict]:
        """Down‑sample 'none' to keep its share ≤ none_ratio."""
        rand = random.Random(seed)
        pos_examples = [d for d in data if d["label"] != "none"]
        none_examples = [d for d in data if d["label"] == "none"]

        target_none = int(len(pos_examples) * none_ratio / (1 - none_ratio))
        if len(none_examples) > target_none:
            none_examples = rand.sample(none_examples, target_none)
        logger.info(f"After balancing: pos={len(pos_examples)}, none={len(none_examples)} (ratio ≈ {none_ratio})")
        return pos_examples + none_examples

    @staticmethod
    def _log_class_distribution(data: List[Dict]):
        dist = Counter(d["label"] for d in data)
        msg = ", ".join(f"{lbl}={cnt} ({cnt/len(data)*100:.1f}%)" for lbl, cnt in dist.items())
        logger.info(f"Class distribution: {msg}")


# -------------------------- main entry --------------------------

def main():
    args = parse_args()
    rand = random.Random(args.seed)

    # 1 — extract
    extractor = XMLDataExtractor(args.xml_dir)
    pairs = extractor.extract_all_pairs()

    # 2 — balance
    pairs = XMLDataExtractor.balance_none_class(pairs, args.none_ratio, args.seed)
    extractor._log_class_distribution(pairs)

    # 3 — shuffle & split
    rand.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }
    logger.info("Split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # 4 — write JSONL
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, records in splits.items():
        out_file = out_dir / f"{split}.jsonl"
        with out_file.open("w", encoding="utf-8") as fw:
            for r in records:
                fw.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(records)} records → {out_file}")


if __name__ == "__main__":
    main()
