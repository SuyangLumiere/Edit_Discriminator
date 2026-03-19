import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Union

# tool function
def path_done_well(*paths):
    return [p if isinstance(p, Path) else Path(p) for p in paths]

def build_pair_list(
    img_paths: List[Union[str, Path]] = None, 
    edit_paths: List[Union[str, Path]] = None, 
    edit_prompts: List[str] = (),
) -> List[Dict[str, str]]:
    """
    construct input pairs
    """
    if len(img_paths) != len(edit_paths):
        print(f"Warning: Image count ({len(img_paths)}) != Edit count ({len(edit_paths)}). Data might be misaligned!")

    has_prompts = True
    if len(img_paths) != len(edit_prompts):
        has_prompts = False
        print(f"Warning: Image count ({len(img_paths)}) != Prompt count ({len(edit_prompts)}). Data might be misaligned!")

    pairs = []
    for i in range(len(img_paths)):
        item = {
            "img": str(img_paths[i]),
            "edit": str(edit_paths[i])
        }
        if has_prompts:
            item["prompt"] = edit_prompts[i]

        pairs.append(item)
        
    return pairs

def filter_oversized_pairs(
    pairs: List[Dict[str, str]], 
    max_pixels: int = 2048 * 2048, 
    max_edge: int = 3000
) -> List[Dict[str, str]]:
    
    valid_pairs = []
    dropped_count = 0

    print(f"Checking {len(pairs)} pairs for size limits...")

    for item in tqdm(pairs, desc="Filtering Images"):
        try:
            with Image.open(item["img"]) as im:
                w, h = im.size
            if (w * h > max_pixels) or (w > max_edge) or (h > max_edge):
                dropped_count += 1
                continue

            with Image.open(item["edit"]) as im_edit:
                w_e, h_e = im_edit.size
            if (w_e * h_e > max_pixels) or (w_e > max_edge) or (h_e > max_edge):
                dropped_count += 1
                continue

            valid_pairs.append(item)

        except Exception as e:
            dropped_count += 1

    print(f"Done! Dropped {dropped_count} oversized/error pairs. Kept {len(valid_pairs)} pairs.")
    return valid_pairs

class ResultLogger:
    def __init__(self, save_path="results.jsonl"):
        self.save_path = save_path

    def __call__(self, data):
        self.batch_log([data])

    def batch_log(self, results):
        with open(self.save_path, "a", encoding="utf-8") as f:
            for item in results:
                f.write(ResultLogger.to_json(item)+ "\n")
    
    @staticmethod
    def to_json(result) -> str:
        return json.dumps(result.to_dict(), ensure_ascii=False) if hasattr(result, "to_dict") else json.dumps(result, ensure_ascii=False)