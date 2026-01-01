import traceback
import pandas as pd
from pathlib import Path

# tool function
def path_done_well(*paths):
    path = (p if isinstance(p, Path) else Path(p) for p in paths)
    return path

# itering for batch processing
class PairDataset:
    def __init__(self, img_dir=None, edit_dir=None, resume_from=None, **kwargs):

        self.img_dir, self.edit_dir = path_done_well(img_dir, edit_dir)
        self.resume_path = path_done_well(resume_from) if resume_from else None
        self.rank = None
        self.world_size = None
        
        try:
            img_list = [p.resolve() for p in sorted(self.img_dir.glob("*.png"))]
            edit_list = [p.resolve() for p in sorted(self.edit_dir.glob("*.png"))]

            if not img_list or not edit_list:
                raise ValueError("Image or Edit directory is empty.")

            if len(edit_list) != len(img_list):
                print(f"Warning: edit count ({len(edit_list)}) does not equal img count ({len(img_list)}). ")

            self.pairs = pd.DataFrame({
                "img": img_list,
                "edit": edit_list,
            })

        except Exception as e:
            traceback.print_exc()
            # print(f"Error loading dataset pairs: {e}")

        if self.resume_path and self.resume_path.exists():
            print(f"Found resume file: {self.resume_path}, calculating remaining tasks...")
            self.length = self.get_resume()
        else:
            self.length = len(self.pairs)


    def get_resume(self):
        try:
            dat = pd.read_json(self.resume_path, lines=True)['edit_path']
            self.pairs = self.pairs[~self.pairs["img"].astype(str).isin(set(dat))]
            length = len(self.pairs)
            print(f"Resumed: {length} tasks remaining...")

        except Exception as e:
            traceback.print_exc()
            print(f"Error resuming from {self.resume_path}: {e}")

        return length

    def prompter(self):
        return "Has this picture edited properly?"
    
    def split(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        if not self.rank:
            for _, row in self.pairs.iterrows():
                img = row["img"]
                item = row["edit"]
                yield img, item, self.prompter()
        else:
            data = self.pairs.iloc[self.rank::self.world_size]
            for _, row in data.iterrows():
                img = row["img"]
                item = row["edit"]
                yield img, item, self.prompter()


