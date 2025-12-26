import pandas as pd
from pathlib import Path

# tool function
def path_done_well(*paths):
    path = (p if isinstance(p, Path) else Path(p) for p in paths)
    return path

# itering for batch processing
class PairDataset:
    def __init__(self, img_dir=None, edit_dir=None, **kwargs):

        self.img_dir, self.edit_dir = path_done_well(img_dir, edit_dir)
        self._iter_index = 0

        try:
            img_list = sorted(self.img_dir.glob("*.png"))
            edit_list =sorted(self.edit_dir.glob("*.png"))

            if not img_list or not edit_list:
                raise ValueError("Image or Edit directory is empty.")

            if len(edit_list) != len(img_list):
                print(f"Warning: edit count ({len(edit_list)}) does not equal img count ({len(img_list)}). ")

            self.pairs = pd.DataFrame({
                "img": img_list,
                "edit": edit_list,
            })

        except Exception as e:
            print(f"Error loading dataset pairs: {e}")
        
        self.length = len(self.pairs)

    def prompter(self):
        return "Has this picture edited properly?"

    def __iter__(self):
        return self
    
    def __next__(self):

        if self._iter_index < self.length:
            dat = self.pairs.iloc[self._iter_index]
            img = dat["img"].resolve()
            item = dat["edit"].resolve()
            self._iter_index += 1

            return img, item, self.prompter()
        else:
            raise StopIteration