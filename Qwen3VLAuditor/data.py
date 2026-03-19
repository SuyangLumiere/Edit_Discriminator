import pandas as pd
from typing import List, Dict, Optional

class FlexiblePairDataset:
    def __init__(self, data: List[Dict[str, str]], resume_path: Optional[str] = None):
        self.df = pd.DataFrame(data)
        if resume_path:
            try:
                processed = pd.read_json(resume_path, lines=True)['edit'].astype(str).tolist()
                self.df = self.df[~self.df['edit'].astype(str).isin(processed)]
                print(f"Resumed from {resume_path}, {len(self.df)} tasks remaining.")
            except Exception as e:
                print(f"Resume file missing or invalid: {e}. Starting from scratch.")

    def split(self, rank: int, world_size: int):
        if world_size > 1:
            self.df = self.df.iloc[rank::world_size]
            print(f"Rank {rank}/{world_size} took {len(self.df)} pairs.")
        return self

    def __iter__(self):
        for _, row in self.df.iterrows():
            prompt = row.get("prompt", "Edit this picture properly.")
            yield row["img"], row["edit"], prompt

    def __len__(self):
        return len(self.df)





