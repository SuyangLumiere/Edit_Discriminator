from Qwen3VLAuditor import Qwen3VLModel, PairDataset, ResultLogger
from tqdm import tqdm

def main():
    img_dir = "/input/"
    edit_dir = "/output/"

    data = PairDataset(img_dir=img_dir, edit_dir=edit_dir)
    model = Qwen3VLModel(model_path="/path/to/Qwen3-VL-8B-Instruct")
    logger = ResultLogger(save_path="results.jsonl")

    with tqdm(data, desc="Processing") as pbar:
        for img_path, edit_path, prompt in pbar:
            res = model([img_path, edit_path], prompt)
            pbar.set_postfix(score=res.score)
            res_dict = res.to_dict()
            res_dict.update({
                "img_path": str(img_path),
                "edit_path": str(edit_path)
            })
            logger(res_dict)

if __name__ == "__main__":
    main()