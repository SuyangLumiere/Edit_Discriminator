from Qwen3VLAuditor import Qwen3VLModel, PairDataset, ResultLogger

def main():
    img_dir = "/input/0001.png"
    edit_dir = "/output/0001.png"

    data = PairDataset(img_dir=img_dir, edit_dir=edit_dir)
    model = Qwen3VLModel(model_path="/path/to/Qwen3-VL-8B-Instruct")
    logger = ResultLogger(save_path="results.jsonl")

    for img_path, edit_path, prompt in data:
        res = model([img_path, edit_path], prompt)
        res_dict = res.to_dict()
        res_dict.update({
            "img_path": str(img_path),
            "edit_path": str(edit_path)
        })
        logger(res_dict)

if __name__ == "__main__":
    main()