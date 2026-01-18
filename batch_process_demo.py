import traceback
from tqdm import tqdm
import torch.multiprocessing as mp
from Qwen3VLAuditor import Qwen3VLModel, PairDataset, ResultLogger


def worker_process(model_path, data, rank, world_size):

    try:
        model = Qwen3VLModel(model_path=model_path)
        logger = ResultLogger(save_path=f"result{rank}.jsonl")
    except Exception as e:
        print(f"[Worker {rank}] Error: {e}")
        traceback.print_exc()
        return
    
    data.split(rank=rank, world_size=world_size)

    with tqdm(data, desc=f"Process{rank}", position=rank) as pbar:
        for img_path, edit_path, prompt in pbar:
            res = model([img_path, edit_path], prompt)
            pbar.set_postfix(score=res.score)
            res_dict = res.to_dict()
            res_dict.update({
                "img_path": str(img_path),
                "edit_path": str(edit_path)
            })
            logger(res_dict)


def multi_workers():
    world_size = 8
    img_dir = "/input/"
    edit_dir = "/output/"
    model_path = "/path/to/Qwen3-VL-8B-Instruct"

    data = PairDataset(img_dir=img_dir, edit_dir=edit_dir)

    mp.set_start_method('spawn', force=True)
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=worker_process,
            args=(model_path, data, rank, world_size)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nAll tasks finish！")
    

def main():
    img_dir = "/input/"
    edit_dir = "/output/"

    data = PairDataset(img_dir=img_dir, edit_dir=edit_dir, max_size=10000, big_image_filter=True)
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