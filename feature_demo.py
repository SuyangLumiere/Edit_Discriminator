import os
import json
from Qwen3VLAuditor import FlexiblePairDataset, build_pair_list

def create_mock_resume_file(filepath: str):
    """just for demo"""
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            f.write(json.dumps({"edit": "/output/edit_000.png", "score": 0.9}) + "\n")
            f.write(json.dumps({"edit": "/output/edit_002.png", "score": -0.5}) + "\n")

def main():
    img_paths = [f"/input/image_{i:03d}.png" for i in range(10)]
    edit_paths = [f"/output/edit_{i:03d}.png" for i in range(10)]
    edit_prompts = ["Make it look better."] * 10

    raw_pairs = build_pair_list(img_paths, edit_paths, edit_prompts)

    resume_file = "results_rank0.jsonl"
    create_mock_resume_file(resume_file) 

    print("--- Init Dataset ---")
    dataset = FlexiblePairDataset(
        data=raw_pairs, 
        resume_path=resume_file
    ).split(rank=0, world_size=2)


    print(f"\n--- start itering (this thread finally is assigned {len(dataset)} tasks) ---")
    for img, edit, prompt in dataset:
        print(f"model -> Img: {img} | Edit: {edit} | Prompt: {prompt}")
        # res = model(img, edit, prompt)

if __name__ == "__main__":
    main()