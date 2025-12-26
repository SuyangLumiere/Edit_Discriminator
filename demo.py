from Qwen3VLAuditor import Qwen3VLModel
def main():
    img_dir = "/input/0001.png"
    edit_dir = "/output/0001.png"

    # Initialize model
    model = Qwen3VLModel(model_path="/path/to/Qwen3-VL-8B-Instruct")

    # Audit an image pair
    res = model([img_dir, edit_dir]) # prompt can be defaulted. "Has this picture edited properly?"

    print(f"Success: {res.is_success} (Score: {res.score:.4f})")
    if not res.is_success:
        print(f"Refinement Suggestion: {res.refine_prompt}")


if __name__ is "__main__":
    main()