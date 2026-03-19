from Qwen3VLAuditor import Qwen3VLModel

def main():
    img_path = "/input/0001.png"
    edit_path = "/output/0001.png"
    
    edit_prompt = "Make the dog wear a red hat." 

    model = Qwen3VLModel(model_path="/path/to/Qwen3-VL-8B-Instruct")

    res = model(
        img_path=img_path, 
        edit_path=edit_path, 
        edit_prompt=edit_prompt, 
        review_mode=True # False will not give a comment
    )

    # or directly print(res))
    # print(res)
    print(f"Success: {res.is_success} (Score: {res.score:.4f})")
    print(f"Comment: {res.comment}")
    
    if not res.is_success:
        print(f"Refinement Suggestion: {res.refine_prompt}")

if __name__ == "__main__":
    main()