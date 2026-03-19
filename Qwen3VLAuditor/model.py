import torch
from dataclasses import dataclass, asdict, field
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# result object
@dataclass
class VLResult:
    score: float = 0.0
    comment: str = ""
    refine_prompt: str = ""
    is_success: bool = False
    
    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return f"Success: {self.is_success} | Score: {self.score:.4f}\nPrompt: {self.comment}"


# qwen3VL
class Qwen3VLModel:
    def __init__(self, model_path: str, device: str = "auto", score_threshold: float = 1e-2):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, device_map=device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.score_threshold = score_threshold
        self.system_prompt = (
            "You are an expert AI Image Quality Auditor.\n"
            "Output Protocol:\n"
            "- Decision: Start exactly with 'Yes' or 'No'.\n"
            "- Reasoning: Provide a concise, professional critique of the edit based on the user's criteria.\n"
            "- Refinement: If 'No', provide a precise prompt to fix the issues, starting with 'RP'."
        )

    def __call__(
        self, 
        img_path: str, 
        edit_path: str, 
        edit_prompt: str, 
        review_mode: bool = False,
        criteria: str = """Evaluation Criteria:
                            1. **Instruction Following**: Does the edit in the 2nd image strictly follow the text prompt?
                            2. **Local Consistency**: Was the edit accurately applied to the area defined by the bounding box or any mark if there is?
                            3. **Global Preservation**: Are all non-target areas (background, other objects, textures) identical to the original image?""",
    ) -> VLResult:
        """
        router
        """
        if review_mode:
            return self.evaluate_full(img_path, edit_path, edit_prompt, criteria)
        else:
            return self.fast_score_only(img_path, edit_path, edit_prompt, criteria)

    @torch.no_grad()
    def fast_score_only(self, 
        img_path: str, 
        edit_path: str, 
        edit_prompt: str, 
        criteria: str = """Evaluation Criteria:
                            1. **Instruction Following**: Does the edit in the 2nd image strictly follow the text prompt?
                            2. **Local Consistency**: Was the edit accurately applied to the area defined by the bounding box or any mark if there is?
                            3. **Global Preservation**: Are all non-target areas (background, other objects, textures) identical to the original image?""",
    ):
        inputs = self._prepare_inputs(img_path, edit_path, edit_prompt, criteria)
        outputs = self.model(**inputs)
        next_token_logits = outputs.logits[:, -1, :] 

        score = self._calculate_confidence(next_token_logits)
        
        return VLResult(
            score=score, 
            is_success=(score > self.score_threshold)
        )
        
    @torch.no_grad()
    def evaluate_full(self, 
        img_path: str, 
        edit_path: str, 
        edit_prompt: str, 
        criteria: str = """Evaluation Criteria:
                            1. **Instruction Following**: Does the edit in the 2nd image strictly follow the text prompt?
                            2. **Local Consistency**: Was the edit accurately applied to the area defined by the bounding box or any mark if there is?
                            3. **Global Preservation**: Are all non-target areas (background, other objects, textures) identical to the original image?"""
    ):
        inputs = self._prepare_inputs(img_path, edit_path, edit_prompt, criteria)

        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_scores=True
        )

        # 1. parse text (Summary & Refine Prompt)
        generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
        raw_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        comment, refine_prompt = self._parse_response(raw_text)

        # 2. get first token Logits to figure Confidence Score
        first_token_logits = outputs.scores[0]
        score = self._calculate_confidence(first_token_logits)

        return VLResult(
            score=score,
            comment=comment,
            refine_prompt=refine_prompt,
            is_success=(score > self.score_threshold)
        )
    
    def _prepare_inputs(self, img_path: str, edit_path: str, edit_prompt: str, criteria: str):
        user_prompt = (
            f"The user attempted to edit the first image into the second image using the following instruction:\n"
            f"Command: \"{edit_prompt}\"\n\n"
            f"Evaluate the fidelity and alignment of the image edit based on the following criteria:\n"
            f"{criteria}\n\n"
            f"Has this picture been edited properly?"
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "image", "image": str(edit_path)},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        return self.processor.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).to(self.model.device)

    def _parse_response(self, text: str):
        parts = text.split("RP")
        comment = parts[0].strip()
        refine = parts[1].strip() if len(parts) > 1 else ""
        # pop res
        if comment.lower().startswith("yes"):
            comment = comment[3:].strip("- \n")
        elif comment.lower().startswith("no"):
            comment = comment[2:].strip("- \n")
        return comment, refine

    def _calculate_confidence(self, logits):
        # Confidence Score
        probs = torch.softmax(logits, dim=-1)[0]
        yes_ids = [self.tokenizer.encode(w)[0] for w in ["Yes", "yes", " Yes", " yes", "y", "Y"] if self.tokenizer.encode(w)]
        no_ids = [self.tokenizer.encode(w)[0] for w in ["No", "no", " No", " no", "n", "N"] if self.tokenizer.encode(w)]
        
        yes_prob = max([probs[tid].item() for tid in yes_ids]) if yes_ids else 1e-10
        no_prob = max([probs[tid].item() for tid in no_ids]) if no_ids else 1e-10
        
        # avoid denominator 0
        no_prob = max(no_prob, 1e-10)
        return (yes_prob - no_prob) / no_prob