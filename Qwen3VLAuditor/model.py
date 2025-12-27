import torch
from dataclasses import dataclass, asdict
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# result object
@dataclass
class VLResult:
    score: float = 0.0
    raw_response: str = ""
    flag: str = "ROP"

    @property
    def is_success(self) -> bool:
        return  self.score > 1e-2

    @property
    def refine_prompt(self) -> str:
        if self.is_success:
            return ""
        parts = self.raw_response.split(self.flag)
        return parts[-1].strip() if len(parts) > 1 else ""
    
    @property
    def comment(self) -> str:
        parts = self.raw_response.split(self.flag)
        return parts[0].strip()
    
    def to_dict(self):
        # data = asdict(self)
        data = {}
        data.update({
            "score":self.score,
            "is_success": self.is_success,
            "comment": self.comment,
            "refine_prompt": self.refine_prompt
        })
        return data

    def __str__(self):
        return f"Success: {self.is_success} | Score: {self.score:.4f}\nPrompt: {self.comment}"


# qwen3VL
class Qwen3VLModel:
    def __init__(self, model_path: str):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.messages = [{"role": "system", "content": [{"type": "text", "text": """You are an expert AI Image Quality Auditor. Your task is to evaluate the fidelity and alignment of image edits.
                                                                        ### Evaluation Criteria:
                                                                        1. **Instruction Following**: Does the edit in the 2nd image strictly follow the text prompt?
                                                                        2. **Local Consistency**: Was the edit accurately applied to the area defined by the bounding box or any mark if there is?
                                                                        3. **Global Preservation**: Are all non-target areas (background, other objects, textures) identical to the original image?

                                                                        ### Output Protocol:
                                                                        - **Decision**: Start your response with a clear "Yes" (if the edit is perfect) or "No" (if it fails any criteria).
                                                                        - **Reasoning**: Provide a concise, professional critique regarding the criteria above.
                                                                        - **Refinement (ROP)**: 
                                                                            - If "No": Provide a precise, descriptive prompt to fix the issues, starting with "ROP ".
                                                                            - If "Yes": Provide only the symbol "ROP" with no additional refine prompt.

                                                                        Maintain a professional, objective tone."""}]},]
        

    def __call__(self, img_pair, user_prompt: str = "Has this picture edited properly?"):
        img_pair = [str(p) for p in img_pair]
        inputs = self._prepare_inputs(img_pair, user_prompt)
        raw_text = self._generate_text(inputs)
        score_val = self._calculate_score(inputs)
        
        return VLResult(raw_response=raw_text[0], score=score_val)

    def _prepare_inputs(self, img_pair, user_prompt):
        messages = self.messages.copy()
        messages.append(
            {"role": "user", "content": [
                {"type": "image", "image": img_pair[0]},
                {"type": "image", "image": img_pair[1]},
                {"type": "text", "text": user_prompt}
            ]}
        )
        return self.processor.apply_chat_template(messages, return_tensors="pt", return_dict=True, tokenize=True, add_generation_prompt=True).to(self.model.device)
    
    def _generate_text(self, inputs):
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        return output_text
    
    def _calculate_score(self, inputs):
        with torch.no_grad():
            output = self.model(**inputs, max_new_tokens=256)
            logits = output.logits[:, -1, :]
        prob = Qwen3VLModel._get_word_probability(logits, self.tokenizer, ["yes","no"])
        score = (prob["yes"] - prob["no"]) / prob["no"]
        print(f"Probability Yes: {prob['yes']}, Probability No: {prob['no']}")
        return score

    @staticmethod
    def _get_word_probability(logits, tokenizer, words, probs_softmax=False):
        probs = torch.softmax(logits, dim=-1) if probs_softmax else logits
        res = {}
        for word in words:
            candidate = []
            for i in [word, word.capitalize(), " " + word, " " + word.capitalize()]:
                token_id = tokenizer.convert_tokens_to_ids(i)
                if token_id != tokenizer.unk_token_id:
                    candidate.append(probs[0, token_id].item())
            res[word] = max(candidate) if candidate else 1e-10
        return res