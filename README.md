# 🔍 Edit Discriminator
Edit Discriminator is an automated Quality Assurance (QA) toolkit for AIGC image editing tasks. Powered by the Qwen3-VL multimodal model, it evaluates instruction following, local consistency, and global preservation, providing both a confidence score and a Refinement Prompt for failed edits.

---

## 🌟 Key Features
Automated Audit: Replaces manual inspection by judging whether an edit strictly follows the user instruction.

Confidence Scoring: Uses logit-based relative difference between "Yes" and "No" tokens to provide a reliable quality score.

Refinement Loop: Automatically generates a precise "Refinement Prompt" to guide the diffusion model in fixing errors.

Dataset Ready: Built-in support for processing paired directories (/input and /output) and logging results to JSONL.

---

## 🚀 Quick Start

**Installation**

- Python 3.10

1. Clone the repository:
   ```bash
   git clone https://github.com/SuyangLumiere/EditDiscriminator.git
   cd Edit_Discriminator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. update key packages:  
   
   ```bash
   pip install --upgrade transformers accelerate
   ```

**Basic Usage**

Use the Qwen3VLModel to perform a quick check on a single image pair:

    ```python
    python demo.py
    ```

**Batch Processing**

Perfect for cleaning large-scale synthetic datasets:

    ```python
    python batch_process_demo.py
    ```

---

## 🛠️ Internal Logic
1. Scoring Mechanism

The auditor doesn't just give a "Yes" or "No". It calculates the confidence score based on the model's logits:

$$
Score= \frac{P(Yes)−P(No)}{P(No)}
$$​	
 
This allows you to set custom thresholds for data filtering.

2. Refinement Protocol

When an edit fails, the model identifies specific issues (e.g., "color mismatch", "broken textures") and outputs a prompt starting with ROP to be fed back into your generation pipeline.

## 📂 Project Structure

```
Edit Discriminator/
├── Qwen3VLAuditor/          # Core package
│   ├── __init__.py         # Interface exports
│   ├── model.py            # Model & Result logic
│   ├── data.py             # Data iteration
│   └── utils.py            # Logging & Helpers
├── demo.py                 # Single inference demo
├── batch_process_demo.py    # Batch processing demo
└── requirements.txt        # Dependencies
```

---

## 🤝 Contributing
Feel free to open issues or submit PRs if you have ideas for better system prompts or scoring algorithms!

---

## Star History

If you find this project helpful or interesting, a star would be greatly appreciated! Your support motivates us to keep improving. ⭐


[![Star History Chart](https://api.star-history.com/svg?repos=SuyangLumiere/Edit_Discriminator&type=date&legend=top-left)](https://www.star-history.com/#SuyangLumiere/Edit_Discriminator&type=date&legend=top-left)
