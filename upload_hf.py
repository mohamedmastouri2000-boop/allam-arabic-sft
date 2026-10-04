"""Create the PRIVATE Hugging Face repo and upload merged weights, GGUF quants and the card.

Visibility is private and this script never changes it. Making the repo public is a separate,
explicit step the owner asks for.
"""
import os
from huggingface_hub import HfApi

NAME = os.environ.get("MODEL_NAME", "ALLaM-7B-Arabic-SFT")
REPO = f"mastouri/{NAME}"
api = HfApi()  # token from ~/.cache/huggingface/token
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
info = api.repo_info(REPO)
assert info.private, f"{REPO} exists and is PUBLIC; refusing to upload over it"

api.upload_folder(repo_id=REPO, folder_path=r"D:\arabic8b\out\merged",
                  commit_message="Merged bf16 weights (LoRA merged into ALLaM-7B-Instruct-preview)")
api.upload_folder(repo_id=REPO, folder_path=r"D:\arabic8b\out\gguf", allow_patterns=["*.gguf"],
                  commit_message="GGUF quants: F16, Q8_0, Q5_K_M, Q4_K_M")
api.upload_file(repo_id=REPO, path_or_fileobj=r"D:\arabic8b\out\README.md", path_in_repo="README.md",
                commit_message="Model card with measured benchmarks, quant stats and samples")
print("uploaded to", f"https://huggingface.co/{REPO}", "| private:", api.repo_info(REPO).private)
