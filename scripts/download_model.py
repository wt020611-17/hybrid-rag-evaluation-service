import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the pinned BGE model before Docker build")
    parser.add_argument("--output", type=Path, default=Path("models/bge-small-zh-v1.5"))
    parser.add_argument("--endpoint", default=os.getenv("HF_ENDPOINT", "https://huggingface.co"))
    args = parser.parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id="BAAI/bge-small-zh-v1.5",
        local_dir=str(args.output.resolve()),
        allow_patterns=["*.json", "*.txt", "model.safetensors"],
    )
    print("model={}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
