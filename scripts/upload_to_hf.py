"""Upload the dataset and the paper (3-class) weights to a private Hugging Face dataset repo.

Run once from the Mac after `hf auth login`:
    python scripts/upload_to_hf.py

Re-running is safe: files already on the Hub are not uploaded again.
"""
import argparse

from huggingface_hub import HfApi

from dmgseg import paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=paths.HF_REPO)
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)

    print(f"Uploading {paths.CVAT_DIR} -> Dataset_CVAT/")
    api.upload_folder(
        folder_path=str(paths.CVAT_DIR),
        path_in_repo="Dataset_CVAT",
        repo_id=args.repo,
        repo_type="dataset",
        ignore_patterns=[".DS_Store", "**/.DS_Store"],
    )

    print(f"Uploading {paths.PRIOR_WEIGHTS.name} -> weights/paper_dinov2_emb_3class.pth")
    api.upload_file(
        path_or_fileobj=str(paths.PRIOR_WEIGHTS),
        path_in_repo="weights/paper_dinov2_emb_3class.pth",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print("Done.")


if __name__ == "__main__":
    main()
