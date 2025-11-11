import os
from huggingface_hub import HfApi, login
from pathlib import Path
import time

class HFCheckpointUploader:
    def __init__(self, repo_id, token, folder_path="Member 3"):
        """
        repo_id: e.g., "hossamaladdin/Assignment5"
        token: HuggingFace token
        folder_path: folder inside the repo to upload to
        """
        self.repo_id = repo_id
        self.folder_path = folder_path
        self.api = HfApi()
        
        # Authenticate
        try:
            login(token=token, add_to_git_credential=False)
            print(f"✓ Authenticated with HuggingFace")
            print(f"✓ Using dataset repo: {repo_id}")
            print(f"✓ Upload path: {folder_path}/checkpoints/")
        except Exception as e:
            print(f"⚠ Authentication warning: {e}")
    
    def upload_checkpoint(self, checkpoint_path):
        """Upload a single checkpoint file to dataset repo"""
        try:
            ckpt_file = Path(checkpoint_path)
            if not ckpt_file.exists():
                print(f"⚠ Checkpoint not found: {ckpt_file}")
                return False
            
            size_mb = ckpt_file.stat().st_size / (1024*1024)
            print(f"📤 Uploading {ckpt_file.name} ({size_mb:.1f} MB)...")
            
            # Upload to HuggingFace Dataset repo
            path_in_repo = f"{self.folder_path}/checkpoints/{ckpt_file.name}"
            
            self.api.upload_file(
                path_or_fileobj=str(ckpt_file),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="dataset"  # Important: it's a dataset repo
            )
            print(f"✓ Uploaded to {path_in_repo}")
            
            # Delete local copy to free space
            ckpt_file.unlink()
            print(f"✓ Deleted local copy (freed {size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"❌ Upload error for {checkpoint_path}: {e}")
            return False
    
    def upload_all_checkpoints(self, checkpoint_dir):
        """Upload all .pth.tar files in directory"""
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            print(f"⚠ Checkpoint directory not found: {ckpt_dir}")
            return
        
        checkpoints = sorted(ckpt_dir.glob("SeqTrackEpoch*.pth.tar"))
        if not checkpoints:
            print("No checkpoints found to upload")
            return
        
        print(f"\nFound {len(checkpoints)} checkpoint(s) to upload")
        for ckpt in checkpoints:
            self.upload_checkpoint(ckpt)
            time.sleep(1)  # Small delay between uploads
        
        print("\n✅ All checkpoints uploaded!")

