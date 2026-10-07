"""重建 / 增量同步知识库索引"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.indexer import sync_index

if __name__ == "__main__":
    force = "--force" in sys.argv
    if force:
        print("⚠️ 全量重建模式，将清空 manifest")
    sync_index(force=force)