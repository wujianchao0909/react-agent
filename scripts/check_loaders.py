"""快速验证多格式加载器"""
import sys
from pathlib import Path

# 把项目根加到 sys.path，保证 import app 能找到
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.loaders import load_and_split


def main():
    docs_dir = Path("docs")
    for f in sorted(docs_dir.iterdir()):
        if not f.is_file():
            continue
        try:
            chunks = load_and_split(f)
            print(f"✅ {f.name}: {len(chunks)} chunks")
            for c in chunks[:2]:
                md = c.metadata
                print(
                    f"   [{md['file_type']}|page={md['page']}|"
                    f"sec={md['section']}] {c.page_content[:60]}..."
                )
        except Exception as e:
            print(f"❌ {f.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()