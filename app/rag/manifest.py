"""
索引清单：记录 docs/ 下每个文件的 hash、chunk 数、索引时间。
用于增量更新：只有变化的文件才重新 embedding。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict

class FileEntry(TypedDict):
    hash: str
    chunks: int
    indexed_at: str

class Manifest(TypedDict):
    version: int
    updated_at: str
    files: dict[str, FileEntry]

MANIFEST_VERSION = 1

def load_manifest(path: Path) -> Manifest:
    if not path.exists():
        return {"version": MANIFEST_VERSION, "updated_at": "", "files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != MANIFEST_VERSION:
            print("[manifest] 版本不匹配，重置清单")
            return {"version": MANIFEST_VERSION, "updated_at": "", "files": {}}
        return data
    except Exception as e:
        print(f"[manifest] 读取失败，重置：{e}")
        return {"version": MANIFEST_VERSION, "updated_at": "", "files": {}}

def save_manifest(path: Path, manifest: Manifest) -> None:
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
