"""
增量索引：对比 docs/ 和 manifest，只处理变化的文件。
"""
from pathlib import Path

from app.config import settings
from app.rag.loaders import is_supported, load_and_split
from app.rag.loaders.common import compute_file_hash
from app.rag.manifest import load_manifest, save_manifest, Manifest
from app.rag.vectorstore import get_vectorstore

def list_docs_files(docs_dir: Path) -> list[Path]:
    return [
        f for f in sorted(docs_dir.iterdir())
        if f.is_file() and is_supported(f)
    ]

def rel_posix(path: Path, docs_dir: Path) -> str:
    return path.resolve().relative_to(docs_dir.resolve()).as_posix()

def delete_file_chunks(vs, rel_path: str) -> int:
    """从 Chroma 删除某个文件的全部 chunk。返回删除数量。"""
    try:
        existing = vs.get(where={"file_path": rel_path})
        ids = existing.get("ids", [])
        if ids:
            vs.delete(ids=ids)
        return len(ids)
    except Exception as e:
        print(f"[indexer] 删除 {rel_path} 失败：{e}")
        return 0

def sync_index(force: bool = False) -> dict:
    """
    增量同步 docs/ 到向量库。
    force=True 时忽略 manifest，全量重建。
    """
    docs_dir = Path(settings.docs_dir).resolve()
    manifest_path = Path(settings.manifest_path).resolve()

    if not docs_dir.exists():
        print(f"[indexer] docs 目录不存在：{docs_dir}")
        return {"added": 0, "updated": 0, "skipped": 0, "deleted": 0, "total_chunks": 0}

    if force:
        manifest: Manifest = {"version": 1, "updated_at": "", "files": {}}
        save_manifest(manifest_path, manifest)
    else:
        manifest = load_manifest(manifest_path)

    vs = get_vectorstore()
    files = list_docs_files(docs_dir)
    current_rel_paths = {rel_posix(f, docs_dir) for f in files}

    stats = {"added": 0, "updated": 0, "skipped": 0, "deleted": 0, "total_chunks": 0}

    for f in files:
        rel_path = rel_posix(f, docs_dir)
        file_hash = compute_file_hash(f)
        old = manifest["files"].get(rel_path)

        if old and old["hash"] == file_hash:
            stats["skipped"] += 1
            stats["total_chunks"] += old["chunks"]
            continue

        if old:
            delete_file_chunks(vs, rel_path)
            stats["updated"] += 1
        else:
            stats["added"] += 1

        try:
            chunks = load_and_split(f, docs_dir)
        except Exception as e:
            print(f"[indexer] 加载 {rel_path} 失败：{e}")
            continue

        if not chunks:
            print(f"[indexer] {rel_path} 无有效内容，跳过")
            continue

        vs.add_documents(chunks)
        manifest["files"][rel_path] = {
            "hash": file_hash,
            "chunks": len(chunks),
            "indexed_at": ""
        }
        stats["total_chunks"] += len(chunks)
        print(f"[indexer] {'更新' if old else '新增'} {rel_path}: {len(chunks)} chunks")

    for rel_path in list(manifest["files"].keys()):
        if rel_path not in current_rel_paths:
            delete_file_chunks(vs, rel_path)
            del manifest["files"][rel_path]
            stats["deleted"] += 1
            print(f"[indexer] 删除 {rel_path}")

    save_manifest(manifest_path, manifest)
    print(
        f"[indexer] 同步完成："
        f"新增 {stats['added']}，更新 {stats['updated']}，"
        f"跳过 {stats['skipped']}，删除 {stats['deleted']}，"
        f"当前共 {stats['total_chunks']} chunks"
    )
    return stats