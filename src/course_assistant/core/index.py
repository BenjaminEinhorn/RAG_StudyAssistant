"""Catalog: the hybrid-RAG index.

Keeps:
- text chunks in Chroma (vector) *and* BM25 (keyword),
- slide images in a separate Chroma "visual" collection,
- a JSONL registry of every chunk (source of truth for removal/rebuild),
- re-ranking of keyword+vector candidates.

Document add/remove keeps text and visual indexes separate, and removal drops
the document's chunks AND images so later answers never rely on removed files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config.settings import Settings
from ..services.embeddings import EmbeddingProvider
from ..services.reranker import Reranker
from .chunking import Chunk, chunk_document
from .parsing import ParsedDoc, parse_file


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    text: str
    image_path: str | None
    score: float


@dataclass
class RetrievedSlide:
    doc_id: str
    doc_name: str
    page: int
    image_path: str
    text: str
    score: float


@dataclass
class Catalog:
    settings: Settings
    text_embedder: EmbeddingProvider
    visual_embedder: EmbeddingProvider
    reranker: Reranker
    chunk_method: str = "recursive"
    _chunks: dict = field(default_factory=dict)   # chunk_id -> Chunk
    _bm25: object = None
    _chroma: object = None
    _text_col: object = None
    _visual_col: object = None

    def __post_init__(self):
        import chromadb

        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        self._text_col = self._chroma.get_or_create_collection(
            "text", metadata={"hnsw:space": "cosine"})
        self._visual_col = self._chroma.get_or_create_collection(
            "visual", metadata={"hnsw:space": "cosine"})
        self._load_registry()
        self._build_bm25()

    # ------------------------------------------------------------ persistence
    def _chunk_id(self, c: Chunk) -> str:
        return f"{c.doc_id}:{c.page}:{c.chunk_index}"

    def _load_registry(self):
        p = self.settings.text_meta
        if p.exists():
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                self._chunks[d["chunk_id"]] = Chunk(**d["chunk"])

    def _save_registry(self):
        lines = [
            json.dumps({"chunk_id": cid, "chunk": {
                "doc_id": c.doc_id, "doc_name": c.doc_name, "page": c.page,
                "text": c.text, "image_path": c.image_path,
                "chunk_index": c.chunk_index}})
            for cid, c in self._chunks.items()
        ]
        self.settings.text_meta.write_text("\n".join(lines) + "\n")

    def _build_bm25(self):
        import bm25s

        self._text_list = [c.text for c in self._chunks.values()]
        self._chunk_ids = list(self._chunks.keys())
        if self._text_list:
            tokens = bm25s.tokenize(self._text_list)
            bm = bm25s.BM25()
            bm.index(tokens)
            self._bm25 = bm
        else:
            self._bm25 = None

    # ------------------------------------------------------------ documents
    def list_documents(self) -> list[dict]:
        by_doc: dict[str, dict] = {}
        for c in self._chunks.values():
            d = by_doc.setdefault(c.doc_id, {
                "doc_id": c.doc_id, "doc_name": c.doc_name,
                "chunks": 0, "slides": set()})
            d["chunks"] += 1
            d["slides"].add(c.page)
        out = []
        for d in by_doc.values():
            d["slide_count"] = len(d["slides"])
            d.pop("slides", None)
            out.append(d)
        out.sort(key=lambda d: d["doc_name"])
        return out

    def doc_exists(self, sha: str) -> bool:
        return any(c.doc_id == sha[:16] for c in self._chunks.values())

    def add_file(self, path: str):
        """Parse, dedupe, embed and index a user-uploaded file. Records only
        supported types; returns a message + the doc (or 'duplicate')."""
        from ..core.parsing import sha256_of

        src = Path(path)
        if src.suffix.lower() not in {".pdf", ".pptx", ".docx", ".txt", ".md"}:
            return {"ok": False, "message": (
                f"Unsupported type '{src.suffix}'. Supported: pdf, pptx, "
                "docx, txt, md "
                "(pptx converted via installed LibreOffice).")}
        sha = sha256_of(src)
        if self.doc_exists(sha):
            return {"ok": True, "message": f"'{src.name}' was already loaded — skipped.", "duplicate": True}

        parsed = parse_file(src, self.settings)
        chunks = chunk_document(parsed, self.settings, self.chunk_method)
        doc_id = parsed.sha256[:16]

        # embed + index text chunks
        if chunks:
            texts = [c.text for c in chunks]
            vecs = self.text_embedder.embed_texts(texts)
            ids = [self._chunk_id(c) for c in chunks]
            metas = [{
                "doc_id": c.doc_id, "doc_name": c.doc_name, "page": c.page,
                "text": c.text, "image_path": c.image_path or "",
                "chunk_index": c.chunk_index,
            } for c in chunks]
            self._text_col.upsert(ids=ids, embeddings=vecs, metadatas=metas,
                                  documents=texts)
            for cid, c in zip(ids, chunks):
                self._chunks[cid] = c
        # embed + index slide images (visual index)
        imgs = [pg.image_path for pg in parsed.pages if pg.image_path]
        n_images = 0
        if imgs:
            ivecs = self.visual_embedder.embed_images(imgs)
            iids = [f"{doc_id}:{pg.page_num}" for pg in parsed.pages if pg.image_path]
            imetas = [{
                "doc_id": doc_id, "doc_name": parsed.name,
                "page": pg.page_num, "image_path": pg.image_path,
                "text": pg.text[:2000],
            } for pg in parsed.pages if pg.image_path]
            self._visual_col.upsert(ids=iids, embeddings=ivecs, metadatas=imetas)
            n_images = len(imgs)
        self._save_registry()
        self._build_bm25()
        return {"ok": True, "message": f"Added '{src.name}'", "doc_id": doc_id,
                "chunks": len(chunks), "slides": parsed.slide_count(),
                "images": n_images}

    def remove_document(self, doc_name: str):
        """Remove a document (by name) and ALL its indexed content/images."""
        doc_ids = {c.doc_id for c in self._chunks.values() if c.doc_name == doc_name}
        if not doc_ids:
            return {"ok": False, "message": f"No document named '{doc_name}'."}
        for did in doc_ids:
            self._text_col.delete(where={"doc_id": did})
            self._visual_col.delete(where={"doc_id": did})
            self._chunks = {cid: c for cid, c in self._chunks.items() if c.doc_id != did}
        self._save_registry()
        self._build_bm25()
        return {"ok": True, "message": f"Removed '{doc_name}' and its searchable content."}

    # ------------------------------------------------------------ retrieval
    def _merge_candidates(self, query: str, k: int) -> list[RetrievedChunk]:
        ids_to_score: dict[str, float] = {}
        # keyword
        if self._bm25:
            q = __import__("bm25s", fromlist=["tokenize"]).tokenize([query])
            nk = min(max(6, k), len(self._text_list))  # bm25s: k <= corpus size
            results, scores = self._bm25.retrieve(q, k=nk)
            for j, idx in enumerate(results[0]):
                if idx < len(self._chunk_ids):
                    cid = self._chunk_ids[int(idx)]
                    ids_to_score[cid] = max(ids_to_score.get(cid, 0), float(scores[0][j]))
        # vector (text)
        qv = self.text_embedder.embed_texts([query])[0]
        nvec = min(max(6, k), max(1, len(self._chunks)))
        res = self._text_col.query(query_embeddings=[qv], n_results=nvec)
        for i, row in enumerate(res.get("ids", [[]])[0]):
            meta = res["metadatas"][0][i]
            ids_to_score[row] = max(ids_to_score.get(row, 0), 1.0 - i * 0.05)
        # build candidate objects with combined keyword+vector score
        cands = []
        for cid, base in ids_to_score.items():
            c = self._chunks.get(cid)
            if c is None:
                continue
            cands.append(RetrievedChunk(
                chunk_id=cid, doc_id=c.doc_id, doc_name=c.doc_name,
                page=c.page, text=c.text, image_path=c.image_path, score=base))
        if not cands:
            return []
        # rerank
        scores = self.reranker.rescore(query, [c.text for c in cands])
        for c, s in zip(cands, scores):
            c.score = s
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[:k]

    def search(self, query: str, k: int = 5,
               doc_names: list[str] | None = None) -> list[RetrievedChunk]:
        results = self._merge_candidates(query, k)
        if doc_names:
            names = set(doc_names)
            results = [c for c in results if c.doc_name in names]
        # if filtering removed everything, allow a wider net within the filter
        if not results and doc_names:
            cands = self._merge_candidates(query, max(12, k * 2))
            names = set(doc_names)
            results = [c for c in cands if c.doc_name in names][:k]
        return results

    def visual_search(self, query: str, k: int = 3) -> list[RetrievedSlide]:
        qv = self.visual_embedder.embed_texts([query])[0]
        try:
            res = self._visual_col.query(query_embeddings=[qv], n_results=max(1, k))
        except Exception:
            return []
        out = []
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            out.append(RetrievedSlide(
                doc_id=meta["doc_id"], doc_name=meta["doc_name"],
                page=meta["page"], image_path=meta["image_path"],
                text=meta.get("text", ""), score=float(dist)))
        return out
