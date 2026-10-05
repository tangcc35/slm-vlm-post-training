import os
import re
from fastapi import APIRouter, HTTPException, Request
from story_rp_engine.core.types import Lorebook
from story_rp_engine.storage.store import _sanitize_key

router = APIRouter(prefix="/api/v1", tags=["Lorebooks"])


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower())
    return slug.strip("_") or "lorebook"


@router.post("/lorebooks")
def save_lorebook(lorebook: Lorebook, request: Request):
    store = request.app.state.store
    lb_id = _slugify(lorebook.name)
    try:
        store.save_lorebook(lb_id, lorebook)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "saved", "lorebook_id": lb_id}


@router.get("/lorebooks")
def list_lorebooks(request: Request):
    store = request.app.state.store
    return store.list_lorebooks()


@router.get("/lorebooks/{lorebook_id}")
def get_lorebook(lorebook_id: str, request: Request):
    store = request.app.state.store
    try:
        lb = store.get_lorebook(lorebook_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not lb:
        raise HTTPException(status_code=404, detail="Lorebook not found")
    return lb


@router.delete("/lorebooks/{lorebook_id}")
def delete_lorebook(lorebook_id: str, request: Request):
    store = request.app.state.store
    try:
        clean_id = _sanitize_key(lorebook_id)
        path = os.path.join(store.lorebooks_dir, f"{clean_id}.json")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Lorebook not found")
        os.remove(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "deleted", "lorebook_id": clean_id}
