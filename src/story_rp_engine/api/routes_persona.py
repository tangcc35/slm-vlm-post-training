from fastapi import APIRouter, HTTPException, Request
from story_rp_engine.core.types import Persona
from story_rp_engine.storage.store import _sanitize_key

router = APIRouter(prefix="/api/v1", tags=["Personas"])


@router.post("/personas")
async def save_persona(persona: Persona, request: Request):
    try:
        await request.app.state.store.save_persona(persona.persona_id, persona)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "saved", "persona_id": persona.persona_id}


@router.get("/personas")
async def list_personas(request: Request):
    return await request.app.state.store.list_personas()


@router.get("/personas/{persona_id}")
async def get_persona(persona_id: str, request: Request):
    try:
        persona = await request.app.state.store.get_persona(persona_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona


@router.delete("/personas/{persona_id}")
async def delete_persona(persona_id: str, request: Request):
    try:
        clean_id = _sanitize_key(persona_id)
        deleted = await request.app.state.store.delete_persona(clean_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail="Persona not found")
    return {"status": "deleted", "persona_id": clean_id}
