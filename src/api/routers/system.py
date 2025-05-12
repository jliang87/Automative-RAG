from fastapi import APIRouter, Depends
from src.core.background.actors.system import system_watchdog
from src.api.dependencies import get_token_header

router = APIRouter()

@router.post("/watchdog")
async def trigger_system_watchdog(token: str = Depends(get_token_header)):
    """Trigger the system watchdog task."""
    system_watchdog.send()
    return {"status": "watchdog_triggered"}