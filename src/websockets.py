from fastapi import WebSocket
from typing import Dict

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        self.active_connections[task_id] = websocket

    def disconnect(self, task_id: str):
        self.active_connections.pop(task_id, None)

    async def send_progress(self, task_id: str, message: str):
        if task_id in self.active_connections:
            await self.active_connections[task_id].send_text(message)

manager = ConnectionManager()