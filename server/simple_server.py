import os
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Phlox")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
BUILD_DIR = Path("../build")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Simple Phlox server is running"}

# React app routes
@app.get("/new-patient")
@app.get("/settings")
@app.get("/rag")
@app.get("/clinic-summary")
@app.get("/outstanding-tasks")
async def serve_react_app():
    return FileResponse(BUILD_DIR / "index.html")

# Serve static files
app.mount("/static", StaticFiles(directory=BUILD_DIR / "static"), name="static")

# Catch-all route for any other paths
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    # If the path is a file that exists, serve it
    file_path = BUILD_DIR / full_path
    if file_path.is_file():
        return FileResponse(file_path)
    # Otherwise serve the React app
    return FileResponse(BUILD_DIR / "index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Phlox server on port {port}")
    print(f"React build directory: {BUILD_DIR.resolve()}")
    uvicorn.run(app, host="0.0.0.0", port=port) 