from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio
import io

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB
executor: ThreadPoolExecutor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor
    executor = ThreadPoolExecutor(max_workers=4)
    yield
    executor.shutdown(wait=False)


app = FastAPI(title="Image Metadata Extractor", lifespan=lifespan)


def _extract(body: bytes) -> dict:
    img = Image.open(io.BytesIO(body))
    img.load()
    dpi_raw = img.info.get("dpi")
    return {
        "format": img.format or "UNKNOWN",
        "mode": img.mode,
        "width_px": img.size[0],
        "height_px": img.size[1],
        "dpi_x": round(dpi_raw[0], 2) if dpi_raw else None,
        "dpi_y": round(dpi_raw[1], 2) if dpi_raw else None,
        "size_bytes": len(body),
    }


@app.post("/metadata")
async def extract_metadata(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="No image data received")
    if len(body) > MAX_BODY_SIZE:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, _extract, body)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid or unsupported image format")

    return JSONResponse(result)
