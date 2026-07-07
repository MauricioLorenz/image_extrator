import asyncio
import io
import logging
import traceback

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

MAX_BODY_SIZE = 4 * 1024 * 1024  # Vercel's own request body limit is ~4.5 MB

logger = logging.getLogger("image_extrator")

app = FastAPI(title="Image Metadata Extractor")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url.path, traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _extract(body: bytes) -> dict:
    # Metadata only needs the header, so we deliberately skip img.load() (full pixel
    # decode) — also means Pillow's decompression-bomb guard never applies here,
    # which is fine since pixel data is never touched.
    img = Image.open(io.BytesIO(body))
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


async def _read_upload(request: Request) -> bytes:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        # n8n's "n8n Binary File" body option wraps the file in a multipart
        # envelope instead of sending raw bytes — unwrap it here.
        form = await request.form()
        for value in form.values():
            if hasattr(value, "read"):
                return await value.read()
        return b""
    return await request.body()


@app.post("/metadata")
async def extract_metadata(request: Request):
    body = await _read_upload(request)
    if not body:
        raise HTTPException(status_code=400, detail="No image data received")
    if len(body) > MAX_BODY_SIZE:
        raise HTTPException(status_code=413, detail="Image too large (max 4 MB)")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _extract, body)
    except Exception as exc:
        # Temporary debug fields (error + first bytes) to diagnose a specific
        # n8n integration issue — remove once resolved.
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Invalid or unsupported image format",
                "debug_exception": str(exc),
                "debug_first_bytes_hex": body[:16].hex(),
                "debug_size_bytes": len(body),
            },
        )

    return JSONResponse(result)
