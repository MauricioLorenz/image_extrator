import asyncio
import io
import logging
import os
import traceback

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

MAX_BODY_SIZE = 4 * 1024 * 1024  # Vercel's own request body limit is ~4.5 MB
HEADER_FETCH_BYTES = 2 * 1024 * 1024  # margin for EXIF/ICC data before the header we need

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


def _blob_auth_headers() -> dict:
    token = os.environ.get("BLOB_READ_WRITE_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _fetch_blob_header(url: str) -> bytes:
    headers = {**_blob_auth_headers(), "Range": f"bytes=0-{HEADER_FETCH_BYTES - 1}"}
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.content


async def _fetch_blob_full(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=25) as client:
        resp = await client.get(url, headers=_blob_auth_headers())
        resp.raise_for_status()
        return resp.content


async def _delete_blob(url: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.delete(url, headers=_blob_auth_headers())
    except Exception:
        logger.warning("Failed to delete blob %s", url, exc_info=True)


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
    content_type = request.headers.get("content-type", "")
    loop = asyncio.get_event_loop()

    if content_type.startswith("application/json"):
        # Large files: caller uploads directly to Blob storage (bypassing this
        # function's ~4.5 MB body limit) and gives us just the URL here.
        payload = await request.json()
        blob_url = payload.get("url")
        if not blob_url:
            raise HTTPException(status_code=400, detail="No image data received")

        try:
            body = await _fetch_blob_header(blob_url)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch blob (status {exc.response.status_code}): {exc.response.text[:200]}",
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch blob: {exc}")

        try:
            result = await loop.run_in_executor(None, _extract, body)
        except Exception:
            # Header-only fetch wasn't enough (e.g. TIFF keeps its IFD at the
            # end of the file) — retry with the full object before giving up.
            try:
                body = await _fetch_blob_full(blob_url)
                result = await loop.run_in_executor(None, _extract, body)
            except Exception as exc:
                await _delete_blob(blob_url)
                # Temporary debug fields to diagnose the blob-fetch path — remove once resolved.
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Invalid or unsupported image format",
                        "debug_exception": str(exc),
                        "debug_size_bytes": len(body),
                        "debug_first_bytes_hex": body[:32].hex(),
                        "debug_first_bytes_text": body[:200].decode("utf-8", errors="replace"),
                    },
                )
        await _delete_blob(blob_url)
        return JSONResponse(result)

    body = await _read_upload(request)
    if not body:
        raise HTTPException(status_code=400, detail="No image data received")
    if len(body) > MAX_BODY_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image too large (max 4 MB) — upload to Blob storage first and send its URL instead",
        )

    try:
        result = await loop.run_in_executor(None, _extract, body)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid or unsupported image format")

    return JSONResponse(result)
