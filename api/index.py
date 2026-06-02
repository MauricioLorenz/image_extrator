from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI(title="Image Metadata Extractor")


@app.post("/metadata")
async def extract_metadata(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="No image data received")

    try:
        img = Image.open(io.BytesIO(body))
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid or unsupported image format")

    width, height = img.size
    fmt = img.format or "UNKNOWN"
    mode = img.mode

    dpi_raw = img.info.get("dpi")
    if dpi_raw:
        dpi_x = round(dpi_raw[0], 2)
        dpi_y = round(dpi_raw[1], 2)
    else:
        dpi_x = None
        dpi_y = None

    return JSONResponse({
        "format": fmt,
        "mode": mode,
        "width_px": width,
        "height_px": height,
        "dpi_x": dpi_x,
        "dpi_y": dpi_y,
        "size_bytes": len(body),
    })
