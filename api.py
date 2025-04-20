from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.requests import Request
import cv2
import numpy as np
import tempfile
import os
from vortex import detect_vortices_by_convolution
import pathlib
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
from pydantic import BaseModel
import multipart


app = FastAPI(debug=True)
templates=Jinja2Templates(directory=f"{pathlib.Path.cwd()}/templates/")
staticfiles=StaticFiles(directory=f"{pathlib.Path.cwd()}/static/")
app.mount("/static",staticfiles,name="static")

@app.post("/detect_vortices")
async def detect_vortices(
    file: UploadFile = File(...),
    min_radius: int = Form(2),
    max_radius: int = Form(7),
    color_threshold: float = Form(0.5),
    split: float = Form(0.7),
    more_precise: int = Form(3),
    erosion: int = Form(0),
    inverse: bool = Form(False),
    local_contrast: bool = Form(False),
    watershad: bool = Form(False)
):
    try:
        # 保存上传的临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
            temp.write(await file.read())
            temp_path = temp.name
        
        # 调用原函数
        vortices = detect_vortices_by_convolution(
            temp_path,
            min_radius=min_radius,
            max_radius=max_radius,
            color_threshold=color_threshold,
            split=split,
            more_precise=more_precise,
            erosion=erosion,
            inverse=inverse,
            local_contrast=local_contrast,
            watershad=watershad
        )
        
        # 读取并处理图像
        image = cv2.imread(temp_path)
        for (x, y) in vortices:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        # 将处理后的图像转换为base64
        _, buffer = cv2.imencode('.jpg', image)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        # 删除临时文件
        os.unlink(temp_path)
        
        return JSONResponse({
            "processed_image": processed_image,
            "vortex_count": len(vortices)
        })
    except cv2.error as e:
        raise HTTPException(status_code=400, detail=f"OpenCV处理错误: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"参数值错误: {str(e)}")
    except IOError as e:
        raise HTTPException(status_code=400, detail=f"文件IO错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.get("/",response_class=HTMLResponse)
async def get_response(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001)