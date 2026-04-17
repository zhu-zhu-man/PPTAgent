"""Web后端入口模块：提供任务API与端到端PPT生成异步编排流程。"""

import asyncio
import hashlib
import importlib
import json
import os
import sys
import traceback
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
sys.path.append('..')
import pptagent.induct as induct
import pptagent.pptgen as pptgen
from pptagent.document import Document
from pptagent.model_utils import ModelManager, parse_pdf
from pptagent.multimodal import ImageLabler
from pptagent.presentation import Presentation
from pptagent.utils import Config, get_logger, package_join, pjoin, ppt_to_images_async

# constants
DEBUG = True if len(sys.argv) == 1 else False
RUNS_DIR = os.path.abspath('..') + "/runs" # /root/autodl-tmp/PPTAgent/runs
STAGES = [
    "PPT Parsing",
    "PDF Parsing",
    "PPT Analysis",
    "PPT Generation",
    "Success!",
]


models = ModelManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    assert await models.test_connections(), "Model connection test failed"
    yield


# server
logger = get_logger(__name__)
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
progress_store: dict[str, dict] = {}
active_connections: dict[str, WebSocket] = {}


class ProgressManager:
    def __init__(self, task_id: str, stages: list[str], debug: bool = True):
        self.task_id = task_id
        self.stages = stages
        self.debug = debug
        self.task_id = task_id
        self.failed = False
        self.current_stage = 0
        self.total_stages = len(stages)

    async def report_progress(self):
        assert (
            self.task_id in active_connections
        ), "WebSocket connection is already closed"
        self.current_stage += 1
        progress = int((self.current_stage / self.total_stages) * 100)
        await send_progress(
            active_connections[self.task_id],
            f"Stage: {self.stages[self.current_stage - 1]}",
            progress,
        )

    async def fail_stage(self, error_message: str):
        await send_progress(
            active_connections[self.task_id],
            f"{self.stages[self.current_stage]} Error: {error_message}",
            100,
        )
        self.failed = True
        active_connections.pop(self.task_id, None)
        if self.debug:
            logger.error(
                f"{self.task_id}: {self.stages[self.current_stage]} Error: {error_message}"
            )


@app.post("/api/upload")
async def create_task(
    pptxFile: UploadFile = File(None),
    pdfFile: UploadFile = File(None),
    topic: str = Form(None),
    numberOfPages: int = Form(...),
):
    # 1) 生成“日期+UUID”唯一任务ID
    task_id = datetime.now().strftime("20%y-%m-%d") + "/" + str(uuid.uuid4())
    # 2) 记录任务创建日志
    logger.info(f"task created: {task_id}")
    # 3) 在 runs/ 下创建本次任务目录
    os.makedirs(pjoin(RUNS_DIR, task_id))
    # 4) 初始化任务参数（页数+默认模板）
    task = {
        "numberOfPages": numberOfPages,
        "pptx": "default_template",
    }
    if pptxFile is not None:
        # 以MD5作为模板缓存键，避免同一模板重复解析
        pptx_blob = await pptxFile.read()
        pptx_md5 = hashlib.md5(pptx_blob).hexdigest()
        task["pptx"] = pptx_md5
        pptx_dir = pjoin(RUNS_DIR, "pptx", pptx_md5)
        if not os.path.exists(pptx_dir):
            os.makedirs(pptx_dir, exist_ok=True)
            with open(pjoin(pptx_dir, "source.pptx"), "wb") as f:
                f.write(pptx_blob)
    if pdfFile is not None:
        # 以MD5作为PDF缓存键，避免重复解析同一文档
        pdf_blob = await pdfFile.read()
        pdf_md5 = hashlib.md5(pdf_blob).hexdigest()
        task["pdf"] = pdf_md5
        pdf_dir = pjoin(RUNS_DIR, "pdf", pdf_md5)
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir, exist_ok=True)
            with open(pjoin(pdf_dir, "source.pdf"), "wb") as f:
                f.write(pdf_blob)
    if topic is not None:
        # 23-24) 主题生成模式：用主题文本替代PDF标识
        task["pdf"] = topic
    # 25) 注册到内存任务池
    progress_store[task_id] = task
    # 26) 异步启动主流程
    asyncio.create_task(ppt_gen(task_id))
    # 27) 返回前端可轮询/连接的任务ID
    return {"task_id": task_id.replace("/", "|")}


async def send_progress(websocket: Optional[WebSocket], status: str, progress: int):
    # 断线场景允许静默返回，避免中断后端流水线
    if websocket is None:
        logger.info(f"websocket is None, status: {status}, progress: {progress}")
        return
    # 统一推送进度协议，前端据此更新阶段状态
    await websocket.send_json({"progress": progress, "status": status})


@app.websocket("/wsapi/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    task_id = task_id.replace("|", "/")
    if task_id in progress_store:
        await websocket.accept()
    else:
        raise HTTPException(status_code=404, detail="Task not found")
    active_connections[task_id] = websocket
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("websocket disconnected: %s", task_id)
        active_connections.pop(task_id, None)


@app.get("/api/download")
async def download(task_id: str):
    task_id = task_id.replace("|", "/")
    if not os.path.exists(pjoin(RUNS_DIR, task_id)):
        raise HTTPException(status_code=404, detail="Task not created yet")
    file_path = pjoin(RUNS_DIR, task_id, "final.pptx")
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="application/pptx",
            headers={"Content-Disposition": "attachment; filename=pptagent.pptx"},
        )
    raise HTTPException(status_code=404, detail="Task not finished yet")


@app.post("/api/feedback")
async def feedback(request: Request):
    body = await request.json()
    feedback = body.get("feedback")
    task_id = body.get("task_id")

    with open(pjoin(RUNS_DIR, "feedback", f"{task_id}.txt"), "w") as f:
        f.write(feedback)
    return {"message": "Feedback submitted successfully"}


@app.get("/")
async def hello():
    return {"message": "Hello, World!"}


async def ppt_gen(task_id: str, rerun=False):
    if DEBUG:
        # 1-2) 开发模式热重载，保证改动即时生效
        importlib.reload(induct)
        importlib.reload(pptgen)
    if rerun:
        # 3-6) 重跑模式：恢复task_id格式、占位连接并从task.json恢复任务参数
        task_id = task_id.replace("|", "/")
        active_connections[task_id] = None
        progress_store[task_id] = json.load(open(pjoin(RUNS_DIR, task_id, "task.json")))

    # 7-10) 最多等待约2秒WebSocket连接，超时则取消任务
    for _ in range(100):
        if task_id in active_connections:
            break
        await asyncio.sleep(0.02)
    else:
        # 前端长时间未连接则放弃执行，避免无主任务占用资源
        progress_store.pop(task_id)
        return

    # 11) 取出并移除任务配置
    task = progress_store.pop(task_id)
    # 12-13) 读取模板与文档标识
    pptx_md5 = task["pptx"] # 默认 default_template，如果上传ppt则为ppt的id
    pdf_md5 = task["pdf"] # 如果上传pdf则为pdf的id
    # 14-16) 构建运行目录配置并落盘task.json以支持重跑
    generation_config = Config(pjoin(RUNS_DIR, task_id)) # /root/autodl-tmp/PPTAgent/runs/2025-05-13/4fecf64f-b74a-4cb4-aad4-676710960784/
    pptx_config = Config(pjoin(RUNS_DIR, "pptx", pptx_md5)) # /root/autodl-tmp/PPTAgent/runs/pptx/pptx_md5
    json.dump(task, open(pjoin(generation_config.RUN_DIR, "task.json"), "w"))
    # 17-20) 初始化进度管理与关键目录
    progress = ProgressManager(task_id, STAGES)
    parsedpdf_dir = pjoin(RUNS_DIR, "pdf", pdf_md5) # /root/autodl-tmp/PPTAgent/runs/pdf/pdf_md5
    ppt_image_folder = pjoin(pptx_config.RUN_DIR, "slide_images") # /root/autodl-tmp/PPTAgent/runs/pptx/pptx_md5/slide_images

    await send_progress(
        active_connections[task_id], "task initialized successfully", 10
    )

    try:
        # Step 1: 解析参考PPT并准备页图
        # 21) 加载模板PPT到中间结构
        presentation = Presentation.from_file(
            pjoin(pptx_config.RUN_DIR, "source.pptx"), pptx_config
        )
        # 22-28) 页图缺失或数量不一致时，重建并按真实页序重排
        if not os.path.exists(ppt_image_folder) or len( # 第一次上传的PPT需要解析
            os.listdir(ppt_image_folder)
        ) != len(presentation):
            await ppt_to_images_async(
                pjoin(pptx_config.RUN_DIR, "source.pptx"), ppt_image_folder
            )
            print(ppt_image_folder, len(os.listdir(ppt_image_folder)), len(presentation), len(
                presentation.error_history
            ))
            

            for err_idx, _ in presentation.error_history:
                os.remove(pjoin(ppt_image_folder, f"slide_{err_idx:04d}.jpg"))
            for i, slide in enumerate(presentation.slides, 1):
                slide.slide_idx = i
                os.rename(
                    pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg"),
                    pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg"),
                )
    # 29-34) 图片打标：优先复用缓存，否则调用视觉模型生成caption
        labler = ImageLabler(presentation, pptx_config)
        if os.path.exists(pjoin(pptx_config.RUN_DIR, "image_stats.json")):
            image_stats = json.load(
                open(pjoin(pptx_config.RUN_DIR, "image_stats.json"))
            )
            labler.apply_stats(image_stats) 
        else:
            await labler.caption_images_async(models.vision_model) # 为幻灯片中的图像生成标题
            json.dump(
                labler.image_stats,
                open(pjoin(pptx_config.RUN_DIR, "image_stats.json"), "w"),
                ensure_ascii=False,
                indent=4,
            )
        await progress.report_progress()

        # Step 2: 解析PDF为 markdown + 图片 + 元数据
        # 36-39) 文档解析：首次解析PDF，否则复用source.md缓存
        if not os.path.exists(pjoin(parsedpdf_dir, "source.md")):
            text_content = parse_pdf( # 从PDF中解析出文本、图像和元数据，分别存储为markdown、jpeg、json格式
                pjoin(RUNS_DIR, "pdf", pdf_md5, "source.pdf"),
                parsedpdf_dir,
                models.marker_model,
            )
        else:
            text_content = open(pjoin(parsedpdf_dir, "source.md")).read()
        await progress.report_progress()

        # Step 3: 将文档结构化为章节/子章节/媒体索引
        # 40-44) 文档结构化：首次生成refined_doc.json，否则反序列化缓存
        if not os.path.exists(pjoin(parsedpdf_dir, "refined_doc.json")):
            source_doc = await Document.from_markdown_async(
                text_content,
                models.language_model,
                models.vision_model,
                parsedpdf_dir,
            )
            json.dump(
                source_doc.to_dict(),
                open(pjoin(parsedpdf_dir, "refined_doc.json"), "w"),
                ensure_ascii=False,
                indent=4,
            )
        else:
            source_doc = json.load(open(pjoin(parsedpdf_dir, "refined_doc.json")))
            source_doc = Document.from_dict(source_doc, parsedpdf_dir)
        await progress.report_progress()

        # Step 4: 归纳模板布局与内容Schema（Stage I）
        # 45-53) Stage I：布局归纳与内容schema抽取，结果缓存为slide_induction.json
        if not os.path.exists(pjoin(pptx_config.RUN_DIR, "slide_induction.json")):
            deepcopy(presentation).save(
                pjoin(pptx_config.RUN_DIR, "template.pptx"), layout_only=True
            )
            await ppt_to_images_async(
                pjoin(pptx_config.RUN_DIR, "template.pptx"),
                pjoin(pptx_config.RUN_DIR, "template_images"),
            )
            slide_inducter = induct.SlideInducterAsync( # PPT分析，将PPT按照布局聚类，并提取内容模式
                presentation,
                ppt_image_folder,
                pjoin(pptx_config.RUN_DIR, "template_images"),
                pptx_config,
                models.image_model,
                models.language_model,
                models.vision_model,
            )
            layout_induction = await slide_inducter.layout_induct() # 自动识别幻灯片的布局模板
            slide_induction = await slide_inducter.content_induct(layout_induction) # 从幻灯片中提取内容
            json.dump(
                slide_induction,
                open(pjoin(pptx_config.RUN_DIR, "slide_induction.json"), "w"),
                ensure_ascii=False,
                indent=4,
            )
        else:
            slide_induction = json.load(
                open(pjoin(pptx_config.RUN_DIR, "slide_induction.json"))
            )
        await progress.report_progress()

        # Step 5: 基于文档+模板生成最终演示文稿（Stage II）
        # 54-59) Stage II：构造生成器、注入参考信息并生成final.pptx
        ppt_agent = pptgen.PPTAgentAsync( # PPT生成，输入是参考PPT，输出是大纲和最终PPT
            models.text_model,
            models.language_model,
            models.vision_model,
            error_exit=False,
            retry_times=5,
        )
        ppt_agent.set_reference( # 设置参考演示文稿及提取信息
            config=generation_config,
            slide_induction=slide_induction,
            presentation=presentation,
        )
        
        prs, _ = await ppt_agent.generate_pres( # 生成PPT
            source_doc=source_doc,
            num_slides=task["numberOfPages"],
        )
        prs.save(pjoin(generation_config.RUN_DIR, "final.pptx"))
        logger.info(f"{task_id}: generation finished")
        await progress.report_progress()
    except Exception as e:
        # 60-62) 统一异常兜底：推送失败状态并输出堆栈
        await progress.fail_stage(str(e))
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn

    ip = "0.0.0.0"
    uvicorn.run(app, host=ip, port=9297)
