from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import os
import json
from main import generate_math_modeling_paper, get_last_tavily_payload, get_last_tavily_run_dir
import threading
import queue

app = FastAPI(
    title="数学建模论文生成API",
    description="用于生成数学建模论文的后端API服务",
    version="1.0.0"
)

# 静态挂载：用于下载/查看每次 Tavily 爬取落盘的 CSV 与 payload.json
if os.path.exists("tavily_runs"):
    app.mount("/tavily_runs", StaticFiles(directory="tavily_runs"), name="tavily_runs")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型
class ProblemRequest(BaseModel):
    problem: str
    # 可选：多轮对话上下文（用于“再次生成/迭代扩写”时补充信息）
    conversation_context: str | None = None

class GenerationStatus(BaseModel):
    status: str
    progress: int
    message: str

class GenerationResult(BaseModel):
    status: str
    results: dict
    charts: list
    web_data: dict
    raw_csvs: list | None = None
    cleaned_csvs: list | None = None
    run_dir: str | None = None

# 全局变量，用于存储生成状态
generation_status = {
    "status": "idle",
    "progress": 0,
    "message": "等待开始"
}

# 进度队列，用于实时传递进度信息
progress_queue = queue.Queue()

# 生成结果存储
generation_result = {
    "results": {},
    "charts": [],
    "web_data": {}
}

# 模拟生成结果
mock_results = {
    "abstract": "本文通过建立和分析数学模型，解决了城市交通拥堵问题，对城市规划和交通管理具有重要的理论和实践意义。在模型建立方面，针对问题一，我们构建了一个基于线性回归的模型；针对问题二，我们发展了一个基于时间序列的模型；对于问题三，我们提出了一个基于优化理论的模型。通过这些模型的建立和验证，我们得到了交通流量预测结果。在分析模型的优缺点后，我们提出了改进措施，以期提高模型的准确性。",
    "problem_restatement": "1.1 问题背景：随着城市化进程的加速，城市交通拥堵问题日益严重，给人们的出行带来了极大的不便，同时也造成了巨大的经济损失和环境压力。如何有效缓解交通拥堵，提高交通系统的运行效率，成为城市规划和交通管理部门亟待解决的问题。\n\n1.2 问题提出：针对城市交通拥堵问题，我们需要建立数学模型来分析和预测交通流量，评估不同交通管理策略的效果，并提出最优的交通疏导方案。",
    "problem_analysis": "2.1 问题一分析：交通流量预测主要包括时间因素、空间因素和天气因素等几个方面，根据历史交通数据的特征，建立基于线性回归的预测模型，提取了交通流量、车速、道路占有率等指标对数据进行特征训练。\n\n2.2 问题二分析：交通拥堵评估主要包括拥堵程度、持续时间和影响范围等几个方面，根据交通状态数据的特征，建立基于时间序列的评估模型，提取了拥堵指数、延误时间、排队长度等指标对数据进行特征训练。\n\n2.3 问题三分析：交通疏导方案优化主要包括信号配时、路径诱导和流量控制等几个方面，根据交通网络的特征，建立基于优化理论的决策模型，提取了通行能力、延误成本、环境影响等指标对数据进行特征训练。",
    "model_assumption": "1. 假设交通流量在短时间内具有稳定性\n2. 假设天气因素对交通流量的影响可以量化\n3. 假设道路通行能力是固定的\n4. 假设驾驶员的行为是理性的\n5. 假设交通信号的控制是有效的\n6. 假设数据采集是准确的",
    "symbol_explanation": "交通流量---Q\n车速---V\n道路占有率---O\n拥堵指数---CI\n延误时间---D\n排队长度---L\n通行能力---C\n延误成本---DC\n环境影响---EI",
    "model_building": "3.1 问题一模型建立：我们构建了一个基于线性回归的交通流量预测模型，模型的数学表达式为Q = β0 + β1*T + β2*W + β3*H + ε，其中T为时间因素，W为天气因素，H为节假日因素，β为回归系数，ε为误差项。模型的建立步骤包括数据收集、特征提取、模型训练和参数估计。\n\n3.2 问题二模型建立：我们构建了一个基于时间序列的交通拥堵评估模型，模型的数学表达式为CI(t) = α*CI(t-1) + β*Q(t) + γ*V(t) + ε(t)，其中CI(t)为t时刻的拥堵指数，Q(t)为t时刻的交通流量，V(t)为t时刻的车速，α、β、γ为模型参数，ε(t)为误差项。\n\n3.3 问题三模型建立：我们构建了一个基于优化理论的交通疏导方案模型，目标函数为min Σ(DCi + EIi)，约束条件包括交通流量守恒、道路通行能力限制和信号配时约束等。\n\n3.4 模型关联性分析：三个模型之间存在密切的关联关系，问题一的预测结果是问题二评估的输入，问题二的评估结果是问题三优化的依据，三个模型协同工作，共同解决城市交通拥堵问题。",
    "model_solving": "4.1 问题一模型求解：我们使用最小二乘法对线性回归模型进行求解，通过Python和NumPy库实现了模型的训练和参数估计，求解结果显示时间因素和天气因素对交通流量的影响显著。\n\n4.2 问题二模型求解：我们使用ARIMA模型对时间序列数据进行分析和预测，通过Python和StatsModels库实现了模型的训练和参数估计，求解结果显示拥堵指数具有明显的时间相关性。\n\n4.3 问题三模型求解：我们使用遗传算法对优化模型进行求解，通过Python和SciPy库实现了模型的求解和参数优化，求解结果显示最优的交通疏导方案可以显著减少延误成本和环境影响。",
    "results_analysis": "5.1 问题一结果分析：交通流量预测结果显示，工作日的交通流量明显高于周末，早高峰和晚高峰的交通流量明显高于其他时段，天气因素对交通流量的影响也很显著。\n\n5.2 问题二结果分析：交通拥堵评估结果显示，城市中心区域的拥堵程度明显高于郊区，早高峰和晚高峰的拥堵程度明显高于其他时段，拥堵持续时间与交通流量呈正相关。\n\n5.3 问题三结果分析：交通疏导方案优化结果显示，合理的信号配时和路径诱导可以显著减少交通拥堵，提高交通系统的运行效率，同时减少环境影响。"
}

mock_charts = [
    {"name": "交通流量时间序列", "file": "timeseries_1.png"},
    {"name": "交通流量分布图", "file": "barchart_2.png"},
    {"name": "交通流量与车速关系", "file": "scatter_3.png"},
    {"name": "交通拥堵热力图", "file": "heatmap_4.png"},
    {"name": "车速分布直方图", "file": "histogram_5.png"}
]

mock_web_data = {}

# API端点
def update_progress(progress, message):
    """更新进度信息"""
    global generation_status
    # 如果进度为 None，则只更新消息，不更新进度值
    if progress is not None:
        generation_status.update({
            "progress": progress,
            "message": message
        })
        progress_queue.put({"progress": progress, "message": message})
    else:
        # 对于耗时信息，只更新消息，保持当前进度值
        current_progress = generation_status.get("progress", 0)
        generation_status.update({
            "progress": current_progress,
            "message": message
        })
        progress_queue.put({"progress": current_progress, "message": message})

async def monitor_progress():
    """监控进度并更新状态"""
    while True:
        try:
            # 非阻塞获取队列中的进度信息
            progress_info = progress_queue.get(block=False)
            update_progress(progress_info["progress"], progress_info["message"])
        except queue.Empty:
            pass
        await asyncio.sleep(0.1)

@app.post("/generate", response_model=GenerationResult)
async def generate_paper(request: ProblemRequest):
    """生成数学建模论文"""
    global generation_status, generation_result
    
    if generation_status["status"] == "running":
        raise HTTPException(status_code=400, detail="已有生成任务正在进行中")
    
    # 更新状态为运行中
    generation_status = {
        "status": "running",
        "progress": 0,
        "message": "开始生成..."
    }
    
    try:
        # 启动进度监控
        progress_task = asyncio.create_task(monitor_progress())
        
        # 定义进度回调函数
        def progress_callback(progress, message):
            """进度回调函数"""
            update_progress(progress, message)
        
        # 直接调用 main.py 中的函数生成论文，传递进度回调
        paper_path = generate_math_modeling_paper(
            request.problem,
            progress_callback=progress_callback,
            conversation_context=request.conversation_context or "",
        )
        
        # 取消进度监控任务
        progress_task.cancel()
        
        # 模拟结果数据（实际项目中应该从生成的论文中提取）
        # 这里暂时使用模拟数据，后续可以从生成的Word文档中提取真实内容
        mock_results = {
            "abstract": "本文通过建立和分析数学模型，解决了问题，对相关领域具有重要的理论和实践意义。",
            "problem_restatement": "1.1 问题背景：...\n\n1.2 问题提出：...",
            "problem_analysis": "2.1 问题一分析：...\n\n2.2 问题二分析：...\n\n2.3 问题三分析：...",
            "model_assumption": "1. 假设...\n2. 假设...\n3. 假设...\n4. 假设...\n5. 假设...\n6. 假设...",
            "symbol_explanation": "符号1---S1\n符号2---S2\n符号3---S3",
            "model_building": "3.1 问题一模型建立：...\n\n3.2 问题二模型建立：...\n\n3.3 问题三模型建立：...\n\n3.4 模型关联性分析：...",
            "model_solving": "4.1 问题一模型求解：...\n\n4.2 问题二模型求解：...\n\n4.3 问题三模型求解：...",
            "results_analysis": "5.1 问题一结果分析：...\n\n5.2 问题二结果分析：...\n\n5.3 问题三结果分析：..."
        }
        
        # 生成图表数据
        charts_dir = "charts"
        chart_files = []
        if os.path.exists(charts_dir):
            for file in os.listdir(charts_dir):
                if file.endswith(".png"):
                    chart_files.append({"name": file, "file": file})
        
        # Tavily 联网数据（真实抓取结果，含 sources + tables + summary）
        tavily_payload = get_last_tavily_payload() or {}
        run_dir = get_last_tavily_run_dir() or ""
        if tavily_payload and run_dir:
            tavily_payload = dict(tavily_payload)
            tavily_payload["run_dir"] = os.path.basename(run_dir)
        
        # 更新结果
        generation_result = {
            "results": mock_results,
            "charts": chart_files,
            "web_data": tavily_payload,
            "raw_csvs": tavily_payload.get("raw_csvs", []) if isinstance(tavily_payload, dict) else [],
            "cleaned_csvs": tavily_payload.get("cleaned_csvs", []) if isinstance(tavily_payload, dict) else [],
            "run_dir": tavily_payload.get("run_dir") if isinstance(tavily_payload, dict) else None,
        }
        
        # 更新状态为完成
        generation_status = {
            "status": "completed",
            "progress": 100,
            "message": f"生成完成！论文保存路径：{paper_path}"
        }
        
        # 返回结果
        return GenerationResult(
            status="success",
            results=mock_results,
            charts=chart_files,
            web_data=tavily_payload,
            raw_csvs=generation_result["raw_csvs"],
            cleaned_csvs=generation_result["cleaned_csvs"],
            run_dir=generation_result["run_dir"],
        )
        
    except Exception as e:
        # 更新状态为错误
        generation_status = {
            "status": "error",
            "progress": 0,
            "message": f"生成失败：{str(e)}"
        }
        raise HTTPException(status_code=500, detail=f"生成失败：{str(e)}")

@app.get("/status", response_model=GenerationStatus)
async def get_status():
    """获取生成状态"""
    global generation_status
    return generation_status

@app.get("/charts")
async def get_charts():
    """获取图表列表"""
    return mock_charts

@app.get("/web-data")
async def get_web_data():
    """获取网络数据"""
    return get_last_tavily_payload() or {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
