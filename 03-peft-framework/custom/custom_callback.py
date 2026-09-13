"""自定义训练回调示例（LLaMA-Factory / Transformers 通用）。

TrainerCallback 会在训练各生命周期被调用，适合做：平滑 loss、记录显存、
早停、把指标上报到本地文件或可视化面板，且无需改动训练主循环。

接入方式（二选一）：
1) 用 LLaMA-Factory 时，在自定义启动脚本里构造 trainer 后调用
   trainer.add_callback(LossRecorderCallback("train_log.jsonl"))
2) 直接用 Transformers Trainer 时，通过 Trainer(callbacks=[...]) 传入。
"""
import json
import time
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class LossRecorderCallback(TrainerCallback):
    """把平滑后的 loss、学习率、显存占用逐步写入 jsonl，便于事后画图分析。"""

    def __init__(self, log_path: str = "train_log.jsonl", smooth: float = 0.9):
        self.log_path = log_path
        self.smooth = smooth          # 指数滑动平均系数
        self.ema_loss = None
        self.start = time.time()

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kw):
        if not logs or "loss" not in logs:
            return
        raw = float(logs["loss"])
        # 指数滑动平均，让曲线更易读
        self.ema_loss = raw if self.ema_loss is None else (
            self.smooth * self.ema_loss + (1 - self.smooth) * raw
        )
        rec = {
            "step": state.global_step,
            "loss_raw": round(raw, 4),
            "loss_ema": round(self.ema_loss, 4),
            "lr": logs.get("learning_rate"),
            "elapsed_s": round(time.time() - self.start, 1),
        }
        # torch 在时顺手记录已分配显存（MB），没有 GPU 也不报错
        try:
            import torch
            if torch.cuda.is_available():
                rec["cuda_mem_MB"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        except Exception:
            pass
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class EarlyStopOnPlateauCallback(TrainerCallback):
    """极简早停：连续 patience 次日志 loss 都没再下降就停止训练。"""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad = 0

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs or "loss" not in logs:
            return
        cur = float(logs["loss"])
        if cur < self.best - self.min_delta:
            self.best, self.bad = cur, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                print(f"[EarlyStop] 连续 {self.patience} 次无改善，在 step {state.global_step} 停止")
                control.should_training_stop = True   # 框架识别这个标志后安全停止