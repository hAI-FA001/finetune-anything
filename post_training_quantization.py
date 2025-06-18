# import os
# for now, don't use GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from extend_sam import get_model
import argparse
from omegaconf import OmegaConf


def quantize(model, baseModelPath):
    # backend = "qnnpack"  # mobile, which uses ARM
    backend = "x86"  # PC, which uses x86

    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend

    print("Preparing...")
    model_static_quantized = torch.quantization.prepare(model, inplace=False)

    print("Converting...")
    model_static_quantized = torch.quantization.convert(
        model_static_quantized, inplace=False
    )

    saveModelPath = baseModelPath.replace('.pt', '-q.pt')
    print(f"Saving to {saveModelPath}")
    torch.save(model.state_dict(), saveModelPath)


supported_tasks = ["detection", "semantic_seg", "instance_seg"]
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="./experiment/model/semantic_sam/model.pth", type=str)
parser.add_argument("--task_name", default="semantic_seg", type=str)
parser.add_argument("--cfg", default=None, type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    task_name = args.task_name
    if args.cfg is not None:
        config = OmegaConf.load(args.cfg)
    else:
        assert task_name in supported_tasks, "Please input the supported task name."
        config = OmegaConf.load(
            "./config/{task_name}.yaml".format(task_name=args.task_name)
        )

    train_cfg = config.train
    gpuFound = torch.cuda.is_available()

    print("Loading model...")
    model = get_model(
        model_name=train_cfg.model.sam_name,
        haveGPU=gpuFound,
        **train_cfg.model.params,
    )

    print("Loading weights...")
    model.load_state_dict(
        torch.load(
            args.model_path,
            weights_only=True,
            map_location=torch.device("cpu" if not gpuFound else "cuda"),
        )
    )

    print("Beginning quantization")
    quantize(model, path=args.model_path)
