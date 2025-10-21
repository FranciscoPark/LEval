import os
import math
from functools import partial
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
from LEval_config import *

def main():
    start_idx = 0
    for file_name, data in key_data_pairs.items():
        save_path = os.path.join(data_save_path, os.path.basename(file_name))
        with open(save_path, "w") as fw:
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
            sys_prompt = get_sys_prompt(args, file_name)

            for d in tqdm(data):
                document = d["input"]
                cnt = 0
                while num_tokens_from_string(document, tokenizer) > max_length:
                    if "code" not in file_name:
                        document = " ".join(document.split(" ")[:max_length - cnt])
                    else:
                        document = " ".join(document.split(" ")[cnt - max_length:])
                    cnt += 250

                for inst, out in zip(d["instructions"], d["outputs"]):
                    save_d = {"query": inst, "gt": out}
                    context = f"Document is as follows. {document} \nQuestion: {inst}"
                    message = f"{B_INST}{B_SYS}{sys_prompt}{E_SYS}{context}{E_INST}\nAnswer:"
                    text_inputs = message.format(document=document, inst=inst)
                    save_d["prompt"] = message.replace(document, "<long document>")

                    inputs = tokenizer(text_inputs, return_tensors="pt").to(model.device)
                    sample = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
                    prompt_length = inputs.input_ids.size(-1)
                    output = tokenizer.decode(sample[0][prompt_length:])
                    save_d[f"{open_source_model}_pred"] = output.replace("</s>", "")
                    save_d["evaluation"] = d["evaluation"]

                    fw.write(json.dumps(save_d) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", required=True)
    parser.add_argument("--max_length", default="4k")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scale", default="7b", choices=["7b", "13b"])
    args = parser.parse_args()

    model_path = "/mnt/jy/Llama-2-7b-hf"
    open_source_model = f"llama2-{args.scale}-chat-{args.max_length}"
    data_save_path = f"/mnt/jy/LEval/Predictions/{args.metric}/{open_source_model}"
    os.makedirs(data_save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        legacy=True,
        trust_remote_code=True,
    )

    # ✅ This will automatically shard across all available GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",             
        torch_dtype=torch.bfloat16,    
        low_cpu_mem_usage=True
    ).eval()

    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())