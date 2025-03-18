import hydra

from transformers import (
    pipeline,
    AutoTokenizer
)

@hydra.main(version_base=None, config_path="../config", config_name="finetune.yaml")
def infer(config):

    model_hub = config["model"]["hub_model_id"]
    print("Infer with model: ", model_hub)
    tokenizer = AutoTokenizer.from_pretrained(model_hub)

    print("tokenizer: ", tokenizer)
    chat_model = pipeline("text-generation", model=model_hub, tokenizer=tokenizer)
    _input = input("Instruction: ")
    messages = [
        {"role": "system", "content": config["data"]["prompt_system"]},
        {"role": "user", "content": _input}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    output = chat_model(input_text, max_length=50)
    print(output)
    
if __name__ == "__main__":
    infer()