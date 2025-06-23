import sys
from llama_cpp import Llama



llm = Llama(
    model_path=sys.argv[1],
    n_ctx=512,
    n_threads=4,
    n_gpu_layers=1,
    verbose=False
)

print("Chat with Llama (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]: break

    prompt = f"### Human: {user_input}\n### Assistant:"
    output = llm(
        prompt,
        max_tokens=100,
        stop=["###", "### Human:", "\n###"]
    )
    response = output["choices"][0]["text"].strip()
    print("Bot:", response)
