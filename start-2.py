import ollama

# chatting with ollama
res=ollama.chat(
    model="llama3.2",
    messages=[
        {"role":"user", "content":"Why is the sky blue"}
    ],
    stream=True
    )
# print(res["message"]["content"])

# generate example
res=ollama.generate(
    model="llama3.2",
    prompt="Why is the sky blue"
)

# create a new model with modelfile
modelfile = """
FROM llama3.2
SYSTEM You are a very smart assistant who knows everything about oceans. You are very succinct and informative
PARAMETER temperature 0.1
"""
ollama.create(model="knowitall", modelfile=modelfile)
res=ollama.generate(model="knowitall", prompt="why is the ocean so salty?")
print(res["response"])

ollama.delete("knowitall")