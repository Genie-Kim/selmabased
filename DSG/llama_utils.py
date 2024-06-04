from langchain_groq import ChatGroq
import os
# from dotenv import load_dotenv, find_dotenv

def groq_setup(key_path="DSG/_groq_KEY.txt"):
    with open(key_path) as f:
        key = f.read().strip()
    print("Read key from", key_path)
    # Set the GROQ_API_KEY environment variable
    os.environ["GROQ_API_KEY"] = key


def groq_completion(
    prompt,
    model="llama3-70b-8192",
    temperature=0,
    max_tokens=1000,
    # return_response=False, 
):
    
    llm = ChatGroq(temperature=temperature, model_name=model, max_tokens=max_tokens)
    resp = llm.invoke(prompt)
    # resp = openai.chat.completions.create(
    #     model=model,
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    # )
    # print(prompt)

    # if return_response:
    #     return resp
    output_str = resp.content
    if '\n\n' in output_str:
        start_index = output_str.index('\n\n')
        output = output_str[start_index+len('\n\n'):]
        output = output.strip()
    else:
        output = output_str
    return output


# 그록으로 하면 llama 70b 기준 분당 3장 정도 가능 (분당 10번 prompt 생성하므로,)
# groq + langchain
# ollama는 quantization 안하면 서버에서도 안돌아감.

# chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
# prompt = ChatPromptTemplate.from_messages([("human", "Write a Limerick about {topic}")])
# chain = prompt | chat
# # await chain.ainvoke({"topic": "The Sun"})

# chat = ChatGroq(temperature=0, model_name="llama2-70b-4096")
# prompt = ChatPromptTemplate.from_messages([("human", "Write a haiku about {topic}")])
# chain = prompt | chat
# for chunk in chain.stream({"topic": "The Moon"}):
#     print(chunk.content, end="", flush=True)



