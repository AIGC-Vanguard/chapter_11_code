import requests


def get_ans(prompt,history):
    message = []
    for item in history:
        message.append(item[1])
    try:
        ans = chatglm_api(prompt, history=message)
    except:
        ans = "服务调用失败"
    return ans


def chatglm_api(query, history=[], top_p=0.7, temperature=0.95):
    headers = {
        'Content-Type': 'application/json',
    }
    # api_link = "http://10.164.67.35:8410"
    api_link = "http://ip:port"
    output = requests.post(api_link, headers=headers,
                           json={"prompt": query, "history": history, "top_p": top_p, "temperature": temperature,
                                 "max_length": 4096})
    return output.json()["response"]