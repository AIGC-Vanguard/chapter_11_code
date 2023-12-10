import openai

openai.api_key = "your_key"
model = 'gpt-3.5-turbo'


def get_ans(prompt,history):
    message = []
    for item in history:
        message.append({"role": item[0], "content": item[1]})
    message.append({"role": "user", "content": prompt})

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=message
        )
        ans = response.choices[0].message.content
    except:
        ans = "服务调用失败"
    return ans