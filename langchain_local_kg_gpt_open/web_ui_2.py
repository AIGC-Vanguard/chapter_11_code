import os
import gradio as gr
from collections import defaultdict
from application.qa_bot import QABot,QABotConfig


# 用户历史消息记录
user_chat_history = defaultdict(list)
# 产品名称到id的映射
name_id_map = {"思加图复古凉鞋":"145","ONG透气跑鞋":"196"}
#QA boot
config = QABotConfig(embedding_model_name="./embed_model/shibing624-text2vec-base-chinese",
                     docs_path="./test_data/shoes_data.tsv",
                     vector_store_path = "./test_data",
                     history_len=10)
bot = QABot(config)


def predict(query,
            large_language_model,
            top_k,
            goods_name,
            history=None):
    if history == None:
        history = []
    goods_id = name_id_map[goods_name]
    result,search = bot.get_knowledge_based_answer(query,
                                                    large_language_model,
                                                    goods_id,
                                                    top_k=top_k,
                                                    chat_history=user_chat_history[goods_id])

    if result != "服务调用失败":
        user_chat_history[goods_id].append(["user",query])
        user_chat_history[goods_id].append(["assistant",result])
    history.append([query, result])
    return '', history, history, search


def clear_session(request: gr.Request):
    return '', None


init_message = "我是ShoesGPT，你高兴为你服务。"
block_css = """
.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}
"""
default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    gr.Markdown("""<h1><center>ShoesGPT</center></h1><center><font size=3></center></font>""")
    state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            goods_name = gr.Dropdown(
                [
                    "思加图复古凉鞋",
                    "ONG透气跑鞋",
                ],
                label="商品列表",
                value="思加图复古凉鞋")

            embedding_model = gr.Dropdown(
                [
                "shibing624/text2vec-base-chinese"
                ],
                label="向量化模型",
                value="shibing624/text2vec-base-chinese")

            large_language_model = gr.Dropdown(
                [
                    "ChatGPT",
                    "ChatGLM-6B",
                    "ChatGLM-6B-SFT",
                ],
                label="large language model",
                value="ChatGPT")

            top_k = gr.Slider(3,
                              10,
                              value=3,
                              step=1,
                              label="检索top-k文档",
                              interactive=True)

        with gr.Column(scale=4):
            with gr.Row():
                chatbot = gr.Chatbot([[None, init_message]],label=" ").style(height=400)
            with gr.Row():
                message = gr.Textbox(label='请输入问题')
            with gr.Row():
                clear_history = gr.Button("🧹 清除历史对话")
                send = gr.Button("🚀 发送")

        with gr.Column(scale=2):
            search = gr.Textbox(label='搜索结果')

        # ============= 触发动作=============
        # 发送按钮 提交
        send.click(predict,
                   inputs=[
                       message,
                       large_language_model,
                       top_k,
                       goods_name
                   ],
                   outputs=[message, chatbot, state, search])
        # 清空历史对话按钮 提交
        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[chatbot, state],
                            queue=False)
        # 输入框 回车
        message.submit(predict,
                       inputs=[
                           message,
                           large_language_model,
                           top_k,
                           goods_name,
                           state
                       ],
                       outputs=[message, chatbot, state, search])

demo.queue().launch(
    server_name='0.0.0.0',
    server_port=7860,
    share=False,
    show_error=True,
    debug=True,
    enable_queue=True,
    inbrowser=True
)
