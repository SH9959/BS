import gradio as gr
import socket
import random
import time

# 客户端配置
client_ip = '127.0.0.1'  # 服务器的IP地址
client_port = 22223  # 服务器的端口号
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((client_ip, client_port))
print(f'Connected to server at {client_ip}:{client_port}')

# 设置超时时间为 1 秒
client_socket.settimeout(1)

# 发送消息给服务器的函数
def send_message(message):
    client_socket.sendall(message.encode('utf-8'))

# 从服务器接收消息的函数
def receive_message():
    try:
        return client_socket.recv(1024).decode('utf-8')
    except socket.timeout:
        return None

with gr.Blocks() as demo:
    chatbot = gr.Chatbot() # 对话框
    msg = gr.Textbox() # 输入文本框
    clear = gr.ClearButton([msg, chatbot]) # 清除按钮
    def chat_bot(message, chat_history):
        send_message(message)
        response = receive_message()
        if response is None:
            return "No response from server"
        
        chat_history.append((message, response))
        time.sleep(0.5)
        return "", chat_history
    msg.submit(chat_bot, [msg, chatbot], [msg, chatbot])
    
demo.launch()


# def chat_bot(message):
#     send_message(message)
#     response = receive_message()
#     if response is None:
#         return "No response from server"
#     return response

# iface = gr.Interface(fn=chat_bot, inputs="textbox", outputs="textbox", title="ChatBot")
# iface.launch()
