import socket
import json


HOST = "127.0.0.1"
PORT = 8000
BUFFER_SIZE = 1024


def receive_line(sock, buffer):
    while b"\n" not in buffer:
        data = sock.recv(BUFFER_SIZE)

        if not data:
            return None, buffer

        buffer += data

    message_bytes, buffer = buffer.split(b"\n", 1)
    message = message_bytes.decode("utf-8")
    return message, buffer


client_socket = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)
client_socket.connect((HOST, PORT))

request_count = 0
reply_buffer = b""

while True:
    message = input("请输入内容，输入 quit 退出：")
    command = message.strip()
    if command == "quit":
        break

    if not command:
        print("消息不能为空")
        continue

    request_count += 1
    request_id = f"req-{request_count}"

    if command == "bad-json":
        json_text = (
            f'{{"request_id": "{request_id}", '
            f'"action": "chat", "content": }}'
        )

    elif command == "missing-content":
        request = {
            "request_id": request_id,
            "action": "chat"
        }
        json_text = json.dumps(
            request,
            ensure_ascii=False
        )

    elif command == "bad-action":
        request = {
            "request_id": request_id,
            "action": "not-exist",
            "content": "你好"
        }

        json_text = json.dumps(
            request,
            ensure_ascii=False
        )

    elif command.startswith("upper:"):
        content = message.split(":", 1)[1].strip()
        request = {
            "request_id": request_id,
            "action": "upper",
            "content": content
        }

        json_text = json.dumps(
            request,
            ensure_ascii=False
        )

    else:
        request = {
            "request_id": request_id,
            "action": "chat",
            "content": message
        }
        json_text = json.dumps(
            request,
            ensure_ascii=False
        )

    wire_message = json_text + "\n"
    client_socket.sendall(
        wire_message.encode("utf-8")
    )

    reply_text, reply_buffer = receive_line(
        client_socket,
        reply_buffer
    )

    if reply_text is None:
        print("服务端关闭了连接")
        break

    response = json.loads(reply_text)
    print("请求编号：", response["request_id"])

    if response["ok"]:
        print("服务端回复：", response["result"])
    else:
        print("请求失败：", response["error"])

client_socket.close()
