import socket


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

try:
    client_socket.connect((HOST, PORT))
    reply_buffer = b""

    while True:
        message = input("请输入内容，输入 quit 退出：")
        command = message.strip()

        if command == "quit":
            break

        if not command:
            print("消息不能为空")
            continue

        wire_message = message + "\n"
        client_socket.sendall(wire_message.encode("utf-8"))

        reply_text, reply_buffer = receive_line(
            client_socket,
            reply_buffer
        )

        if reply_text is None:
            print("服务端关闭了连接")
            break

        print("服务端回复：", reply_text)

finally:
    client_socket.close()
