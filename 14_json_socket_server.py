import socket
import threading
import json


HOST = "127.0.0.1"
PORT = 8000
BUFFER_SIZE = 1024


def handle_client(client_socket, client_address):
    buffer = b""

    try:
        while True:
            data = client_socket.recv(BUFFER_SIZE)

            if not data:
                print("客户端正常关闭：", client_address)
                break

            buffer += data

            while b"\n" in buffer:
                message_bytes, buffer = buffer.split(b"\n", 1)
                text = message_bytes.decode("utf-8")
                request_id = None

                try:
                    request = json.loads(text)
                    request_id = request["request_id"]
                    content = request["content"]
                    action = request["action"]
                    print("客户端说：", content)

                    if action == "chat":
                        response = {
                            "request_id": request_id,
                            "ok": True,
                            "action": action,
                            "result": f"服务端收到：{content}"
                        }

                    elif action == "upper":
                        response = {
                            "request_id": request_id,
                            "ok": True,
                            "action": action,
                            "result": content.upper()
                        }

                    else:
                        response = {
                            "request_id": request_id,
                            "ok": False,
                            "error": f"不支持的动作：{action}"
                        }
                except json.JSONDecodeError:
                    response = {
                        "request_id": request_id,
                        "ok": False,
                        "error": "JSON格式不合法"
                    }

                except KeyError as error:
                    missing_field = error.args[0]
                    response = {
                        "request_id": request_id,
                        "ok": False,
                        "error": f"缺少必要字段：{missing_field}"
                    }

                response_text = json.dumps(
                    response,
                    ensure_ascii=False
                )
                wire_reply = response_text + "\n"
                client_socket.sendall(wire_reply.encode("utf-8"))
    except ConnectionResetError:
        print("客户端强制断开：", client_address)

    except OSError as error:
        print("Socket 通信出错：", error)

    finally:
        client_socket.close()


server_socket = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)
server_socket.bind((HOST, PORT))
server_socket.listen()
server_socket.settimeout(1.0)

print("服务端正在等待客户端连接……")
try:
    while True:
        try:
            client_socket, client_address = server_socket.accept()
        except socket.timeout:
            continue

        thread = threading.Thread(
            target=handle_client,
            args=(client_socket, client_address),
            daemon=True
        )
        thread.start()

except KeyboardInterrupt:
    print("\n服务端停止运行")

finally:
    server_socket.close()
