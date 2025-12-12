import socket
import threading
import json


class BaseClient:
    def __init__(self, host='127.0.0.1', port=5555):
        self.host = host
        self.port = port

    def send_request(self, task_type, judge_type, partial_trajectories, question=""):
        request = {
            "task-type": task_type,
            "judge-type": judge_type,
            "partial_trajectories": partial_trajectories,
            "question": question,
        }
        data = json.dumps(request).encode("utf-8")
        msglen = len(data).to_bytes(4, "big")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            sock.sendall(msglen + data)
            header = sock.recv(4)
            if len(header) < 4:
                raise Exception("Invalid response header")
            resp_len = int.from_bytes(header, "big")
            resp_data = b""
            while len(resp_data) < resp_len:
                part = sock.recv(resp_len - len(resp_data))
                if not part:
                    break
                resp_data += part
            response = json.loads(resp_data.decode("utf-8"))
            return response


if __name__ == "__main__":
    client = BaseClient()
    response = client.send_request(
        task_type="math",
        judge_type="judge",
        question="What is 2+2?",
        partial_trajectories=[
            {"context": "Let me calculate: 2+2 = 4", "current-step": "Calculating sum"},
            {"context": "The sum of 2 and 2 equals 5... wait, that's wrong. It's 4.", "current-step": "Correcting calculation"},
            {"context": "2+2 is obviously 4", "current-step": "Returning answer"}
        ]
    )
    print("Client received response:")
    print("Rankings:", response.get("rankings"))
    print("\nJudge feedback:")
    print(response.get("judge-feedback"))
