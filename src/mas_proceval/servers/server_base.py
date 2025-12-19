import socket
import threading
import json
from ..clients.client_base import BaseClient


class BaseServer:
    def __init__(self, host='127.0.0.1', port=5555):
        self.host = host
        self.port = port
        self.server_socket = None
        self.is_running = False

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1000)
        self.is_running = True
        print(f"Server listening on {self.host}:{self.port}")
        try:
            while self.is_running:
                client_sock, addr = self.server_socket.accept()
                print(f"Accepted connection from {addr}")
                client_thread = threading.Thread(
                    target=self.handle_client, args=(client_sock,))
                client_thread.daemon = True
                client_thread.start()
        finally:
            self.server_socket.close()

    def handle_client(self, client_sock):
        with client_sock:
            data = b""
            # Read message size header (first 4 bytes)
            header = client_sock.recv(4)
            if len(header) < 4:
                print("Invalid header received.")
                return
            msglen = int.from_bytes(header, "big")
            while len(data) < msglen:
                part = client_sock.recv(msglen - len(data))
                if not part:
                    break
                data += part
            if not data:
                print("No data received.")
                return
            try:
                request_json = data.decode("utf-8")
                request = json.loads(request_json)
                print(f"Received request: {request}")
                response = self.process_request(request)
                response_json = json.dumps(response).encode("utf-8")
                resp_header = len(response_json).to_bytes(4, "big")
                client_sock.sendall(resp_header + response_json)
            except Exception as e:
                print(f"Error handling client: {e}")

    def process_request(self, request):
        # Validate request schema
        required_keys = {"task-type", "judge-type", "partial-trajectories"}
        if not all(k in request for k in required_keys):
            return {"error": "Invalid request format"}
        # Example: task dispatch
        task_type = request["task-type"]
        judge_type = request["judge-type"]
        partial_trajectories = request["partial-trajectories"]

        result = {
            "task-type": task_type,
            "judge-type": judge_type,
            "steps-count": len(partial_trajectories),
            "status": "received",
            "echo": partial_trajectories,
        }
        return result


# Example usage (remove or modify in actual deployment):
if __name__ == "__main__":
    import time
    server = BaseServer()
    server.start()