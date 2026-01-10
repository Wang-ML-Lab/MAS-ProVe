import socket
import threading
import json
import backoff


class BaseClient:
    def __init__(self, host='127.0.0.1', port=5555, max_retries=3, initial_backoff=0.1, max_backoff=5.0):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

    def _send_request_internal(self, msglen, data):
        """Internal method that performs the actual socket operation with retry logic."""
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

    def send_request(self, task_type, judge_type, partial_trajectories, question=""):
        request = {
            "task-type": task_type,
            "judge-type": judge_type,
            "partial_trajectories": partial_trajectories,
            "question": question,
        }
        data = json.dumps(request).encode("utf-8")
        msglen = len(data).to_bytes(4, "big")

        # Apply backoff decorator with instance-specific parameters
        @backoff.on_exception(
            backoff.expo,
            (ConnectionRefusedError, ConnectionError,
             OSError, socket.timeout, socket.error),
            max_tries=self.max_retries,
            base=self.initial_backoff,
            max_value=self.max_backoff,
            on_backoff=lambda details: print(
                f"Connection attempt {details['tries']} failed: {details['exception']}. Retrying in {details['wait']:.2f} seconds...")
        )
        def send_with_retry(msglen, data):
            return self._send_request_internal(msglen, data)

        return send_with_retry(msglen, data)


if __name__ == "__main__":
    client = BaseClient()
    response = client.send_request(
        task_type="math",
        judge_type="judge",
        question="What is 2+2?",
        partial_trajectories=[
            {"context": "2+2 is obviously 4", "current-step": "Returning answer"},
            {"context": "Let me calculate: 2+2 = 4",
                "current-step": "Calculating sum"},
            {"context": "The sum of 2 and 2 equals 5... wait, that's wrong. It's 4.",
                "current-step": "Correcting calculation"},
        ]
    )
    print("Client received response:")
    print("Rankings:", response.get("rankings"))
    print("\nJudge feedback:")
    print(response.get("judge-feedback"))
