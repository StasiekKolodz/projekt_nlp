import threading

class MessagePool:
    def __init__(self):
        self.messages = []
        self.lock = threading.Lock()

    def build_message(self, msg_type, content):
        return {"msg_type": msg_type, "content": content}

    def post(self, message):
        with self.lock:
            self.messages.append(message)

    def get_all(self):
        with self.lock:
            return list(self.messages)

    def find(self, predicate):
        with self.lock:
            return [msg for msg in self.messages if predicate(msg)]

    def remove_type(self, msg_type):
        with self.lock:
            self.messages = [msg for msg in self.messages if msg["msg_type"] != msg_type]

    def __len__(self):
        with self.lock:
            return len(self.messages)
