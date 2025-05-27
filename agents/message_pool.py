import threading
import queue

class MessagePool:
    def __init__(self):
        self.queue = queue.Queue()

    def post(self, message):
        self.queue.put(message)

    def get(self, block=True, timeout=None):
        try:
            return self.queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def empty(self):
        return self.queue.empty()
