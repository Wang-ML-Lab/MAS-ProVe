from ..clients.client_base import BaseClient
import asyncio
import json

class MASBase:
    def __init__(self):
        self.trajectory = []
        self.trajectory_lock = asyncio.Lock()

    @property
    def progress(self):
        return json.dumps({f"step_{i+1}": step for i, step in enumerate(self.trajectory)}, indent=4)

    @staticmethod
    def update_trajectory(func):
        """
        Appends the latest step/result to the trajectory and updates the progress.
        Should be called automatically by wrappers of generation calls.
        """
        async def wrapper(self, *args, **kwargs):
            # protocol: if the result is a tuple: first one should be the actual reasoning response.
            result = await func(self, *args, **kwargs)
            if isinstance(result, tuple) or isinstance(result, list):
                result = result[0]
            # protocol: if the result is a dict, it should be converted to a string.
            if isinstance(result, dict):
                result = json.dumps(result)
            
            async with self.trajectory_lock:
                self.trajectory.append(result)
                # check the current progress.
                print(self.progress)
            return result
        return wrapper

    def run(self): 
        raise NotImplementedError("Subclasses must implement this method")
