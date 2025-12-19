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
            # Get the original result from the decorated function
            result = await func(self, *args, **kwargs)
            
            # Extract content for trajectory while preserving original result
            trajectory_content = result
            if isinstance(result, tuple) or isinstance(result, list):
                trajectory_content = result[0]
            # Convert dict to string for trajectory storage
            if isinstance(trajectory_content, dict):
                trajectory_content = json.dumps(trajectory_content)
            
            async with self.trajectory_lock:
                self.trajectory.append(trajectory_content)
                # check the current progress.
                print(self.progress)
            
            # Return the ORIGINAL result unchanged
            return result
        return wrapper

    def run(self): 
        raise NotImplementedError("Subclasses must implement this method")
