from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


class ConcurrentWorker:
    """This class is a wrapper for ThreadPoolExecutor and ProcessPoolExecutor.
    It is used to distribute jobs to multiple threads or processes.
    For simplicity, it is designed for those jobs that have no return value, e.g. these jobs save the results to disk.
    Users do not need to wait for the result.
     When the object is about to be deleted, it will automatically shutdown the executor.
    Note that: if one job failed, it will not stop the whole process but failed silently.
     It is the user's responsibility to ensure that the jobs are correct.
    """

    def __init__(self, max_workers: int = 4, processor: bool = False):
        if processor:
            self.executors = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executors = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, func, /, *args, **kwargs):  # func is positional only
        return self.executors.submit(func, *args, **kwargs)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Shutdown can be executed multiple times."""
        self.executors.shutdown(wait=True)

    @classmethod
    def create(cls, max_workers: int = 4, processor: bool = False):
        return cls(max_workers, processor)
