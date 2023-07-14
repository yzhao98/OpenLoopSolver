import functools
import time
import timeit


class StopWatch:
    def __init__(self, active: bool = True):
        self.active = active
        self.clock = {}

    def start(self, name):
        if self.active:
            self.clock[name] = {"start": time.time()}

    def stop(self, name, report: bool = False, clear: bool = False):
        if self.active:
            self.clock[name]["stop"] = time.time()

        if report:
            self._report(name, clear)

    def clear(self):
        self.clock = {}

    def _report(self, name, clear: bool = False):
        clock = self.clock[name]
        duration = clock["stop"] - clock["start"]
        print(f"{name}: {duration:.3f} seconds")
        if clear:
            self.clock.pop(name)

    def report(self, clear: bool = False):
        for name in self.clock:
            self._report(name, clear)


# A decorator to report the elapsed time of a function
def report_elapsed_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        duration = timeit.default_timer() - start
        print(f"{func.__name__}: {duration:.3f} seconds")
        return result

    return wrapper
