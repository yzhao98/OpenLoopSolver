from collections.abc import Mapping
from typing import Iterator


class LazyDict(Mapping):
    mark: str = "lazy"

    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)

    def __getitem__(self, __k):
        v = self._dict[__k]
        try:
            mark = v[0]
            if mark == self.mark:
                func, args, kwargs = v[1:]
                v = func(*args, **kwargs)
                self._dict[__k] = v
        except (IndexError, TypeError):
            pass

        return v

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator:
        return iter(self._dict)
