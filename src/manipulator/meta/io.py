import abc


class SaveLoad:
    """This class defines save and load which handles both file path and file buffer.
    It leaves the detailed implementation to save_impl and load_impl.
    """

    @abc.abstractmethod
    def save_impl(self, buffer, *args, **kwargs):
        pass

    def save(self, save_path_or_buffer, *args, **kwargs) -> None:
        if hasattr(save_path_or_buffer, "write"):
            self.save_impl(save_path_or_buffer, *args, **kwargs)
        else:
            with open(save_path_or_buffer, "wb") as f:
                self.save_impl(f, *args, **kwargs)

    @classmethod
    @abc.abstractmethod
    def load_impl(cls, buff, *args, **kwargs):
        pass

    @classmethod
    def load(cls, load_path_or_buffer, *args, **kwargs):
        if hasattr(load_path_or_buffer, "read") and hasattr(
            load_path_or_buffer, "readline"
        ):
            return cls.load_impl(load_path_or_buffer, *args, **kwargs)
        else:
            with open(load_path_or_buffer, "rb") as f:
                return cls.load_impl(f, *args, **kwargs)
