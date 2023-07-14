import collections
from typing import DefaultDict, Any, Optional, Type, Callable

from manipulator.registry.meta import DataAugmenter, GeneralCostGetter
from manipulator.meta import Singleton


class Registry(metaclass=Singleton):
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    # this decorator works with or without parameter => parametrized decorator
    # e.g. @... or @...(...)

    # To add a family of types to registry, please add a base type in registry.meta, and inherit from it.
    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(_to_register):
            if assert_type is not None:
                assert issubclass(
                    _to_register, assert_type
                ), "{} must be a subclass of {}".format(_to_register, assert_type)
            register_name = _to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = _to_register
            return _to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def _get_impl(cls, _type: str, name: str) -> Type:
        return cls.mapping[_type].get(name, None)

    @classmethod
    def register_data_augmenter(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("data_augmenter", to_register, name, DataAugmenter)

    @classmethod
    def get_data_augmenter(cls, name: str) -> Type[DataAugmenter]:
        _cls = cls._get_impl("data_augmenter", name)
        assert issubclass(_cls, DataAugmenter)
        return _cls

    @classmethod
    def register_cost_model_getter(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        return cls._register_impl(
            "cost_model_getter", to_register, name, GeneralCostGetter
        )

    @classmethod
    def get_cost_model_getter(cls, name: str) -> Type[GeneralCostGetter]:
        _cls = cls._get_impl("cost_model_getter", name)
        assert issubclass(_cls, GeneralCostGetter)
        return _cls


registry = Registry()
