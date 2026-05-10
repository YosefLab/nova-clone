import os
import pickle
from typing import Any, List, Tuple
from inspect import signature
from dataclasses import dataclass

@dataclass
class CachedResult:
    result: dict
    cache_path: str

def get_argnames_and_values(func, *args, **kwargs) -> List[Tuple[str, Any]]:
    """
    Get the argnames and values of the function
    """
    binding = signature(func).bind(*args, **kwargs)
    binding.apply_defaults()
    argnames_and_values = list(binding.arguments.items())
    return argnames_and_values

def caching_decorator(exclude_args: List[str] = []):
    """
    Decorator to cache the result of the functions in evaluate.py
    """
    def caching_without_specific_args(func):
        def eval_function_caching(*args, **kwargs):
            argnames_and_values = get_argnames_and_values(func, *args, **kwargs)
            cached_result_path = f"_cache/{func.__name__}/" + "/".join(f"{argname}_{val}" for argname, val in argnames_and_values if argname not in exclude_args) + "/result.pickle"
            if os.path.exists(cached_result_path):
                with open(cached_result_path, "rb") as f:
                    return CachedResult(result=pickle.load(f), cache_path=cached_result_path)
            else:
                result = func(*args, **kwargs)
                os.makedirs(f"_cache/{func.__name__}/" + "/".join(f"{argname}_{val}" for argname, val in argnames_and_values if argname not in exclude_args), exist_ok=True)
                with open(cached_result_path, "wb") as f:
                    pickle.dump(result, f)
                    f.flush()
                return CachedResult(result=result, cache_path=cached_result_path)
        return eval_function_caching
    return caching_without_specific_args