import inspect
from typing import Union, Optional, List, Any

import torch as th


def th_cat(tensor_lst: List[th.Tensor], dim: int = 0):
    if len(tensor_lst) == 0:
        return th.tensor([])
    return th.cat(tensor_lst, dim=dim)


def torch_uniform_sample_scalar(min_value: float, max_value: float):
    assert max_value >= min_value, f'{max_value=} is smaller than {min_value=}'
    if max_value == min_value:
        return min_value
    return min_value + (max_value - min_value) * th.rand(1).item()


def clamp(value: Union[int, float], smallest: Union[int, float], largest: Union[int, float]):
    return max(smallest, min(value, largest))


def subsample_list(lst: List[Any], num: int, offset: Optional[int] = 0) -> List[Any]:
    """Sample `num` items from a `lst` by taking every k-th element."""
    assert len(lst) >= num, f'{len(lst)=} is smaller than {num=}'
    assert num >= 1, f'{num=}'
    k = len(lst) // num
    return lst[offset::k][:num]


def list2d_to_list1d(lst_2d: Union[List[List], List, Any]) -> List:
    """Flatten a list of list to a 1D list.
    Should also record the original shape so that we can unflatten it later.
    """
    if not (isinstance(lst_2d, list) and isinstance(lst_2d[0], list)):
        return lst_2d, None
    lst_lens = [len(lst) for lst in lst_2d]
    lst_1d = [item for lst in lst_2d for item in lst]
    return lst_1d, lst_lens


def list1d_to_list2d(lst_1d: Union[List, Any], lst_lens: Optional[List[int]] = None) -> List[List]:
    """Unflatten a 1D list to a list of list."""
    if lst_lens is None or not isinstance(lst_lens, list):
        return lst_1d
    lst_2d = []
    start_idx = 0
    for lst_len in lst_lens:
        lst_2d.append(lst_1d[start_idx:start_idx + lst_len])
        start_idx += lst_len
    return lst_2d


def temporal_wrapper(func):
    """A wrapper to make the model compatible with List or List[List] inputs.

    The wrapped function is applied individually to each element of the list.
    Sometimes the variable is of shape [L, B], sometimes just [B].
    Here we will flatten them to 1D ([L*B] or just [B]), apply the function,
      and then unflatten them back to the original shape.
    """

    def f(*args, **kwargs):
        """x is either list of list of data, or list of data.
        if x is not a list, then we don't do anything to it.
        Same rule applies to the function output.
        """
        # Get the signature of the target function
        signature = inspect.signature(func)
        # Bind the arguments passed to the target function
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        # Extract the argument values
        params_2d = bound_args.arguments.values()

        # flatten the input if applicable
        params_1d, lst_lens = [], []
        for param_2d in params_2d:
            param_1d, _lst_lens = list2d_to_list1d(param_2d)
            params_1d.append(param_1d)
            if _lst_lens is not None:
                lst_lens.append(_lst_lens)

        # Perform operations on the parameters
        outputs_1d = func(*params_1d)
        if not isinstance(outputs_1d, tuple):
            outputs_1d = (outputs_1d,)

        # unflatten the output if applicable
        outputs_2d, idx = [], 0
        for output_1d in outputs_1d:
            if isinstance(output_1d, list) and len(lst_lens) > 0:
                _lst_lens = lst_lens[idx]
                idx += 1
            else:
                _lst_lens = None
            output_2d = list1d_to_list2d(output_1d, _lst_lens)
            outputs_2d.append(output_2d)

        assert idx == len(lst_lens) or idx == 0, f'{idx=} != {len(lst_lens)=}'
        return tuple(outputs_2d) if len(outputs_2d) > 1 else outputs_2d[0]

    return f


def del_dirs(pattern: str):
    """Delete all pattern-matched dirs if its creator is the current user."""
    # e.g. del_dirs('datasets/pseudo_gen1/*/*tflip*')
    import os
    import pwd
    import glob
    usr = pwd.getpwuid(os.getuid())[0]
    dirs = glob.glob(pattern)
    for d in dirs:
        if not os.path.exists(d):
            continue
        if os.stat(d).st_uid == pwd.getpwnam(usr).pw_uid:
            os.system(f'rm -rf {d}')
            print(f'Deleted dir: {d}')


def replace_link(pattern: str):
    """Replace soft-link src."""
    # e.g. replace_link('datasets/pseudo_gen*/*/*')
    import os
    import pwd
    import glob
    usr = pwd.getpwuid(os.getuid())[0]
    dirs = glob.glob(pattern)
    for d in dirs:
        if not os.path.exists(d):
            continue
        if os.stat(d).st_uid == pwd.getpwnam(usr).pw_uid:
            val_dst = os.path.join(d, 'val')
            test_dst = os.path.join(d, 'test')
            if not os.readlink(val_dst).startswith('/h/lichothu/'):
                assert not os.readlink(test_dst).startswith('/h/lichothu/')
                continue
            print('Before:', os.system(f'ls -hl {d}'))
            assert os.readlink(test_dst).startswith('/h/lichothu/')
            os.system(f'rm -f {val_dst}')
            os.system(f'rm -f {test_dst}')
            dst = 'gen1' if 'gen1' in d else 'gen4'
            os.system(f'ln -s /datasets/automotive-megapixel/{dst}/val/ {val_dst}')
            os.system(f'ln -s /datasets/automotive-megapixel/{dst}/test/ {test_dst}')
            print('After:', os.system(f'ls -hl {d}'))


def extract_weight(fn):
    """Extract model weight only from a saved checkpoint."""
    import torch
    ckpt = torch.load(fn, map_location='cpu')
    torch.save(ckpt['state_dict'], fn)
