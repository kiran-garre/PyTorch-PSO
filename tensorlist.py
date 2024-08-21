import torch

"""
The TensorList class simplifies operations between lists of torch Tensor objects.
"""
class TensorList:
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.shape = []
        for t in self.tensors:
            self.shape.append(t.shape)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        return self.tensors[index]

    def __add__(self, other):
        return self._check_and_generate_output(other, torch.add)

    def __sub__(self, other):
        return self._check_and_generate_output(other, torch.sub)

    def __mul__(self, other):
        return self._check_and_generate_output(other, torch.mul)

    def __radd__(self, other):      # operations are commutative; only called when the other operand is a scalar
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def _check_and_generate_output(self, other, fn):
        output = []
        if isinstance(other, TensorList):
            assert len(self) == len(other), "Operands must contain the same number of tensors"
            for a, b in zip(self.tensors, other.tensors):
                output.append(fn(a, b))
            return TensorList(output)
        else:
            for a in self.tensors:
                output.append(fn(a, other))
            return TensorList(output)

    def multi_copy_(self, other):
        for a, b in zip(self, other):
            a.copy_(b)

    @staticmethod
    def zeros_like(other):
        return TensorList._apply_fn(other, torch.zeros_like)

    @staticmethod
    def empty_like(other):
        return TensorList._apply_fn(other, torch.empty_like)

    @staticmethod
    def uniform_like(other):
        return TensorList._apply_fn(other, torch.rand_like)

    @staticmethod
    def normal_like(other):
        return TensorList._apply_fn(other, torch.randn_like)

    @staticmethod
    def clone(other):
        return TensorList._apply_fn(other, torch.clone)

    @staticmethod
    def abs(other):
        return TensorList._apply_fn(other, torch.abs)

    def __abs__(self):
        return TensorList.abs(self)

    @staticmethod
    def clip(other, min, max):
        return TensorList._apply_fn(other, torch.clamp, min, max)

    @staticmethod
    def _apply_fn(other, fn, *args):
        lst = []
        for t in other:
            lst.append(fn(t, *args))
        return TensorList(lst)
    
    def equals(self, other):
        if len(other) != len(self):
            return False
        for i in range(len(self)):
            if not other[i].equal(self[i]):
                return False
        return True

# Takes both list[Tensor] and TensorList
def multi_copy(list1, list2):
    for a, b in zip(list1, list2):
        a.copy_(b)
