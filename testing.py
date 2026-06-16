import numpy as np
from numpy.typing import NDArray
from typing import TypeVar


T = TypeVar("T")


# %%
def test():
    print("1")
    arr: NDArray[np.str_] = np.array(["H", "e"])
    print(arr)


# %%
def main():
    print("Running test")
    test()
    print("Finished running test 1")
    print("Finished running test 2")
    print("Finished running test 3")
    print("Finished running test 4")
    print("Finished running test 5")
    x = [1, 2, 3, 4]

    value = identity(42)  # mypy infers value: int


def identity(x: T) -> T:
    return x


# %%


def add(a: int, b: int) -> int:
    return a + b


# %%
if __name__ == "__main__":
    main()
