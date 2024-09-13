import pytest
import re
import math
from typing import Optional

from guidance.library._regex_utils import rx_int_range, rx_float_range, float_to_str


def do_test_int_range(rx: str, left: Optional[int], right: Optional[int]) -> None:
    if left is None:
        test_left = -1000
    else:
        test_left = left - 1000
    if right is None:
        test_right = 1000
    else:
        test_right = right + 1000
    for n in range(test_left, test_right):
        m = re.fullmatch(rx, str(n)) is not None
        f = (left is None or left <= n) and (right is None or n <= right)
        assert m == f
        n += 1


@pytest.mark.parametrize(
    ["left", "right"],
    [
        (0, 9),
        (1, 7),
        (0, 99),
        (13, 170),
        (13, 17),
        (13, 27),
        (13, 57),
        (72, 91),
        (723, 915),
        (23, 915),
        (-1, 915),
        (-9, 9),
        (-3, 3),
        (-3, 0),
        (-72, 13),
        (None, 0),
        (None, 7),
        (None, 23),
        (None, 725),
        (None, -1),
        (None, -17),
        (None, -283),
        (0, None),
        (2, None),
        (33, None),
        (234, None),
        (-1, None),
        (-87, None),
        (-329, None),
        (None, None),
    ],
)
def test_int_range(left: Optional[int], right: Optional[int]) -> None:
    rx = rx_int_range(left, right)
    do_test_int_range(rx, left, right)


@pytest.mark.parametrize(
    ["left", "right"],
    [
        (0, 10),
        (-10, 0),
        (0.5, 0.72),
        (0.5, 1.72),
        (0.5, 1.32),
        (0.3245, 0.325),
        (1, 2.34),
        (1.33, 2),
        (1, 10.34),
        (1.33, 10),
        (-1.33, 10),
        (-17.23, -1.33),
        (-1.23, -1.221),
        (-10.2, 45293.9),
    ],
)
def test_float_range(left: float, right: float) -> None:
    print(f"Testing range {left}-{right}")
    rx = rx_float_range(left, right)
    do_test_int_range(rx, math.ceil(left), math.floor(right))
    eps = 0.000001
    for x in [left, right, 0, int(left), int(right)]:
        for off in [0, -eps, eps, 1, -1]:
            n = x + off
            ns = float_to_str(n)
            m = re.fullmatch(rx, ns) is not None
            f = left <= n <= right
            assert m == f
