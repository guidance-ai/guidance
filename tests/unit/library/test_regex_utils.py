import pytest
import re
import math
from typing import Optional

from guidance.library._regex_utils import rx_int_range, rx_float_range, float_to_str


def do_test_int_range(rx: str, left: int, right: int, left_none: bool = False, right_none: bool = False) -> None:
    for n in range(left - 1000, right + 1000):
        m = re.fullmatch(rx, str(n)) is not None
        f = (left_none or left <= n) and (right_none or n <= right)
        assert m == f

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
        (-13, -13),
        (-1, -1),
        (0, 0),
        (1, 1),
        (13, 13),
    ],
)
def test_int_range(left: Optional[int], right: Optional[int]) -> None:
    rx = rx_int_range(left, right)
    do_test_int_range(
        rx,
        (left or 0),
        (right or 0),
        left_none=(left is None),
        right_none=(right is None),
    )


@pytest.mark.parametrize(
    "left_inclusive",
    [True, False],
)
@pytest.mark.parametrize(
    "right_inclusive",
    [True, False],
)
@pytest.mark.parametrize(
    ["left", "right"],
    [
        (0, 10),
        (-10, 0),
        (0.5, 0.72),
        (0.5, 1.72),
        (0.5, 1.32),
        (0.45, 0.5),
        (0.3245, 0.325),
        (0.443245, 0.44325),
        (1, 2.34),
        (1.33, 2),
        (1, 10.34),
        (1.33, 10),
        (-1.33, 10),
        (-17.23, -1.33),
        (-1.23, -1.221),
        (-10.2, 45293.9),
        (None, 0),
        (None, 1),
        (None, 1.5),
        (None, 1.55),
        (None, -17.23),
        (None, -1.33),
        (None, -1.23),
        (None, 103.74),
        (None, 100),
        (0, None),
        (1, None),
        (1.5, None),
        (1.55, None),
        (-17.23, None),
        (-1.33, None),
        (-1.23, None),
        (103.74, None),
        (100, None),
        (None, None),
        (-103.4, -103.4),
        (-27, -27),
        (-1.5, -1.5),
        (-1, -1),
        (0, 0),
        (1, 1),
        (1.5, 1.5),
        (27, 27),
        (103.4, 103.4),
    ],
)
def test_float_range(
    left: Optional[float], right: Optional[float], left_inclusive: bool, right_inclusive: bool
) -> None:
    l = "[" if left_inclusive else "("
    r = "]" if right_inclusive else ")"
    print(f"Testing range {l}{left}-{right}{r}")

    if (left is not None) and (left == right) and not (left_inclusive and right_inclusive):
        with pytest.raises(ValueError):
            rx_float_range(left, right, left_inclusive, right_inclusive)
    else:
        rx = rx_float_range(left, right, left_inclusive, right_inclusive)
        do_test_float_range(rx, (left or 0), (right or 0), left_inclusive, right_inclusive, left is None, right is None)

def do_test_float_range(rx, left: float, right: float, left_inclusive: bool = True, right_inclusive: bool = True, left_none: bool = False, right_none: bool = False):
    left_int = math.ceil(left)
    right_int = math.floor(right)
    if left_int < right_int:
        if not left_inclusive and left_int == left:
            left_int += 1
        if not right_inclusive and right_int == right:
            right_int -= 1
        do_test_int_range(rx, left_int, right_int, left_none, right_none)
    eps = 0.000001
    eps2 = 0.01
    test_cases = [left, right, 0, int(left), int(right)]
    if left_none:
        test_cases.append(left - 1000)
    if right_none:
        test_cases.append(right + 1000)
    for x in test_cases:
        for off in [0, -eps, eps, -eps2, eps2, 1, -1]:
            n = x + off
            ns = float_to_str(n)
            m = re.fullmatch(rx, ns) is not None
            lcond = left_none or left < n or (left == n and left_inclusive)
            rcond = right_none or right > n or (right == n and right_inclusive)
            f = lcond and rcond
            if m != f:
                print(f"Failed float for {ns} match={m} expected={f}")
                assert False
