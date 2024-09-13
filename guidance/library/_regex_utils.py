import math
from typing import Optional, cast


def mk_or(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return "(" + "|".join(parts) + ")"


def num_digits(n: int) -> int:
    return len(str(n))


def rx_int_range(left: Optional[int] = None, right: Optional[int] = None) -> str:
    if left is None and right is None:
        return "0|-?([1-9][0-9]*)"
    if right is None:
        left = cast(int, left)
        if left < 0:
            return mk_or([rx_int_range(left, -1), rx_int_range(0, None)])
        return mk_or(
            [rx_int_range(left, int("9" * num_digits(left))), f"[1-9][0-9]{{{num_digits(left)},}}"]
        )
    if left is None:
        right = cast(int, right)
        if right >= 0:
            return mk_or([rx_int_range(0, right), rx_int_range(None, -1)])
        return "-" + rx_int_range(-right, None)

    assert left <= right
    if left < 0:
        if right < 0:
            return "(-" + rx_int_range(-right, -left) + ")"
        else:
            return "(-" + rx_int_range(0, -left) + "|" + rx_int_range(0, right) + ")"
    else:
        if num_digits(left) == num_digits(right):
            l = str(left)
            r = str(right)
            if left == right:
                return f"({l})"

            lpref = l[:-1]
            lx = l[-1]
            rpref = r[:-1]
            rx = r[-1]

            # 1723-1728 => 172[3-8]
            if lpref == rpref:
                return f"({lpref}[{lx}-{rx}])"

            # general case 723-915 => 72[3-9]|(73-90)[0-9]|91[0-5]
            left_rec = int(lpref)
            right_rec = int(rpref)
            assert left_rec < right_rec
            parts = []

            # optimizations:
            # for 0, we have 720-915 => (72-90)[0-9]|91[0-5]
            if lx != "0":
                left_rec += 1
                parts.append(f"{lpref}[{lx}-9]")
            # for 9 we have 723-919 => 72[3-9]|(73-91)[0-9]
            if rx != "9":
                right_rec -= 1
                parts.append(f"{rpref}[0-{rx}]")

            # the middle can be empty 723-734 => 72[3-9]|73[0-4]
            if left_rec <= right_rec:
                inner = rx_int_range(left_rec, right_rec)
                parts.append(f"{inner}[0-9]")

            return mk_or(parts)
        else:
            break_point = 10 ** num_digits(left) - 1
            return mk_or([rx_int_range(left, break_point), rx_int_range(break_point + 1, right)])


def lexi_x_to_9(x: str) -> str:
    if x == "":
        return "[0-9]*"
    if len(x) == 1:
        return f"[{x}-9][0-9]*"
    x0 = int(x[0])
    parts = [x[0] + lexi_x_to_9(x[1:])]
    if x0 < 9:
        parts.append(f"[{x0 + 1}-9][0-9]*")
    return mk_or(parts)


def lexi_0_to_x(x: str) -> str:
    if x == "":
        return ""  # don't allow trailing zeros
    x0 = int(x[0])
    parts = [x[0] + lexi_0_to_x(x[1:])]
    if x0 > 0:
        parts.append(f"[0-{x0 - 1}][0-9]*")
    return mk_or(parts)


def lexi_range(ld: str, rd: str) -> str:
    assert len(ld) == len(rd)
    if ld == rd:
        return ld
    l0 = int(ld[0])
    r0 = int(rd[0])
    # common prefix: 137-144 => 1(37-44)
    if r0 == l0:
        return ld[0] + lexi_range(ld[1:], rd[1:])
    assert l0 < r0
    # 23470-82142 => 2(347-999)|[3-7][0-9]*|8(0000-2142)
    parts = [ld[0] + lexi_x_to_9(ld[1:].rstrip("0"))]
    # is the [3-7][0-9]* part empty?
    if l0 + 1 < r0:
        parts.append(f"[{l0 + 1}-{r0 - 1}][0-9]*")
    parts.append(rd[0] + lexi_0_to_x(rd[1:].rstrip("0")))
    return mk_or(parts)


def float_to_str(f: float) -> str:
    s = f"{f:f}"
    return s.rstrip("0").rstrip(".")


def rx_float_range(left: float, right: float) -> str:
    assert left <= right
    if left < 0:
        if right < 0:
            return "(-" + rx_float_range(-right, -left) + ")"
        else:
            return "(-" + rx_float_range(0.0, -left) + "|" + rx_float_range(0.0, right) + ")"
    else:
        l = float_to_str(left)
        r = float_to_str(right)
        if left == right:
            return f"({l})"

        if "e" in l or "e" in r:
            raise ValueError("Scientific notation not supported")
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Infinite numbers not supported")

        left_rec = int(l.split(".")[0])
        right_rec = int(r.split(".")[0])

        ld = (l.split(".") + [""])[1]
        rd = (r.split(".") + [""])[1]

        # 17.123-17.1448 -> 17.((1230-1447)[0-9]*|1448)
        if left_rec == right_rec:
            while len(ld) < len(rd):
                ld += "0"
            while len(rd) < len(ld):
                rd += "0"
            suff = "\\." + lexi_range(ld, rd)
            if int(ld) == 0:
                return f"({left_rec}({suff})?)"
            else:
                return f"({left_rec}{suff})"

        parts = []

        # 7.321-22.123 -> 7.(321-999)|8-21(.[0-9]+)?|22.(000-123)
        if ld:
            parts.append(f"({left_rec}\\.{lexi_x_to_9(ld)})")
            left_rec += 1

        if right_rec - 1 >= left_rec:
            inner = rx_int_range(left_rec, right_rec - 1)
            parts.append(f"({inner}(\\.[0-9]+)?)")

        if rd:
            parts.append(f"({right_rec}(\\.{lexi_0_to_x(rd)})?)")
        else:
            parts.append(f"{right_rec}")

        return mk_or(parts)
