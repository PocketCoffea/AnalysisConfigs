import numpy as np

# import ordered dict
from collections import OrderedDict


def pol0(x, p0):
    return np.full_like(x, p0)

def pol1(x, p0, p1):
    return p0 + p1 * np.log10(x)

def pol2(x, p0, p1, p2):
    return p0 + p1 * np.log10(x) + p2 * np.log10(x) ** 2

def pol3(x, p0, p1, p2, p3):
    return p0 + p1 * np.log10(x) + p2 * np.log10(x) ** 2 + p3 * np.log10(x) ** 3


def pol4(x, p0, p1, p2, p3, p4):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
    )


def pol5(x, p0, p1, p2, p3, p4, p5):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
    )


def pol6(x, p0, p1, p2, p3, p4, p5, p6):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
    )


def pol7(x, p0, p1, p2, p3, p4, p5, p6, p7):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
    )


def pol8(x, p0, p1, p2, p3, p4, p5, p6, p7, p8):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
        + p8 * np.log10(x) ** 8
    )


def pol9(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
        + p8 * np.log10(x) ** 8
        + p9 * np.log10(x) ** 9
    )


def pol10(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
        + p8 * np.log10(x) ** 8
        + p9 * np.log10(x) ** 9
        + p10 * np.log10(x) ** 10
    )


def pol11(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
        + p8 * np.log10(x) ** 8
        + p9 * np.log10(x) ** 9
        + p10 * np.log10(x) ** 10
        + p11 * np.log10(x) ** 11
    )


def pol12(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
        + p8 * np.log10(x) ** 8
        + p9 * np.log10(x) ** 9
        + p10 * np.log10(x) ** 10
        + p11 * np.log10(x) ** 11
        + p12 * np.log10(x) ** 12
    )


def pol13(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
        + p8 * np.log10(x) ** 8
        + p9 * np.log10(x) ** 9
        + p10 * np.log10(x) ** 10
        + p11 * np.log10(x) ** 11
        + p12 * np.log10(x) ** 12
        + p13 * np.log10(x) ** 13
    )


def pol14(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
        + p8 * np.log10(x) ** 8
        + p9 * np.log10(x) ** 9
        + p10 * np.log10(x) ** 10
        + p11 * np.log10(x) ** 11
        + p12 * np.log10(x) ** 12
        + p13 * np.log10(x) ** 13
        + p14 * np.log10(x) ** 14
    )


def pol15(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
        + p8 * np.log10(x) ** 8
        + p9 * np.log10(x) ** 9
        + p10 * np.log10(x) ** 10
        + p11 * np.log10(x) ** 11
        + p12 * np.log10(x) ** 12
        + p13 * np.log10(x) ** 13
        + p14 * np.log10(x) ** 14
        + p15 * np.log10(x) ** 15
    )


def pol16(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
        + p8 * np.log10(x) ** 8
        + p9 * np.log10(x) ** 9
        + p10 * np.log10(x) ** 10
        + p11 * np.log10(x) ** 11
        + p12 * np.log10(x) ** 12
        + p13 * np.log10(x) ** 13
        + p14 * np.log10(x) ** 14
        + p15 * np.log10(x) ** 15
        + p16 * np.log10(x) ** 16
    )


def pol17(
    x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17
):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
        + p8 * np.log10(x) ** 8
        + p9 * np.log10(x) ** 9
        + p10 * np.log10(x) ** 10
        + p11 * np.log10(x) ** 11
        + p12 * np.log10(x) ** 12
        + p13 * np.log10(x) ** 13
        + p14 * np.log10(x) ** 14
        + p15 * np.log10(x) ** 15
        + p16 * np.log10(x) ** 16
        + p17 * np.log10(x) ** 17
    )


def pol18(
    x,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    p16,
    p17,
    p18,
):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
        + p8 * np.log10(x) ** 8
        + p9 * np.log10(x) ** 9
        + p10 * np.log10(x) ** 10
        + p11 * np.log10(x) ** 11
        + p12 * np.log10(x) ** 12
        + p13 * np.log10(x) ** 13
        + p14 * np.log10(x) ** 14
        + p15 * np.log10(x) ** 15
        + p16 * np.log10(x) ** 16
        + p17 * np.log10(x) ** 17
        + p18 * np.log10(x) ** 18
    )


def pol19(
    x,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    p16,
    p17,
    p18,
    p19,
):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
        + p8 * np.log10(x) ** 8
        + p9 * np.log10(x) ** 9
        + p10 * np.log10(x) ** 10
        + p11 * np.log10(x) ** 11
        + p12 * np.log10(x) ** 12
        + p13 * np.log10(x) ** 13
        + p14 * np.log10(x) ** 14
        + p15 * np.log10(x) ** 15
        + p16 * np.log10(x) ** 16
        + p17 * np.log10(x) ** 17
        + p18 * np.log10(x) ** 18
        + p19 * np.log10(x) ** 19
    )


def pol20(
    x,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    p16,
    p17,
    p18,
    p19,
    p20,
):
    return (
        p0
        + p1 * np.log10(x)
        + p2 * np.log10(x) ** 2
        + p3 * np.log10(x) ** 3
        + p4 * np.log10(x) ** 4
        + p5 * np.log10(x) ** 5
        + p6 * np.log10(x) ** 6
        + p7 * np.log10(x) ** 7
        + p8 * np.log10(x) ** 8
        + p9 * np.log10(x) ** 9
        + p10 * np.log10(x) ** 10
        + p11 * np.log10(x) ** 11
        + p12 * np.log10(x) ** 12
        + p13 * np.log10(x) ** 13
        + p14 * np.log10(x) ** 14
        + p15 * np.log10(x) ** 15
        + p16 * np.log10(x) ** 16
        + p17 * np.log10(x) ** 17
        + p18 * np.log10(x) ** 18
        + p19 * np.log10(x) ** 19
        + p20 * np.log10(x) ** 20
    )


pol_functions_dict = OrderedDict(
    {
        0: pol0,
        1: pol1,
        2: pol2,
        3: pol3,
        4: pol4,
        5: pol5,
        6: pol6,
        7: pol7,
        8: pol8,
        9: pol9,
        10: pol10,
        11: pol11,
        12: pol12,
        13: pol13,
        14: pol14,
        15: pol15,
        16: pol16,
        17: pol17,
        18: pol18,
        19: pol19,
        20: pol20,
    }
)
