import numpy as np
import pytest

BLUE_BGR  = (180, 80, 30)
BLACK_BGR = (20, 20, 20)
BROWN_BGR = (40, 70, 110)


def build_person(upper_bgr, lower_bgr=BLACK_BGR):
    """Synthetic 200x80 person crop: shirt band then pants band."""
    p = np.zeros((200, 80, 3), dtype=np.uint8)
    p[36:100, :]  = upper_bgr
    p[100:164, :] = lower_bgr
    return p


@pytest.fixture
def blue_black_person():
    return build_person(BLUE_BGR, BLACK_BGR)
