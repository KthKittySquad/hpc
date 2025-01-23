import pytest
from JuliaSet import calc_pure_python


@pytest.mark.parametrize('width, iterations, expected', [(1000, 300, 33219980), (10, 3, 177)])
def test_calc_pure_python(width, iterations, expected):
    output = calc_pure_python(width, iterations)
    assert sum(output) == expected
