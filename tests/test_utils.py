from ..processing.extended_enum import ExtendedEnum


class DummyEnum(ExtendedEnum):
    A = 1
    B = 2
    C = 3


def test_by_value():
    assert DummyEnum.by_value(1) == DummyEnum.A
    assert DummyEnum.by_value(2) == DummyEnum.B
    assert DummyEnum.by_value(42) is None


def test_from_name():
    assert DummyEnum.from_name("a") == DummyEnum.A
    assert DummyEnum.from_name("B") == DummyEnum.B
    assert DummyEnum.from_name("c") == DummyEnum.C
    assert DummyEnum.from_name("notfound") is None


def test_get_members():
    members = DummyEnum.get_members()
    assert set(members) == {DummyEnum.A, DummyEnum.B, DummyEnum.C}


def test_get_values():
    values = DummyEnum.get_values()
    assert set(values) == {1, 2, 3}
