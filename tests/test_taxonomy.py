from eda.taxonomy import MICRO_PER_MACRO, composite_label, is_altro_sub, is_in_perimeter


def test_perimeter() -> None:
    assert is_in_perimeter("documentazione_medica", "prescrizione_medica")
    assert not is_in_perimeter("documentazione_medica", "fattura")
    assert not is_in_perimeter("macro_inesistente", "fattura")


def test_composite_label_disambiguates_shared_micro() -> None:
    a = composite_label("documento_di_spesa", "fattura")
    b = composite_label("documento_di_spesa_medica", "fattura")
    assert a != b


def test_is_altro() -> None:
    assert is_altro_sub("documentazione_medica_altro")
    assert not is_altro_sub("prescrizione_medica")


def test_every_macro_has_subs() -> None:
    assert all(len(v) >= 1 for v in MICRO_PER_MACRO.values())
