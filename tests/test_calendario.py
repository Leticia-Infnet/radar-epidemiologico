from datetime import date

from etl.transform.calendario import calendario, inicio_ano_epi, semanas_no_ano


def test_inicio_do_ano_epidemiologico():
    assert inicio_ano_epi(2024) == date(2023, 12, 31)
    assert inicio_ano_epi(2020) == date(2019, 12, 29)
    assert inicio_ano_epi(2021) == date(2021, 1, 3)


def test_anos_com_53_semanas():
    assert [a for a in range(2010, 2027) if semanas_no_ano(a) == 53] == [2014, 2020, 2025]


def test_calendario_continuo():
    cal = calendario(2019, 2021)
    assert len(cal) == 52 + 53 + 52
    assert (cal["inicio"].diff().dropna().dt.days == 7).all()
