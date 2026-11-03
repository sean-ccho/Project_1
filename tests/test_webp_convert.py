"""차트 캡처 → 사이트용 WebP — 차트분석은 사본(PNG 남김), 전체차트는 바꿈(PNG 지움), 옛 사본 정리, 실패해도 멈추지 않음."""
import sys
from pathlib import Path

from PIL import Image

import charts.webp_convert as wc


def _png(path: Path, w: int = 300, h: int = 200) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (20, 30, 40)).save(path)
    return path


def test_copy_and_replace(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    sp = _png(Path("charts/screenshots_sp500/AAA/AAA_Daily_20261103_074630.png"))
    nq = _png(Path("charts/screenshots_nasdaq/BBB/BBB_Weekly_20261103_080000.png"))
    al = _png(Path("charts/screenshots_all/CCC/CCC_Daily_20261103_115955.png"))
    stale = Path("charts/screenshots_sp500/AAA/AAA_Daily_20261102_074630.webp")  # 지난 캡처의 사본 (PNG 없음)
    stale.write_bytes(b"old")

    assert wc.main(["--workers", "1"]) == 0
    for png in (sp, nq):  # 차트분석: PNG 그대로 (시트 · 메일) + 같은 이름 사본
        assert png.exists() and png.with_suffix(".webp").exists()
    assert not al.exists() and al.with_suffix(".webp").exists()  # 전체차트: WebP 만
    assert not stale.exists()
    with Image.open(sp.with_suffix(".webp")) as im:
        assert im.format == "WEBP" and im.size == (300, 200)  # 크기 그대로
    assert "3장 변환" in capsys.readouterr().out
    assert wc.plan() == []  # 다시 돌리면 할 일 없음 (사본이 PNG 보다 새로움)


def test_broken_png_keeps_going(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    bad = Path("charts/screenshots_all/DDD/DDD_Daily_20261103_120000.png")
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a png")
    good = _png(Path("charts/screenshots_all/EEE/EEE_Daily_20261103_120000.png"))

    assert wc.main(["--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "실패 1" in out and "::warning title=차트 WebP 일부 실패" in out
    assert bad.exists()  # 못 바꾼 건 PNG 로 남아 사이트가 그대로 쓴다
    assert good.with_suffix(".webp").exists()


def test_without_pillow_only_warns(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    png = _png(Path("charts/screenshots_all/FFF/FFF_Daily_20261103_120000.png"))
    monkeypatch.setitem(sys.modules, "PIL", None)  # import PIL → ImportError
    assert wc.main(["--workers", "1"]) == 0
    assert "::warning title=차트 WebP 변환 못 함" in capsys.readouterr().out
    assert png.exists()
