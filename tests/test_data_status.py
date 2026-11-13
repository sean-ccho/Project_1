"""자료 점검 (web_export/data_status.py) — 칸 세기 · 색 규칙 · 지난 밤 비교 · 오래된 파일 · 알림 글 (네트워크 없음)."""
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web_export import data_status as ds  # noqa: E402

NOW = datetime(2026, 11, 13, 20, 10, tzinfo=ds.ET)


def _item(price=1.0, isb=None, sqz=None, mcap=1e9, insider=None):
    return {"price": price, "isb": {} if isb is None else isb, "sqzmom": {} if sqz is None else sqz,
            "fundamentals": {"market_cap": mcap}, "insider": {} if insider is None else insider}


def _rows(rows):
    return {r["key"]: r for r in rows}


def test_stock_rows_count_missing_fields() -> None:
    xs = [_item(), _item(), {**_item(), "isb": None, "sqzmom": None}, {**_item(), "fundamentals": {}, "insider": None}]
    r = _rows(ds.stock_rows(xs, insider=True))
    assert (r["items"]["got"], r["price"]["got"], r["isb"]["got"], r["sqz"]["got"]) == (4, 4, 3, 3)
    assert (r["mcap"]["got"], r["insider"]["got"], r["isb"]["want"]) == (3, 3, 4)
    assert r["items"].get("zero_red") and "want" not in r["items"]
    assert "insider" not in _rows(ds.stock_rows(xs, insider=False))  # 캐나다 · 번외는 내부자를 안 모은다


def test_row_level_rules() -> None:
    assert ds.row_level({"got": 0, "want": 513}) == "red"                    # 11-12 밤 예측 모델
    assert ds.row_level({"got": 90, "want": 100, "floor": 0.95}) == "yellow"
    assert ds.row_level({"got": 96, "want": 100, "floor": 0.95}) == "green"
    assert ds.row_level({"got": 0, "zero_red": True}) == "red"                # 종목 0 개
    assert ds.row_level({"got": 0}) == "green"                                # 0 이어도 되는 칸 (지난 밤 비교 없음)
    assert ds.row_level({"got": 70, "prev": 100}) == "yellow"                 # 지난 밤보다 30% 줄어듦
    assert ds.row_level({"got": 85, "prev": 100}) == "green"
    assert ds.file_level({"rows": [{"got": 1, "want": 1}, {"got": 1, "want": 2, "floor": 0.95}]}) == "yellow"
    assert ds.file_level({"missing": True, "rows": []}) == "red"
    assert ds.file_level({"stale": True, "rows": [{"got": 1, "want": 1}]}) == "red"


def _write(path: Path, data, gz: bool = False) -> None:
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(data, fh)
    else:
        path.write_text(json.dumps(data), encoding="utf-8")


def _market(n: int, when: str = "2026-11-13T19:56:00-05:00", **kw):
    return {"bar_date": "2026-11-13", "generated_at": when, "items": {f"T{i}": _item(**kw) for i in range(n)}}


def test_build_status_files_prev_stale_and_alert(tmp_path: Path) -> None:
    u = tmp_path / "universe"
    u.mkdir()
    _write(u / "sp500.json", _market(5))
    _write(u / "nasdaq.json.gz", _market(4), gz=True)  # .json 이 없으면 .json.gz
    _write(u / "new.json", _market(2))
    _write(u / "value_model.json", {"date": "2026-11-13", "generated": "2026-11-13T20:05-05:00",
                                    "funnel": {"universe": 513, "computed": 0}, "tracking": {"warmup": None}})
    _write(u / "congress.json", {"as_of": "2026-11-13", "generated_at": "2026-11-13T18:30:00-05:00",
                                 "recent": [{}] * 10})
    ca = tmp_path / "ca.json.gz"
    _write(ca, _market(3, when="2026-11-12T18:35:00-05:00"), gz=True)  # 어제 저녁 파일 — Canada Scan 실패
    prev = {"files": {"nasdaq": {"rows": [{"key": "items", "got": 10}]}, "congress": {"rows": [{"key": "recent", "got": 9}]}}}
    st = ds.build_status(u, ca, prev, now=NOW)
    f = st["files"]
    assert set(f) == {"sp500", "nasdaq", "new", "ca", "value_model", "congress"}  # 번외 파일이 없으면 칸도 없다
    assert st["date"] == "2026-11-13"
    assert ds.file_level(f["sp500"]) == "green"
    assert _rows(f["nasdaq"]["rows"])["items"]["prev"] == 10 and ds.file_level(f["nasdaq"]) == "yellow"  # 10 → 4
    assert "prev" not in _rows(f["nasdaq"]["rows"])["price"]  # want 있는 칸은 비율로만
    assert f["ca"].get("stale") and ds.file_level(f["ca"]) == "red"
    assert ds.file_level(f["value_model"]) == "red" and ds.file_level(f["congress"]) == "green"
    title, body = ds.alert(st)
    assert title == "자료 점검 · 실패 2곳"
    assert "예측 모델: 계산된 종목 0/513" in body and "전체분석 · 캐나다: 이번 밤 파일 아님" in body


def test_value_model_warmup_and_missing_file(tmp_path: Path) -> None:
    vm = {"funnel": {"universe": 2060, "computed": 1300}, "tracking": {"warmup": {"have": 1600, "want": 1957}}}
    r = _rows(ds.value_model_rows(vm))
    assert ds.row_level(r["computed"]) == "green" and ds.row_level(r["warmup"]) == "yellow"  # 기록 준비 중 = 일부만
    st = ds.build_status(tmp_path, None, None, now=NOW)  # 아무 파일도 없음
    assert all(f.get("missing") for f in st["files"].values()) and "extra" not in st["files"]
    assert ds.alert({"files": {"x": {"label": "x", "rows": [{"key": "a", "label": "a", "got": 1, "want": 1}]}}}) is None


def test_main_writes_file_without_notify(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path / "sp500.json", _market(2))
    sent = []
    monkeypatch.setattr(ds, "send_ntfy", lambda *a: sent.append(a) or True)
    monkeypatch.setenv("NTFY_TOPIC", "t")
    assert ds.main(["--universe-dir", str(tmp_path), "--no-notify"]) == 0
    out = json.loads((tmp_path / ds.OUT_NAME).read_text(encoding="utf-8"))
    assert out["schema"] == 1 and "sp500" in out["files"] and sent == []
    assert ds.main(["--universe-dir", str(tmp_path)]) == 0  # 예측 모델 · 캐나다 파일 없음 → 실패 → 알림
    assert sent and sent[0][1].startswith("자료 점검 · 실패")
