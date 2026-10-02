"""원자적 JSON 저장/로드 유틸리티 (pandas 의존성 없음)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_json(path: Path, default: Any) -> Any:
    """JSON 파일을 읽는다. 파일이 없으면 default를 반환한다."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json_write(path: Path, data: Any) -> None:
    """JSON을 atomic write 방식으로 저장 (임시파일 → rename).

    1) 기존 파일이 있으면 .bak 백업 생성
    2) 같은 디렉토리에 임시파일 생성 → JSON 기록 → fsync
    3) os.replace로 원자적 교체 (중간에 죽어도 파일 손상 없음)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            shutil.copy2(path, path.with_suffix(".json.bak"))
        except Exception as e:
            logger.warning(f"백업 생성 실패 ({path.name}): {e}")

    content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
