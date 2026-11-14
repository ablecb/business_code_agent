import sys
from datetime import datetime, timezone
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from sqlalchemy import text
from app.db.mysql import get_engine

with get_engine().connect() as c:
    r = c.execute(
        text("SELECT id, status, stage, message, processed, total_files, pid, created_at, updated_at "
             "FROM ingest_job WHERE id='job_e6db81fa8ae64c1d8576489839696d1f'")
    ).fetchone()
    print("id:", r[0])
    print("status:", r[1])
    print("stage:", r[2])
    print("message:", r[3])
    print("processed:", r[4], "/", r[5])
    print("pid:", r[6])
    print("created_at (UTC):", r[7])
    print("updated_at (UTC):", r[8])
    now_utc = datetime.utcnow()
    frozen_sec = (now_utc - r[8]).total_seconds()
    total_sec = (now_utc - r[7]).total_seconds()
    print(f"now UTC: {now_utc.isoformat()}")
    print(f"frozen for: {frozen_sec:.0f}s = {frozen_sec/60:.1f} min = {frozen_sec/3600:.2f} h")
    print(f"total elapsed: {total_sec/60:.1f} min")
