"""Debug entry_route fallback api_doc mapping."""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import or_, select

from app.db.mysql import get_session
from app.models.orm import SymbolDescORM, SymbolORM


def main() -> None:
    with get_session() as session:
        rows = list(
            session.scalars(
                select(SymbolORM).where(
                    SymbolORM.project_id == "demo-loan",
                    or_(
                        SymbolORM.kind.in_(["Controller", "Job", "Listener", "Entry"]),
                        SymbolORM.simple_name.like("%Controller"),
                    ),
                ).limit(20)
            )
        )
        print("sample symbols:")
        for r in rows[:10]:
            print(f"  id={r.id!r} simple={r.simple_name!r} meth={r.method_name!r}")
        desc_rows = session.execute(
            select(SymbolDescORM).where(SymbolDescORM.project_id == "demo-loan").limit(20)
        ).all()
        print("\nsample descs:")
        for d in desc_rows[:10]:
            extra = d.extra_json if isinstance(d.extra_json, dict) else {}
            api = str(extra.get("api_doc") or "")[:30]
            print(f"  symbol_id={d.symbol_id!r} api={api!r}")


if __name__ == "__main__":
    main()
