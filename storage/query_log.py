from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Generator

from app.config import settings

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS queries (
    query_id         TEXT PRIMARY KEY,
    transcript       TEXT NOT NULL,
    lang             TEXT NOT NULL DEFAULT 'en',
    routing_path     TEXT NOT NULL DEFAULT 'unknown',
    crag_action      TEXT NOT NULL DEFAULT 'UNKNOWN',
    self_rag_retries INTEGER NOT NULL DEFAULT 0,
    confidence       REAL NOT NULL DEFAULT 0.0,
    latency_ms       TEXT NOT NULL DEFAULT '{}',
    sources          TEXT NOT NULL DEFAULT '[]',
    ragas            TEXT NOT NULL DEFAULT '{}',
    created_at       INTEGER NOT NULL
);
"""


@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(settings.QUERY_LOG_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_query_log() -> None:
    os.makedirs(os.path.dirname(os.path.abspath(settings.QUERY_LOG_PATH)), exist_ok=True)
    with _get_conn() as conn:
        conn.execute(_CREATE_TABLE)


def write_query_record(record: dict[str, Any]) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO queries "
            "(query_id,transcript,lang,routing_path,crag_action,"
            "self_rag_retries,confidence,latency_ms,sources,ragas,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                record["query_id"],
                record.get("transcript", ""),
                record.get("lang", "en"),
                record.get("routing_path", "unknown"),
                record.get("crag_action", "UNKNOWN"),
                record.get("self_rag_retries", 0),
                record.get("confidence", 0.0),
                json.dumps(record.get("latency_ms", {})),
                json.dumps(record.get("sources", [])),
                json.dumps(record.get("ragas", {})),
                int(time.time()),
            ),
        )


def get_query_record(query_id: str) -> dict[str, Any] | None:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM queries WHERE query_id = ?", (query_id,)).fetchone()
    if row is None:
        return None
    r = dict(row)
    r["latency_ms"] = json.loads(r["latency_ms"])
    r["sources"] = json.loads(r["sources"])
    r["ragas"] = json.loads(r["ragas"])
    return r


def get_health_metrics() -> dict[str, Any]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM queries ORDER BY created_at DESC LIMIT 100").fetchall()
    if not rows:
        return {
            "total_queries": 0,
            "avg_latency_ms": None,
            "avg_confidence": None,
            "avg_faithfulness": None,
            "routing_distribution": {},
            "node_latency_avg_ms": {},
        }
    records = [dict(r) for r in rows]
    routing_dist: dict[str, int] = {}
    node_totals: dict[str, list[float]] = {}
    faithfulness: list[float] = []
    for r in records:
        routing_dist[r["routing_path"]] = routing_dist.get(r["routing_path"], 0) + 1
        for node, ms in json.loads(r["latency_ms"]).items():
            node_totals.setdefault(node, []).append(ms)
        f = json.loads(r["ragas"]).get("faithfulness")
        if f is not None:
            faithfulness.append(f)
    confs = [r["confidence"] for r in records]
    totals = [json.loads(r["latency_ms"]).get("total", 0) for r in records]
    return {
        "total_queries": len(records),
        "avg_latency_ms": round(sum(totals) / len(totals), 1),
        "avg_confidence": round(sum(confs) / len(confs), 3),
        "avg_faithfulness": round(sum(faithfulness) / len(faithfulness), 3)
        if faithfulness
        else None,
        "routing_distribution": routing_dist,
        "node_latency_avg_ms": {k: round(sum(v) / len(v), 1) for k, v in node_totals.items()},
    }
