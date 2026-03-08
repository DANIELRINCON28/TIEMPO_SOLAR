"""
db_manager.py — Persistencia SQLite para historial de ejecuciones.
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "solarmotion_history.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Crea la tabla de historial si no existe."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            mode        TEXT    NOT NULL,
            model_name  TEXT    NOT NULL,
            hyperparams TEXT,
            equation    TEXT,
            mse         REAL,
            rmse        REAL,
            r2          REAL,
            data_json   TEXT,
            notes       TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_run(mode: str, model_name: str, hyperparams: dict,
             equation: str, metrics: dict, data_x, data_y, notes: str = ""):
    """Inserta una ejecución en el historial."""
    init_db()
    conn = _get_conn()
    data_json = json.dumps({"x": list(map(float, data_x)),
                            "y": list(map(float, data_y))})
    conn.execute("""
        INSERT INTO runs (timestamp, mode, model_name, hyperparams,
                          equation, mse, rmse, r2, data_json, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        mode,
        model_name,
        json.dumps(hyperparams),
        equation,
        metrics.get("MSE"),
        metrics.get("RMSE"),
        metrics.get("R2"),
        data_json,
        notes,
    ))
    conn.commit()
    conn.close()


def get_history(limit=50):
    """Recupera las últimas *limit* ejecuciones."""
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_run(run_id: int):
    """Elimina una ejecución por ID."""
    conn = _get_conn()
    conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()


def clear_history():
    """Elimina todo el historial."""
    conn = _get_conn()
    conn.execute("DELETE FROM runs")
    conn.commit()
    conn.close()
