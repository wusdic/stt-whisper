import sqlite3
import json
import time
from pathlib import Path
from typing import List, Optional, Dict

DB_PATH = Path("~/.hermes/projects/stt-whisper/projects.db").expanduser()

def get_db() -> sqlite3.Connection:
    """获取数据库连接（线程安全：每次调用新建连接）"""
    conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化数据库和表"""
    conn = get_db()
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            output_dir TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            size INTEGER DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            done_at REAL,
            result_json TEXT,
            error TEXT
        )
    """)
    conn.commit()
    conn.close()

# --- Project CRUD ---

def create_project(name: str, description: str = '', output_dir: Optional[str] = None) -> int:
    """创建项目，返回 project_id"""
    if not output_dir:
        output_dir = str(Path("~/.hermes/projects/stt-whisper/outputs").expanduser() / name)
    now = time.time()
    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO projects (name, description, output_dir, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (name, description, output_dir, now, now)
    )
    project_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return project_id

def list_projects() -> List[Dict]:
    """列出所有项目"""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, description, output_dir, created_at, updated_at FROM projects ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_project(project_id: int) -> Optional[Dict]:
    """获取单个项目"""
    conn = get_db()
    row = conn.execute(
        "SELECT id, name, description, output_dir, created_at, updated_at FROM projects WHERE id = ?",
        (project_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def update_project(project_id: int, name: str = None, description: str = None) -> bool:
    """更新项目"""
    conn = get_db()
    updates = []
    args = []
    if name is not None:
        updates.append("name = ?")
        args.append(name)
    if description is not None:
        updates.append("description = ?")
        args.append(description)
    if updates:
        updates.append("updated_at = ?")
        args.append(time.time())
        args.append(project_id)
        conn.execute(f"UPDATE projects SET {', '.join(updates)} WHERE id = ?", args)
        conn.commit()
    conn.close()
    return True

def delete_project(project_id: int) -> bool:
    """删除项目（级联删除任务）"""
    conn = get_db()
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()
    return True

def get_or_create_default_project() -> int:
    """获取或创建默认项目（id=1，name='默认项目'）"""
    conn = get_db()
    row = conn.execute("SELECT id FROM projects WHERE id = 1").fetchone()
    if row:
        conn.close()
        return row[0]
    # 创建默认项目
    now = time.time()
    default_output = str(Path("~/.hermes/projects/stt-whisper/outputs/default").expanduser())
    cursor = conn.execute(
        "INSERT INTO projects (id, name, description, output_dir, created_at, updated_at) VALUES (1, ?, ?, ?, ?, ?)",
        ("默认项目", "默认转写项目", default_output, now, now)
    )
    conn.commit()
    conn.close()
    return 1

# --- Task CRUD ---

def add_task(project_id: int, name: str, path: str, size: int = 0) -> int:
    """添加任务到项目，返回 task_id"""
    now = time.time()
    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO tasks (project_id, name, path, size, status, created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', ?, ?)",
        (project_id, name, path, size, now, now)
    )
    task_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return task_id

def list_tasks(project_id: int, status: Optional[str] = None) -> List[Dict]:
    """列出项目的任务"""
    conn = get_db()
    if status:
        rows = conn.execute(
            "SELECT id, project_id, name, path, size, status, created_at, updated_at, done_at, result_json, error "
            "FROM tasks WHERE project_id = ? AND status = ? ORDER BY created_at ASC",
            (project_id, status)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, project_id, name, path, size, status, created_at, updated_at, done_at, result_json, error "
            "FROM tasks WHERE project_id = ? ORDER BY created_at ASC",
            (project_id,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_task(task_id: int) -> Optional[Dict]:
    """获取单个任务"""
    conn = get_db()
    row = conn.execute(
        "SELECT id, project_id, name, path, size, status, created_at, updated_at, done_at, result_json, error "
        "FROM tasks WHERE id = ?",
        (task_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def update_task_status(task_id: int, status: str, result_json: str = None, error: str = None) -> bool:
    """更新任务状态"""
    now = time.time()
    done_at = now if status in ('done', 'error') else None
    conn = get_db()
    conn.execute(
        "UPDATE tasks SET status = ?, updated_at = ?, done_at = ?, result_json = ?, error = ? WHERE id = ?",
        (status, now, done_at, result_json, error, task_id)
    )
    conn.commit()
    conn.close()
    return True

def clear_tasks(project_id: int, status: Optional[str] = None) -> int:
    """清空项目的任务，返回删除数量"""
    conn = get_db()
    if status:
        conn.execute("DELETE FROM tasks WHERE project_id = ? AND status = ?", (project_id, status))
    else:
        conn.execute("DELETE FROM tasks WHERE project_id = ?", (project_id,))
    count = conn.total_changes
    conn.commit()
    conn.close()
    return count

def get_project_output_dir(project_id: int) -> str:
    """获取项目的输出目录"""
    project = get_project(project_id)
    if project:
        return project['output_dir']
    # fallback
    return str(Path("~/.hermes/projects/stt-whisper/outputs").expanduser())
