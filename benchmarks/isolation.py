"""Benchmark isolation: cache flushing and server restart between config runs.

Eliminates the cache confound where residual shared_buffers/OS page cache
state from one configuration's benchmark leaks into the next.

Usage:
    from benchmarks.isolation import IsolatedBenchmarkRunner, flush_all_caches

    runner = IsolatedBenchmarkRunner()
    runner.add("Mono", lambda: run_q1(conn, TABLE_MONO, ...))
    runner.add("SPDB", lambda: run_q1(conn, TABLE_SPDB, ...))
    results = runner.run_isolated()
"""

import os
import sys
import time
import platform
import subprocess
import logging

import psycopg2

from spdb import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def _detect_platform():
    """Return 'linux', 'darwin', or 'unknown'."""
    return platform.system().lower()


# ---------------------------------------------------------------------------
# PostgreSQL restart
# ---------------------------------------------------------------------------

def _find_pg_ctl():
    """Locate pg_ctl binary."""
    candidates = [
        "/usr/lib/postgresql/17/bin/pg_ctl",
        "/usr/lib/postgresql/16/bin/pg_ctl",
        "/usr/pgsql-17/bin/pg_ctl",
        "/usr/pgsql-16/bin/pg_ctl",
        "/opt/homebrew/opt/postgresql@17/bin/pg_ctl",
        "/opt/homebrew/bin/pg_ctl",
        "/usr/local/bin/pg_ctl",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Try PATH
    try:
        result = subprocess.run(["which", "pg_ctl"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _find_data_dir():
    """Locate PostgreSQL data directory."""
    candidates = [
        "/var/lib/postgresql/17/main",
        "/var/lib/postgresql/16/main",
        "/var/lib/pgsql/17/data",
        "/opt/homebrew/var/postgresql@17",
        "/opt/homebrew/var/postgres",
        "/usr/local/var/postgres",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def wait_for_pg_ready(timeout=30, dsn=None):
    """Poll until PostgreSQL accepts connections. Returns True on success."""
    if dsn is None:
        dsn = config.dsn()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = psycopg2.connect(dsn)
            conn.close()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def restart_postgresql(pg_ctl=None, data_dir=None):
    """Restart PostgreSQL cleanly. Tries multiple methods.

    Returns dict with method used and timing.
    """
    t0 = time.time()
    plat = _detect_platform()
    method = None
    success = False

    # Method 1: systemctl (Linux)
    if plat == "linux":
        for svc in ["postgresql", "postgresql-17", "postgresql-16"]:
            try:
                r = subprocess.run(
                    ["sudo", "-n", "systemctl", "restart", svc],
                    capture_output=True, text=True, timeout=30,
                )
                if r.returncode == 0:
                    method = f"systemctl restart {svc}"
                    success = True
                    break
            except Exception:
                continue

    # Method 2: pg_ctl
    if not success:
        ctl = pg_ctl or _find_pg_ctl()
        ddir = data_dir or _find_data_dir()
        if ctl and ddir:
            try:
                pg_user = "postgres"
                subprocess.run(
                    ["sudo", "-n", "-u", pg_user, ctl, "restart",
                     "-D", ddir, "-m", "fast", "-w"],
                    capture_output=True, text=True, timeout=30,
                )
                method = f"pg_ctl restart -D {ddir}"
                success = True
            except Exception:
                pass

    # Method 3: brew services (macOS)
    if not success and plat == "darwin":
        for svc in ["postgresql@17", "postgresql@16", "postgresql"]:
            try:
                subprocess.run(
                    ["brew", "services", "restart", svc],
                    capture_output=True, text=True, timeout=30,
                )
                method = f"brew services restart {svc}"
                success = True
                break
            except Exception:
                continue

    elapsed = time.time() - t0

    if success:
        pg_ready = wait_for_pg_ready(timeout=30)
        if not pg_ready:
            log.warning("PostgreSQL restart succeeded but server not ready after 30s")
            success = False
    else:
        log.warning("Could not restart PostgreSQL via any method")

    return {
        "success": success,
        "method": method,
        "elapsed_sec": round(elapsed, 2),
        "pg_ready": success and wait_for_pg_ready(timeout=5),
    }


# ---------------------------------------------------------------------------
# OS page cache flush
# ---------------------------------------------------------------------------

def drop_os_caches(plat=None):
    """Drop OS filesystem caches. Requires sudo.

    Linux: sync + echo 3 > /proc/sys/vm/drop_caches
    macOS: sudo purge
    """
    if plat is None:
        plat = _detect_platform()

    t0 = time.time()
    success = False
    method = None

    if plat == "linux":
        try:
            subprocess.run(["sync"], timeout=10)
            r = subprocess.run(
                ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                capture_output=True, text=True, timeout=10,
            )
            success = r.returncode == 0
            method = "echo 3 > /proc/sys/vm/drop_caches"
        except Exception as e:
            log.warning(f"Failed to drop Linux caches: {e}")

    elif plat == "darwin":
        try:
            r = subprocess.run(
                ["sudo", "-n", "purge"],
                capture_output=True, text=True, timeout=30,
            )
            success = r.returncode == 0
            method = "sudo purge"
        except Exception as e:
            log.warning(f"Failed to purge macOS caches: {e}")

    return {
        "success": success,
        "method": method,
        "elapsed_sec": round(time.time() - t0, 2),
        "platform": plat,
    }


# ---------------------------------------------------------------------------
# PostgreSQL buffer flush via SQL (soft alternative)
# ---------------------------------------------------------------------------

def flush_pg_buffers_sql(conn):
    """Flush PostgreSQL shared buffers without restart.

    Creates and drops a large temp table to push cached pages out of
    shared_buffers. This is a best-effort soft flush.
    """
    t0 = time.time()
    try:
        with conn.cursor() as cur:
            # Discard all cached plans
            cur.execute("DISCARD ALL")

        # Reconnect since DISCARD ALL invalidates the session
        conn.close()
        conn = psycopg2.connect(config.dsn())

        with conn.cursor() as cur:
            # Generate large sequential scan to push out cached pages
            cur.execute("""
                CREATE TEMP TABLE _cache_flush AS
                SELECT generate_series(1, 5000000) AS n,
                       repeat('x', 200) AS padding
            """)
            cur.execute("DROP TABLE IF EXISTS _cache_flush")
        conn.commit()

        return {
            "success": True,
            "method": "temp_table_flush",
            "elapsed_sec": round(time.time() - t0, 2),
            "connection": conn,
        }
    except Exception as e:
        log.warning(f"SQL buffer flush failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return {
            "success": False,
            "method": "temp_table_flush",
            "elapsed_sec": round(time.time() - t0, 2),
            "error": str(e),
            "connection": conn,
        }


# ---------------------------------------------------------------------------
# Combined flush
# ---------------------------------------------------------------------------

def flush_all_caches(plat=None, restart_pg=True, drop_os=True):
    """Flush both PostgreSQL shared buffers and OS page cache.

    This is the primary function to call between benchmark config runs.

    Parameters
    ----------
    plat : str, optional
        'linux' or 'darwin'. Auto-detected if None.
    restart_pg : bool
        Whether to restart PostgreSQL (full buffer flush).
    drop_os : bool
        Whether to drop OS page cache.

    Returns
    -------
    dict with details of what was flushed and timing.
    """
    if plat is None:
        plat = _detect_platform()

    t0 = time.time()
    result = {"platform": plat, "steps": []}

    if restart_pg:
        pg_result = restart_postgresql()
        result["steps"].append({"action": "restart_postgresql", **pg_result})
        if not pg_result["success"]:
            # Fall back to SQL flush
            try:
                conn = psycopg2.connect(config.dsn())
                sql_result = flush_pg_buffers_sql(conn)
                result["steps"].append({"action": "sql_buffer_flush", **{
                    k: v for k, v in sql_result.items() if k != "connection"
                }})
            except Exception as e:
                result["steps"].append({
                    "action": "sql_buffer_flush",
                    "success": False,
                    "error": str(e),
                })

    if drop_os:
        os_result = drop_os_caches(plat)
        result["steps"].append({"action": "drop_os_caches", **os_result})

    # Brief sleep to let things settle
    time.sleep(1.0)

    result["total_elapsed_sec"] = round(time.time() - t0, 2)
    result["all_success"] = all(s.get("success", False) for s in result["steps"])

    return result


# ---------------------------------------------------------------------------
# Isolated Benchmark Runner
# ---------------------------------------------------------------------------

class IsolatedBenchmarkRunner:
    """Runs benchmarks with full cache isolation between configurations.

    Usage:
        runner = IsolatedBenchmarkRunner()
        runner.add("Mono", lambda: run_q1(...))
        runner.add("SPDB", lambda: run_q1(...))
        results = runner.run_isolated()
    """

    def __init__(self, restart_pg=True, drop_os=True, verbose=True):
        self.configs = []
        self.restart_pg = restart_pg
        self.drop_os = drop_os
        self.verbose = verbose
        self.isolation_log = []

    def add(self, name, benchmark_fn):
        """Add a benchmark configuration to run.

        Parameters
        ----------
        name : str
            Configuration name (e.g., "Mono", "SPDB").
        benchmark_fn : callable
            Function that runs the benchmark and returns results.
            Must handle its own DB connection (reconnect after restart).
        """
        self.configs.append((name, benchmark_fn))

    def run_isolated(self):
        """Run all configs with full cache isolation between each.

        Returns dict mapping config name to benchmark results.
        """
        results = {}
        total_t0 = time.time()

        for i, (name, fn) in enumerate(self.configs):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"  Config {i+1}/{len(self.configs)}: {name}")
                print(f"{'='*60}")

            # Flush caches before each config (including the first)
            if self.verbose:
                print(f"  Flushing caches...")
            flush_result = flush_all_caches(
                restart_pg=self.restart_pg,
                drop_os=self.drop_os,
            )
            self.isolation_log.append({
                "config": name,
                "flush": flush_result,
            })

            if self.verbose:
                status = "OK" if flush_result["all_success"] else "PARTIAL"
                print(f"  Cache flush: {status} "
                      f"({flush_result['total_elapsed_sec']:.1f}s)")

            # Run the benchmark
            if self.verbose:
                print(f"  Running benchmark...")

            bench_t0 = time.time()
            try:
                result = fn()
                results[name] = result
            except Exception as e:
                log.error(f"Benchmark {name} failed: {e}")
                results[name] = {"error": str(e)}

            bench_elapsed = time.time() - bench_t0
            if self.verbose:
                print(f"  Benchmark complete ({bench_elapsed:.1f}s)")

        total_elapsed = time.time() - total_t0
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  All configs complete in {total_elapsed:.0f}s")
            print(f"{'='*60}")

        return results

    def get_isolation_log(self):
        """Return the log of all isolation steps performed."""
        return self.isolation_log
