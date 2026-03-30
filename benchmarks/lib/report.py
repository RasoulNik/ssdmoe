"""Unified benchmark report rendering.

Two output targets:
  1. Terminal  — Unicode box-drawing tables, auto-sized columns, consistent style.
  2. JSON      — structured machine-readable output via lib.loader.save_json.

Usage:
    from lib.report import BenchReport, Table, Row

    report = BenchReport(
        title="Window cache hit rate",
        subtitle="fraction of K experts already in RAM",
        tables=[
            Table(
                title="Summary",
                headers=["Window", "Mean%", "Median%", "Min%", "Max%", "RAM cost"],
                rows=[
                    Row(["H=1", "73.2%", "74.5%", "52.0%", "98.0%", "1024 MB"]),
                    Row(["H=2", "81.1%", "82.3%", "60.0%", "99.0%", "2048 MB"], highlight=True),
                ],
            ),
        ],
        notes=["NOTE: Python cache overhead negates these gains."],
        raw={"windows": {...}},
    )
    report.print_terminal()
    report.save_json(Path(args.output))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lib.loader import save_json as _save_json


@dataclass
class Row:
    """One data row in a Table.

    cells:     List of cell strings (will be right-aligned to column width).
    highlight: If True, the row is prefixed with '* ' instead of '  ' in the terminal.
    """
    cells: list[str]
    highlight: bool = False


@dataclass
class Table:
    """A single table inside a BenchReport."""
    headers: list[str]
    rows: list[Row]
    title: str = ""


@dataclass
class BenchReport:
    """A structured benchmark result that can be rendered to terminal or JSON.

    title:    Short title printed at the top of the terminal block.
    tables:   Ordered list of Table objects to render.
    subtitle: Optional subtitle printed below the title.
    notes:    Lines printed below all tables (warnings, caveats, etc.).
    raw:      Machine-readable data dict written by save_json().
    """
    title: str
    tables: list[Table] = field(default_factory=list)
    subtitle: str = ""
    notes: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def print_terminal(self, width: int = 72) -> None:
        """Print a Unicode-box report to stdout."""
        bar = "═" * width
        thin = "─" * width
        print()
        print(bar)
        print(f"  {self.title}")
        if self.subtitle:
            print(f"  {self.subtitle}")
        for table in self.tables:
            print(thin)
            if table.title:
                print(f"  {table.title}")
            _print_table(table)
        if self.notes:
            print(thin)
            for note in self.notes:
                print(f"  {note}")
        print(bar)
        print()

    def save_json(self, path: "Path | str | None") -> None:
        """Write self.raw as indented JSON; no-op if path is None."""
        _save_json(self.raw, path)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _col_widths(table: Table) -> list[int]:
    n = len(table.headers)
    widths = [len(h) for h in table.headers]
    for row in table.rows:
        for i, cell in enumerate(row.cells[:n]):
            widths[i] = max(widths[i], len(cell))
    return widths


def _print_table(table: Table) -> None:
    widths = _col_widths(table)
    header_str = "  ".join(h.rjust(w) for h, w in zip(table.headers, widths))
    print(f"  {header_str}")
    print("  " + "─" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in table.rows:
        cells = [c.rjust(widths[i]) if i < len(widths) else c
                 for i, c in enumerate(row.cells)]
        prefix = "* " if row.highlight else "  "
        print(prefix + "  ".join(cells))
    print()
