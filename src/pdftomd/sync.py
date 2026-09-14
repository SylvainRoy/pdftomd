"""One-way, incremental synchronisation of a source tree into Markdown files.

The source directory is only ever read. State lives in a manifest stored in
the destination directory, keyed by the POSIX relative path of each source
file. A source file is (re)generated when any of the following holds:

* it has no manifest entry,
* its content hash differs from the recorded one,
* the recorded engine differs from the engine in use,
* the destination Markdown file is missing,
* it was explicitly selected / ``force`` was requested.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

from .converters.base import ConversionError, Converter

MANIFEST_NAME = ".pdftomd-manifest.json"
MANIFEST_VERSION = 1


class Reason(str, Enum):
    NEW = "new"
    CHANGED = "changed"
    ENGINE_CHANGED = "engine-changed"
    OUTPUT_MISSING = "output-missing"
    FORCED = "forced"


@dataclass(frozen=True)
class PlannedItem:
    rel_path: str
    source: Path
    destination: Path
    reason: Reason


@dataclass
class SyncPlan:
    to_generate: list[PlannedItem] = field(default_factory=list)
    up_to_date: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    orphans: list[Path] = field(default_factory=list)  # destination files without a source


@dataclass
class SyncResult:
    generated: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    pruned: list[str] = field(default_factory=list)


@dataclass
class ManifestEntry:
    sha256: str
    size: int
    mtime: float
    engine: str
    output: str
    generated_at: float


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def markdown_path_for(rel_path: str) -> str:
    """``sub/dir/report.pdf`` -> ``sub/dir/report.md``."""
    return str(PurePosixPath(rel_path).with_suffix(".md"))


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, ManifestEntry] = {}
        if path.exists():
            data = json.loads(path.read_text("utf-8"))
            self.entries = {k: ManifestEntry(**v) for k, v in data.get("files", {}).items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": MANIFEST_VERSION, "files": {k: asdict(v) for k, v in sorted(self.entries.items())}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), "utf-8")
        os.replace(tmp, self.path)


class Syncer:
    def __init__(
        self,
        source_dir: str | Path,
        dest_dir: str | Path,
        converter: Converter,
        *,
        extensions: Iterable[str] | None = None,
        follow_symlinks: bool = False,
    ) -> None:
        self.source_dir = Path(source_dir).resolve()
        self.dest_dir = Path(dest_dir).resolve()
        if not self.source_dir.is_dir():
            raise NotADirectoryError(self.source_dir)
        if self.dest_dir == self.source_dir or self.source_dir in self.dest_dir.parents:
            raise ValueError("Destination must not be the source directory or inside it.")
        self.converter = converter
        self.extensions = frozenset(e.lower() for e in extensions) if extensions else converter.extensions
        self.follow_symlinks = follow_symlinks
        self.manifest = Manifest(self.dest_dir / MANIFEST_NAME)

    # -- discovery -----------------------------------------------------------

    def iter_source_files(self) -> Iterable[tuple[str, Path]]:
        for root, dirs, files in os.walk(self.source_dir, followlinks=self.follow_symlinks):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(files):
                if name.startswith("."):
                    continue
                path = Path(root) / name
                yield path.relative_to(self.source_dir).as_posix(), path

    def _rel(self, selection: str | Path) -> str:
        p = Path(selection)
        if p.is_absolute():
            p = p.resolve().relative_to(self.source_dir)
        return PurePosixPath(p.as_posix()).as_posix()

    # -- planning ------------------------------------------------------------

    def plan(self, *, force: bool = False, select: Iterable[str | Path] | None = None) -> SyncPlan:
        """Compute what would be (re)generated. Never touches the filesystem.

        ``select`` restricts the plan to the given source paths (relative to
        the source directory, or absolute) and forces their regeneration.
        """
        selected = {self._rel(s) for s in select} if select else None
        plan = SyncPlan()
        seen_outputs: set[str] = set()

        for rel, src in self.iter_source_files():
            if src.suffix.lower() not in self.extensions:
                if selected is None or rel in selected:
                    plan.unsupported.append(rel)
                continue
            out_rel = markdown_path_for(rel)
            seen_outputs.add(out_rel)
            dest = self.dest_dir / out_rel
            if selected is not None:
                if rel in selected:
                    plan.to_generate.append(PlannedItem(rel, src, dest, Reason.FORCED))
                continue
            reason = Reason.FORCED if force else self._staleness(rel, src, dest)
            if reason is None:
                plan.up_to_date.append(rel)
            else:
                plan.to_generate.append(PlannedItem(rel, src, dest, reason))

        if selected is not None:
            missing = selected - {i.rel_path for i in plan.to_generate} - set(plan.unsupported)
            if missing:
                raise FileNotFoundError(f"Selected files not found in source: {sorted(missing)}")

        if self.dest_dir.is_dir():
            for root, dirs, files in os.walk(self.dest_dir):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in files:
                    p = Path(root) / name
                    rel_out = p.relative_to(self.dest_dir).as_posix()
                    if p.suffix.lower() == ".md" and rel_out not in seen_outputs:
                        plan.orphans.append(p)
        return plan

    def _staleness(self, rel: str, src: Path, dest: Path) -> Reason | None:
        entry = self.manifest.entries.get(rel)
        if entry is None:
            return Reason.NEW
        if not dest.exists():
            return Reason.OUTPUT_MISSING
        if entry.engine != self.converter.name:
            return Reason.ENGINE_CHANGED
        st = src.stat()
        # Cheap check first; only hash when size/mtime moved.
        if st.st_size == entry.size and st.st_mtime == entry.mtime:
            return None
        if file_sha256(src) != entry.sha256:
            return Reason.CHANGED
        # Same content, only metadata changed: refresh the recorded metadata.
        entry.size, entry.mtime = st.st_size, st.st_mtime
        return None

    # -- execution -----------------------------------------------------------

    def execute(
        self,
        plan: SyncPlan,
        *,
        prune: bool = False,
        on_progress: Callable[[PlannedItem, int, int], None] | None = None,
        on_error: Callable[[PlannedItem, Exception], None] | None = None,
    ) -> SyncResult:
        result = SyncResult()
        total = len(plan.to_generate)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(plan.to_generate, 1):
            if on_progress:
                on_progress(item, index, total)
            try:
                st = item.source.stat()
                sha = file_sha256(item.source)
                markdown = self.converter.convert_file(item.source)
            except (ConversionError, OSError) as exc:
                result.failed[item.rel_path] = str(exc)
                if on_error:
                    on_error(item, exc)
                continue
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            tmp = item.destination.with_name(item.destination.name + ".tmp")
            tmp.write_text(markdown, "utf-8")
            os.replace(tmp, item.destination)
            self.manifest.entries[item.rel_path] = ManifestEntry(
                sha256=sha,
                size=st.st_size,
                mtime=st.st_mtime,
                engine=self.converter.name,
                output=item.destination.relative_to(self.dest_dir).as_posix(),
                generated_at=time.time(),
            )
            result.generated.append(item.rel_path)
            self.manifest.save()  # persist after each file so a crash loses nothing

        if prune:
            for orphan in plan.orphans:
                orphan.unlink(missing_ok=True)
                result.pruned.append(orphan.relative_to(self.dest_dir).as_posix())
            live = {rel for rel, _ in self.iter_source_files()}
            for rel in list(self.manifest.entries):
                if rel not in live:
                    del self.manifest.entries[rel]
        self.manifest.save()
        return result

    def sync(self, *, force: bool = False, select: Iterable[str | Path] | None = None, prune: bool = False, **callbacks) -> SyncResult:
        return self.execute(self.plan(force=force, select=select), prune=prune, **callbacks)
