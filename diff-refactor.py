import sys
import difflib
import re
from dataclasses import dataclass
from enum import Enum
import argparse
# --- Configuration ---

# TODO: 1. There are double blocks: sometimes we have some lines that are SPLIT inside of a MOVED block
# TODO: 1. ctd: what should we do with these?
# TODO: 2. Remove the autojunk from the diff. (like )} or newlines).
# TODO: 3. Sometimes recognises a combined added line in one file, but not a combined removed line in another.
# TODO: 4. See: test-swap-complex.diff, it does care about the order of lines in a block!

# Minimal block length to catch combines and splits.
MIN_BLOCK_SIZE = 1
# Minimal block length to trigger block header/footer.
BLOCK_HEADER_THRESHOLD = 4

##################
### Parsing #####
#################


class FileDiffStatus(Enum):
    UNKNOWN = "unknown"
    ADDED = "added"
    DELETED = "deleted"
    MODIFIED = "modified"


class LineDiffStatus(Enum):
    UNKNOWN = "unknown"
    ADDED = "added"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


@dataclass
class ParsedHunkDiffHeader:
    old_start: int  # starting line no. in "a/file.ext"
    old_count: int  # number of lines of the hunk in "a/file.ext"
    new_start: int  # starting line no. of the hunk in "b/file.ext"
    new_count: int  # number of lines of the hunk in "b/file.ext


@dataclass
class ParsedHunkDiffLine:
    line_idx: int
    status: LineDiffStatus
    content: str
    normed_content: str
    absolute_old_line_no: int | None
    absolute_new_line_no: int | None


@dataclass
class ParsedHunkDiff:
    hunk_header: ParsedHunkDiffHeader | None
    lines: list[ParsedHunkDiffLine]


@dataclass
class ParsedFileDiff:
    header: list[str]
    file_path: str
    status: FileDiffStatus
    hunks: list[ParsedHunkDiff]


def parse_hunk_header(
    header: str,
) -> ParsedHunkDiffHeader | None:
    """Extract old/new start and count from a hunk header.
    E.g., for header "@@ -12,5 +12,6 @@" return (12, 5, 12, 6)."""
    m = re.search(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", header)
    if m:
        return ParsedHunkDiffHeader(
            old_start=int(m.group(1)),
            old_count=int(m.group(2)) if m.group(2) != "" else 1,
            new_start=int(m.group(3)),
            new_count=int(m.group(4)) if m.group(4) != "" else 1,
        )
    return None


def norm_line(line):
    return line[1:].lstrip() if line and line[0] in "+- " else line.lstrip()


def parse_diff(diff_text: str) -> list[ParsedFileDiff]:
    files: list[ParsedFileDiff] = []
    current_file: str | None = None
    current_header: list[str] = []
    current_hunks: list[ParsedHunkDiff] = []
    current_hunk: ParsedHunkDiff | None = None
    status: FileDiffStatus = FileDiffStatus.MODIFIED
    cur_old_count = 0
    cur_new_count = 0
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            # diff --git a/src/App.js b/src/App.js
            if current_file is not None:
                if current_hunk is not None:
                    current_hunks.append(current_hunk)
                    current_hunk = None
                files.append(
                    ParsedFileDiff(
                        header=current_header,
                        file_path=current_file,
                        status=status,
                        hunks=current_hunks,
                    )
                )
            current_header = [line]
            parts = line.split()
            current_file = parts[3][2:] if len(parts) >= 4 else "unknown"
            current_hunks = []
            current_hunk = None
            cur_old_count = 0
            cur_new_count = 0
            status = FileDiffStatus.MODIFIED
        elif line.startswith("new file mode"):
            # new file mode 100644
            current_header.append(line)
            status = FileDiffStatus.ADDED
        elif line.startswith("deleted file mode"):
            # deleted file mode 100644
            current_header.append(line)
            status = FileDiffStatus.DELETED
        elif line.startswith("@@"):
            # @@ -1,256 +0,0 @@
            if current_hunk is not None:
                current_hunks.append(current_hunk)
            current_hunk = ParsedHunkDiff(
                hunk_header=parse_hunk_header(line),
                lines=[],
            )
        else:
            if current_hunk is not None:
                idx = len(current_hunk.lines)
                new_line = ParsedHunkDiffLine(
                    line_idx=idx,
                    status=LineDiffStatus.UNKNOWN,
                    content=line,
                    normed_content=norm_line(line),
                    absolute_old_line_no=current_hunk.hunk_header.old_start
                    + cur_old_count,
                    absolute_new_line_no=current_hunk.hunk_header.new_start
                    + cur_new_count,
                )
                if line.startswith("+") and not line.startswith("+++"):
                    new_line.status = LineDiffStatus.ADDED
                    new_line.absolute_old_line_no = None
                    cur_new_count += 1
                elif line.startswith("-") and not line.startswith("---"):
                    new_line.status = LineDiffStatus.DELETED
                    new_line.absolute_new_line_no = None
                    cur_old_count += 1
                else:  # Contect, no added/removed lines
                    new_line.status = LineDiffStatus.UNCHANGED
                    cur_old_count += 1
                    cur_new_count += 1
                current_hunk.lines.append(new_line)
            else:
                current_header.append(line)
    if current_file is not None:
        if current_hunk is not None:
            current_hunks.append(current_hunk)
        files.append(
            ParsedFileDiff(
                header=current_header,
                file_path=current_file,
                status=status,
                hunks=current_hunks,
            )
        )
    return files


####################
### Mapping ########
####################

# TODO: need to add line numbers here


class BlockMarker(Enum):
    MOVED = "moved"
    SPLIT = "split"
    COMBINED = "combined"
    UNKNOWN = "unknown"


class LineMarker(Enum):
    ADDED = "+"
    REMOVED = "-"
    COMBINED_ADDED = "C+"
    COMBINED_REMOVED = "C-"
    MOVED_ADDED = "M+"
    MOVED_REMOVED = "M-"
    SPLIT_ADDED = "S+"
    SPLIT_REMOVED = "S-"
    UNKNOWN_ADDED = "?+"
    UNKNOWN_REMOVED = "?-"
    UNKNOWN = "??"


# type alias
LineLocationKey = tuple[str, int, int]
MappingDict = dict[LineLocationKey, list[LineLocationKey]]


def all_maximal_matches(a: list, b: list, min_size: int = 1) -> list[difflib.Match]:
    # a: a sequence (list of characters | list of lines, etc.)
    # b: a sequence (list of characters | list of lines, etc.)
    # difflib SequenceMatcher.get_matching_blocks() returns a list of Match objects
    # so we mimic that behavior here.
    # The difference here is that we return all matches, so we output more match objects.
    matches = []
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                # Ensure we can't extend backward:
                # either i==0 or j==0, or the previous elements are different
                if i == 0 or j == 0 or a[i - 1] != b[j - 1]:
                    # Now extend forward
                    k = 1
                    # Extend match as long as lines are equal.
                    while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                        k += 1
                    if k >= min_size:
                        # Append a Match tuple (like difflib.Match)
                        matches.append(difflib.Match(i, j, k))
    # Append a final empty match to mimic SequenceMatcher behavior.
    matches.append(difflib.Match(len(a), len(b), 0))
    return matches


def build_match_mappings(
    files: list[ParsedFileDiff],
) -> tuple[
    MappingDict,
    MappingDict,
]:
    """Returns:
    added_mapping: dict mapping (dst_file, hunk_idx, added_idx) -> list of (src_file, hunk_idx, removed_idx)
    removed_mapping: dict mapping (src_file, hunk_idx, removed_idx) -> list of (dst_file, hunk_idx, added_idx)
    """
    added_mapping: MappingDict = {}
    removed_mapping: MappingDict = {}
    n = len(files)
    for src_ii in range(n):
        src = files[src_ii]
        if src.status not in (FileDiffStatus.DELETED, FileDiffStatus.MODIFIED):
            # Ignore files that are added, they cannot provide a source line
            continue
        for dst_jj in range(n):
            dst = files[dst_jj]
            if dst.status not in (FileDiffStatus.ADDED, FileDiffStatus.MODIFIED):
                # Ignore files that are deleted, they cannot provide a destination line
                continue
            for srch_ii, src_hunk in enumerate(src.hunks):
                # Find all the normed lines that are removed in this hunk
                src_lines = [
                    line.normed_content
                    for line in src_hunk.lines
                    if line.status == LineDiffStatus.DELETED
                ]
                if len(src_lines) == 0:
                    continue
                for dsth_jj, dst_hunk in enumerate(dst.hunks):
                    # Find all the normed lines that are added in this hunk
                    dst_lines = [
                        line.normed_content
                        for line in dst_hunk.lines
                        if line.status == LineDiffStatus.ADDED
                    ]
                    if len(dst_lines) == 0:
                        continue
                    # Find all the matching blocks in the sequences of src_lines and dst_lines
                    cross_matching_blocks = all_maximal_matches(src_lines, dst_lines)
                    for block in cross_matching_blocks:
                        if block.size >= MIN_BLOCK_SIZE:
                            for k in range(block.size):
                                # Map each line in the src to each line in the dst
                                # src_idx = the matching line no in the hunk
                                src_idx = block.a + k
                                dst_idx = block.b + k
                                src_key = (src.file_path, srch_ii, src_idx)
                                dst_key = (dst.file_path, dsth_jj, dst_idx)
                                added_mapping.setdefault(dst_key, []).append(src_key)
                                removed_mapping.setdefault(src_key, []).append(dst_key)
    return added_mapping, removed_mapping


# type alias
LineMarkerDict = dict[LineLocationKey, LineMarker]


def compute_markers_individual(
    added_mapping: MappingDict,
    removed_mapping: MappingDict,
) -> tuple[LineMarkerDict, LineMarkerDict]:
    """Assign a marker per mapped line."""
    added_markers = {}
    # for all elements:
    # single source, single destination: moved
    # single source, multiple destinations: split
    # multiple sources, single destination: combined
    # multiple sources, multiple destinations: split
    for dst_key, src_keys in added_mapping.items():
        if len(src_keys) == 1:
            src_key = src_keys[0]
            marker = (
                LineMarker.MOVED_ADDED
                if len(removed_mapping.get(src_key, [])) == 1
                else LineMarker.SPLIT_ADDED
            )
        elif len(src_keys) > 1:
            destination_sibling = set(
                [sib for sk in src_keys for sib in added_mapping.get(sk, [])]
            )
            if len(destination_sibling) < len(src_keys):
                marker = LineMarker.COMBINED_ADDED
            elif len(destination_sibling) < len(src_keys):
                marker = LineMarker.MOVED_ADDED
            else:
                marker = LineMarker.SPLIT_ADDED
        else:
            marker = LineMarker.UNKNOWN_ADDED
        added_markers[dst_key] = marker
    removed_markers = {}
    for src_key, dst_keys in removed_mapping.items():
        if len(dst_keys) == 1:
            dst_key = dst_keys[0]
            marker = (
                LineMarker.MOVED_REMOVED
                if len(added_mapping.get(dst_key, [])) == 1
                else LineMarker.COMBINED_REMOVED
            )
        elif len(dst_keys) > 1:
            source_siblings = set(
                [x for dk in dst_keys for x in added_mapping.get(dk, [])]
            )
            if len(source_siblings) < len(dst_keys):
                marker = LineMarker.SPLIT_REMOVED
            elif len(source_siblings) == len(dst_keys):
                marker = LineMarker.MOVED_REMOVED
            else:  # len(source_siblings) > len(dst_keys):
                marker = LineMarker.COMBINED_REMOVED
        else:
            marker = LineMarker.UNKNOWN
        removed_markers[src_key] = marker
    return added_markers, removed_markers


####################
### Outputting #####
####################
# Colors for markers (dull for moved, bright for unmapped).
MARKER_COLORS = {
    LineMarker.MOVED_ADDED: "\033[2;32m",  # dim green
    LineMarker.MOVED_REMOVED: "\033[2;31m",  # dim red
    LineMarker.COMBINED_ADDED: "\033[94m",  # bright blue
    LineMarker.COMBINED_REMOVED: "\033[93m",  # bright magenta
    LineMarker.SPLIT_ADDED: "\033[95m",  # bright cyan
    LineMarker.SPLIT_REMOVED: "\033[96m",  # bright cyan
}

HEADER_HUNK_COLOR = "\033[1;36m"  # bright cyan for header hunk
DEFAULT_ADDED_COLOR = "\033[1;32m"  # bright green for unmapped +
DEFAULT_REMOVED_COLOR = "\033[1;31m"  # bright red for unmapped -

RESET = "\033[0m"


def block_marker_description(marker: LineMarker) -> BlockMarker:
    if marker in (LineMarker.MOVED_ADDED, LineMarker.MOVED_REMOVED):
        return BlockMarker.MOVED
    elif marker in (LineMarker.SPLIT_ADDED, LineMarker.SPLIT_REMOVED):
        return BlockMarker.SPLIT
    elif marker in (LineMarker.COMBINED_REMOVED, LineMarker.COMBINED_ADDED):
        return BlockMarker.COMBINED
    return BlockMarker.UNKNOWN


def print_hunk_header(hunk: ParsedHunkDiffHeader) -> str:
    return (
        f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@"
    )


def process_hunk_line(
    file,
    hunk,
    hunk_idx,
    hunk_line_idx,
    added_counter,
    removed_counter,
    neutral_counter,
    added_markers,
    removed_markers,
    out_lines,
):
    raise NotImplementedError
    return (
        added_counter,
        removed_counter,
        neutral_counter,
        hunk_line_idx,
        out_lines,
    )


def output_annotated_diff(
    files: list[ParsedFileDiff],
    added_markers: LineMarkerDict,
    removed_markers: LineMarkerDict,
    file_dict: dict[str, ParsedFileDiff],
    added_mapping: MappingDict,
    removed_mapping: MappingDict,
):
    out_lines = []
    for f in files:
        for header_line in f.header:
            out_lines.append(header_line)
        for hi, hunk in enumerate(f.hunks):
            if hunk.hunk_header is not None:
                hunk_header_line = print_hunk_header(hunk.hunk_header)
                out_lines.append(f"{HEADER_HUNK_COLOR}{hunk_header_line}{RESET}")
            # Parse destination hunk header for new start line.
            dst_info = hunk.hunk_header
            dst_new_start = dst_info.new_start if dst_info is not None else 0

            hunk_line_idx = 0
            added_counter = 0
            removed_counter = 0
            neutral_counter = 0
            while hunk_line_idx < len(hunk.lines):
                line = hunk.lines[hunk_line_idx]
                # Process added lines
                if line.status == LineDiffStatus.ADDED:
                    key = (f.file_path, hi, added_counter)
                    if key in added_markers:
                        group_lines: list[str] = []
                        group_keys: list[LineLocationKey] = []
                        # Group contiguous mapped added lines.
                        while (
                            hunk_line_idx < len(hunk.lines)
                            and hunk.lines[hunk_line_idx].status == LineDiffStatus.ADDED
                            and ((f.file_path, hi, added_counter) in added_markers)
                        ):
                            group_lines.append(hunk.lines[hunk_line_idx].content)
                            group_keys.append((f.file_path, hi, added_counter))
                            hunk_line_idx += 1
                            added_counter += 1
                        if len(group_lines) >= BLOCK_HEADER_THRESHOLD:
                            # Output with block header/footer
                            marker = added_markers[group_keys[0]]
                            desc = block_marker_description(marker)
                            # Get source info from the mapping if available:

                            src_keys = added_mapping.get(group_keys[0], [])
                            if src_keys:
                                first_src = src_keys[0]
                                src_file, src_hunk_idx, src_line_idx = first_src
                                src_info = (
                                    file_dict[src_file].hunks[src_hunk_idx].hunk_header
                                )
                                src_line = (
                                    # we actually want old_start + src_line_idx - src_removed_counter)
                                    (src_info.old_start + src_line_idx)
                                    if src_info is not None
                                    else src_line_idx
                                )
                            else:
                                src_file, src_line = "unknown", "?"

                            # dst_line = dst_info.new_start = line number of the hunk in the new file
                            # group_keys[0] is the first line of the grouped added block
                            # group_keys[0][2] is the hunk line number of the first line of the block
                            dst_line = (
                                dst_new_start + group_keys[0][2] + neutral_counter
                            )
                            header_blk = f"----- {desc} block from {src_file}:{src_line} to {f.file_path}:{dst_line} -----"
                            footer_blk = f"----- end {desc} block -----"
                            out_lines.append(header_blk)
                            for k, gline in zip(group_keys, group_lines):
                                cur_added_counter = k[2]
                                cur_line = (
                                    hunk.hunk_header.new_start
                                    + cur_added_counter
                                    + neutral_counter
                                )
                                color = MARKER_COLORS.get(added_markers[k], "")
                                out_lines.append(
                                    f"A{cur_line}:{color}{added_markers[k].value}{gline[1:]}{RESET}"
                                )
                            out_lines.append(footer_blk)
                        else:
                            # Output wihtout block header/footer
                            for k, gline in zip(group_keys, group_lines):
                                cur_line = (
                                    hunk.hunk_header.new_start + k[2] + neutral_counter
                                )

                                color = MARKER_COLORS.get(added_markers[k], "")
                                out_lines.append(
                                    f"B{cur_line}:{color}{added_markers[k].value}{gline[1:]}{RESET}"
                                )
                    else:
                        cur_line = (
                            hunk.hunk_header.new_start + hunk_line_idx - removed_counter
                        )
                        out_lines.append(
                            f"C{cur_line}:{DEFAULT_ADDED_COLOR}+{line.content[1:]}{RESET}"
                        )
                        hunk_line_idx += 1
                        added_counter += 1

                # Process removed lines
                elif line.status == LineDiffStatus.DELETED: 
                    # Output with block header/footer
                    key = (f.file_path, hi, removed_counter)
                    if key in removed_markers:
                        group_lines: list[str] = []
                        group_keys: list[LineLocationKey] = []
                        while (
                            hunk_line_idx < len(hunk.lines)
                            and hunk.lines[hunk_line_idx].status
                            == LineDiffStatus.DELETED
                            and ((f.file_path, hi, removed_counter) in removed_markers)
                        ):
                            group_lines.append(hunk.lines[hunk_line_idx].content)
                            group_keys.append((f.file_path, hi, removed_counter))
                            hunk_line_idx += 1
                            removed_counter += 1
                        if len(group_lines) >= BLOCK_HEADER_THRESHOLD:
                            marker = removed_markers[group_keys[0]]
                            desc = block_marker_description(marker)
                            dst_keys = removed_mapping.get(group_keys[0], [])
                            if dst_keys:
                                first_dst = dst_keys[0]
                                dst_file, dst_hunk_idx, dst_line_idx = first_dst
                                dst_info = (
                                    file_dict[dst_file].hunks[dst_hunk_idx].hunk_header
                                )
                                # we actually want new_start + dst_line_idx - dst_removed_counter)
                                dst_line = (
                                    (dst_info.new_start + dst_line_idx)
                                    if dst_info is not None
                                    else dst_line_idx
                                )
                            else:
                                dst_file, dst_line = "unknown", "?"
                            src_info = hunk.hunk_header
                            src_line = (
                                (
                                    src_info.old_start
                                    + group_keys[0][2]
                                    + neutral_counter
                                )
                                if src_info is not None
                                else group_keys[0][2]
                            )
                            header_blk = f"----- {desc} block from {f.file_path}:{src_line} to {dst_file}:{dst_line} -----"
                            footer_blk = f"----- end {desc} block -----"
                            out_lines.append(header_blk)
                            for k, gline in zip(group_keys, group_lines):
                                cur_removed_counter = k[2]
                                cur_line = (
                                    hunk.hunk_header.old_start
                                    + cur_removed_counter
                                    + neutral_counter
                                )
                                color = MARKER_COLORS.get(removed_markers[k], "")
                                out_lines.append(
                                    f"D{cur_line}:{color}{removed_markers[k].value}{gline[1:]}{RESET}"
                                )
                            out_lines.append(footer_blk)
                        else:
                            # Output wihtout block header/footer
                            for k, gline in zip(group_keys, group_lines):
                                cur_removed_counter = k[2]
                                cur_line = (
                                    hunk.hunk_header.old_start
                                    + cur_removed_counter
                                    + neutral_counter
                                )
                                color = MARKER_COLORS.get(removed_markers[k], "")
                                out_lines.append(
                                    f"E{cur_line}:{color}{removed_markers[k].value}{gline[1:]}{RESET}"
                                )
                    else:
                        cur_line = (
                            hunk.hunk_header.old_start + hunk_line_idx - added_counter
                        )

                        out_lines.append(
                            f"F{cur_line}:{DEFAULT_REMOVED_COLOR}-{line.content[1:]}{RESET}"
                        )
                        hunk_line_idx += 1
                        removed_counter += 1
                else:  # does not start with + or -
                    cur_line_a = (
                        hunk.hunk_header.old_start + hunk_line_idx - added_counter
                    )

                    cur_line_b = (
                        hunk.hunk_header.new_start + hunk_line_idx - removed_counter
                    )
                    out_lines.append(f"G{cur_line_a}/{cur_line_b}:{line.content}")
                    hunk_line_idx += 1
                    neutral_counter += 1
            out_lines.append("")  # empty line after each hunk
    return "\n".join(out_lines)


####################
### Main ###########
####################


def main():
    diff_text = sys.stdin.read()
    files = parse_diff(diff_text)
    file_dict = {f.file_path: f for f in files}
    added_mapping, removed_mapping = build_match_mappings(files)
    print(DEBUG)
    if DEBUG:
        from pprint import pprint

        print(
            "(file path, hunk index, line index) -> list[(file path, hunk index, line index)]"
        )
        print("added mapping")
        pprint(added_mapping)
        print("removed mapping")
        pprint(removed_mapping)

    added_markers, removed_markers = compute_markers_individual(
        added_mapping, removed_mapping
    )
    annotated = output_annotated_diff(
        files, added_markers, removed_markers, file_dict, added_mapping, removed_mapping
    )
    print(annotated)


DEBUG: bool = False
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="diff-refactor",
        description="Provides a diff tailored to diffing refactorings",
        epilog="Example: diff-refactor ",
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    if args.debug:
        DEBUG = True
    main()
