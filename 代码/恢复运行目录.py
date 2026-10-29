"""Restore the runtime directory read by the analysis code.

Every file listed in 复现文件清单.json is checked against its recorded size and SHA-256 and
written to its runtime path. Gzip files regain their original header and NPZ archives their
original array names; all other bytes are written unchanged.

usage: python 代码/恢复运行目录.py RUNTIME_DIRECTORY
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import re
import struct
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath
MANIFEST_NAME = '复现文件清单.json'


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.as_posix() != value or path.is_absolute() or ('..' in path.parts) or ('\\' in value) or (':' in value):
        raise ValueError('Unsafe path: ' + value)
    return value

def _gzip_transport(payload: bytes, original_header: str | None=None) -> tuple[bytes, dict]:
    """Remove an optional filename without recompressing, or restore its exact header."""
    if len(payload) < 18 or payload[:3] != b'\x1f\x8b\x08' or payload[3] & 224:
        raise ValueError('Unsupported gzip header')
    (flags, cursor, filename) = (payload[3], 10, None)
    if flags & 4:
        if cursor + 2 > len(payload):
            raise ValueError('Truncated gzip extra length')
        cursor += 2 + struct.unpack_from('<H', payload, cursor)[0]
        if cursor > len(payload):
            raise ValueError('Truncated gzip extra data')
    if flags & 8:
        stop = payload.find(b'\x00', cursor)
        if stop < 0:
            raise ValueError('Unterminated gzip filename')
        filename = (cursor, stop + 1)
        cursor = stop + 1
    if flags & 16:
        stop = payload.find(b'\x00', cursor)
        if stop < 0:
            raise ValueError('Unterminated gzip comment')
        cursor = stop + 1
    if flags & 2:
        if cursor + 2 > len(payload) or struct.unpack_from('<H', payload, cursor)[0] != zlib.crc32(payload[:cursor]) & 65535:
            raise ValueError('Invalid gzip header CRC')
        cursor += 2
    if cursor > len(payload) - 8:
        raise ValueError('Truncated gzip body')
    if original_header is not None:
        header = base64.b64decode(original_header, validate=True)
        restored = header + payload[cursor:]
        return (restored, {})
    if filename is None:
        return (payload, {})
    end = cursor - (2 if flags & 2 else 0)
    header = bytearray(payload[:filename[0]] + payload[filename[1]:end])
    header[3] &= ~8
    if flags & 2:
        header.extend(struct.pack('<H', zlib.crc32(header) & 65535))
    return (bytes(header) + payload[cursor:], {'kind': 'gzip_filename_header_removed_without_recompression', 'original_header': base64.b64encode(payload[:cursor]).decode('ascii')})

def _npz_name_transport(payload: bytes, names: dict | None=None, original: dict | None=None) -> tuple[bytes, dict]:
    """Reversibly rename native NPY entries while retaining their compressed streams."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = sorted(archive.infolist(), key=lambda item: item.header_offset)
        if not entries or len({item.filename for item in entries}) != len(entries):
            raise ValueError('NPZ entries are absent or duplicated')
        eocd = payload.rfind(b'PK\x05\x06')
        if eocd < 0 or eocd + 22 + struct.unpack_from('<H', payload, eocd + 20)[0] != len(payload):
            raise ValueError('Unsupported NPZ end record')
        (disk, cd_disk, disk_count, total, size, start) = struct.unpack_from('<HHHHII', payload, eocd + 4)
        if disk or cd_disk or disk_count != total or (total != len(entries)) or (start + size != eocd):
            raise ValueError('Unsupported multipart or ZIP64 NPZ directory')
        blocks = []
        for (index, entry) in enumerate(entries):
            offset = entry.header_offset
            if payload[offset:offset + 4] != b'PK\x03\x04' or entry.flag_bits & 1:
                raise ValueError('Invalid or encrypted NPZ local entry')
            (length, extra) = struct.unpack_from('<HH', payload, offset + 26)
            end = offset + 30 + length + extra
            stop = entries[index + 1].header_offset if index + 1 < len(entries) else start
            if end > stop:
                raise ValueError('Invalid NPZ local span')
            blocks.append((entry, payload[offset:end], payload[end:stop]))
        if original is not None:
            declared = original['entries']
            if len(declared) != len(blocks) or [item[0].filename for item in blocks] != [item['archive_name'] for item in declared]:
                raise ValueError('NPZ inverse names differ')
            restored = base64.b64decode(original['prefix'], validate=True)
            for ((_, _, body), item) in zip(blocks, declared, strict=True):
                restored += base64.b64decode(item['header'], validate=True) + body
            restored += base64.b64decode(original['tail'], validate=True)
            return (restored, {})
        if names is None or set(names) != {entry.filename for entry in entries}:
            raise ValueError('An exact native-array name mapping is required')
        if any(('/' in value or '\\' in value or (not value.endswith('.npy')) or (not re.search('[\\u4e00-\\u9fff]', value)) for value in names.values())) or len(set(names.values())) != len(names):
            raise ValueError('Native-array transport names must be unique flat Chinese NPY names')
        prefix = payload[:entries[0].header_offset]
        (local, offsets, records) = (bytearray(prefix), {}, [])
        for (entry, header, body) in blocks:
            new_name = names[entry.filename].encode('utf-8')
            offsets[entry.header_offset] = len(local)
            fixed = bytearray(header[:30])
            struct.pack_into('<H', fixed, 6, entry.flag_bits | 2048)
            struct.pack_into('<H', fixed, 26, len(new_name))
            old_name_length = struct.unpack_from('<H', header, 26)[0]
            extra = header[30 + old_name_length:]
            local.extend(fixed + new_name + extra + body)
            records.append(dict(archive_name=names[entry.filename], header=base64.b64encode(header).decode('ascii')))
        (directory, position) = (bytearray(), start)
        for _ in range(total):
            if payload[position:position + 4] != b'PK\x01\x02':
                raise ValueError('Invalid NPZ directory entry')
            (n, x, comment) = struct.unpack_from('<HHH', payload, position + 28)
            fixed = bytearray(payload[position:position + 46])
            flags = struct.unpack_from('<H', fixed, 8)[0]
            old_name = payload[position + 46:position + 46 + n].decode('utf-8' if flags & 2048 else 'cp437')
            offset = struct.unpack_from('<I', fixed, 42)[0]
            if offset not in offsets:
                raise ValueError('Unsupported NPZ directory offset')
            new_name = names[old_name].encode('utf-8')
            struct.pack_into('<H', fixed, 8, flags | 2048)
            struct.pack_into('<H', fixed, 28, len(new_name))
            struct.pack_into('<I', fixed, 42, offsets[offset])
            directory.extend(fixed + new_name + payload[position + 46 + n:position + 46 + n + x + comment])
            position += 46 + n + x + comment
        if position != eocd:
            raise ValueError('Unexpected NPZ directory trailer')
        ending = bytearray(payload[eocd:])
        struct.pack_into('<II', ending, 12, len(directory), len(local))
        transported = bytes(local + directory + ending)
        with zipfile.ZipFile(io.BytesIO(transported)) as other:
            for entry in entries:
                if archive.read(entry.filename) != other.read(names[entry.filename]):
                    raise ValueError('Native-array bytes changed during name transport')
        return (transported, dict(kind='native_npz_names_only', prefix=base64.b64encode(prefix).decode('ascii'), entries=records, tail=base64.b64encode(payload[start:]).decode('ascii')))

def restore(repository: Path, output: Path) -> dict:
    """Write every listed file to a new runtime directory after checking its identity."""
    repository = repository.resolve()
    if output.is_symlink():
        raise FileExistsError('The runtime directory must not be a symlink')
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Use a new runtime directory; existing files are left unchanged')
    manifest = json.loads((repository / MANIFEST_NAME).read_text(encoding='utf-8'))
    (staged, seen) = ([], set())
    for row in manifest['files']:
        runtime = _relative(row['runtime_path'])
        if runtime in seen:
            raise ValueError('Duplicate runtime path: ' + runtime)
        seen.add(runtime)
        payload = (repository / _relative(row['repository_path'])).read_bytes()
        if len(payload) != row['bytes'] or hashlib.sha256(payload).hexdigest() != row['sha256']:
            raise ValueError('File differs from its recorded identity: ' + row['repository_path'])
        transform = row.get('transform') or {}
        if transform.get('kind') == 'gzip_filename_header_removed_without_recompression':
            (payload, _) = _gzip_transport(payload, transform['original_header'])
        elif transform.get('kind') == 'native_npz_names_only':
            (payload, _) = _npz_name_transport(payload, original=transform)
        elif transform:
            raise ValueError('Unknown transform: ' + str(transform.get('kind')))
        if hashlib.sha256(payload).hexdigest() != row['runtime_sha256']:
            raise ValueError('Restored file differs from its recorded identity: ' + runtime)
        staged.append((runtime, payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='runtime-', dir=output.parent) as temporary:
        work = Path(temporary) / 'runtime'
        work.mkdir()
        for (runtime, payload) in staged:
            target = work / runtime
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        work.replace(output)
    return dict(status='restored', files=len(staged), runtime_directory=str(output))


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='new runtime directory')
    args = parser.parse_args()
    print(json.dumps(restore(Path(__file__).resolve().parents[1], args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
