#!/usr/bin/env python3
"""Fetch only sequences 09/10 from official KITTI ZIPs using HTTP ranges.

Run from any directory; completed files are CRC-checked and reused on restart.
SemanticKITTI labels and SuMa poses must already be in data/semantickitti.
"""
import concurrent.futures
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'semantickitti'
BASE = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/'
SEQUENCES = ('09', '10')


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
            self.size = int(r.headers['Content-Length'])
            self.etag = r.headers['ETag']
        self.pos = 0
        self.cache = {}
        self.block_size = 4 * 1024 * 1024

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        if self.pos < 0:
            raise ValueError('Negative seek')
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size)
        result = bytearray()
        while n > 0:
            start = self.pos // self.block_size * self.block_size
            if start not in self.cache:
                end = min(start + self.block_size, self.size) - 1
                for attempt in range(5):
                    try:
                        req = urllib.request.Request(self.url, headers={
                            'Range': f'bytes={start}-{end}', 'If-Match': self.etag})
                        with urllib.request.urlopen(req, timeout=20) as r:
                            if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                                raise ValueError('Server did not honor the requested range')
                            data = r.read(end - start + 2)
                        if len(data) != end - start + 1:
                            raise ValueError('Truncated HTTP range')
                        break
                    except Exception:
                        if attempt == 4:
                            raise
                        print(f'Retrying range {start}-{end} (attempt {attempt + 2}/5)', flush=True)
                        time.sleep(2 ** attempt)
                if len(self.cache) >= 2:
                    del self.cache[next(iter(self.cache))]
                self.cache[start] = data
            data = self.cache[start]
            offset = self.pos - start
            take = min(n, len(data) - offset)
            result.extend(data[offset:offset + take])
            self.pos += take
            n -= take
        return bytes(result)


def crc(path):
    value = 0
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            value = zlib.crc32(block, value)
    return value & 0xffffffff


def selected(z, seq):
    prefix = f'dataset/sequences/{seq}/'
    return sorted((i for i in z.infolist() if i.filename.startswith(prefix) and not i.is_dir()),
                  key=lambda i: i.header_offset)


def fetch(seq, archive, shard=0, shards=1):
    remote = RemoteZip(BASE + archive)
    with zipfile.ZipFile(remote) as z:
        entries = selected(z, seq)
        entries = entries[len(entries) * shard // shards:len(entries) * (shard + 1) // shards]
        label = f'{seq} part {shard + 1}/{shards}'
        total = sum(i.file_size for i in entries)
        print(f'{label} {archive}: {len(entries)} files, {total / 1e9:.2f} GB extracted', flush=True)
        done = 0
        last = time.monotonic()
        for i, entry in enumerate(entries, 1):
            parts = PurePosixPath(entry.filename).parts[1:]
            if '..' in parts:
                raise ValueError('Unsafe ZIP path')
            dest = ROOT.joinpath(*parts)
            if dest.exists():
                if dest.stat().st_size != entry.file_size or crc(dest) != entry.CRC:
                    raise ValueError(f'Existing file differs from official archive: {dest}')
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + '.part')
                with z.open(entry) as src, tmp.open('wb') as out:
                    shutil.copyfileobj(src, out)
                if tmp.stat().st_size != entry.file_size or crc(tmp) != entry.CRC:
                    raise ValueError(f'CRC mismatch: {tmp}')
                tmp.replace(dest)
            done += entry.file_size
            if time.monotonic() - last >= 20 or i == len(entries):
                print(f'{label}: {i}/{len(entries)} files ({done / 1e9:.2f}/{total / 1e9:.2f} GB)', flush=True)
                last = time.monotonic()
        return {'sequence': seq, 'part': shard + 1, 'url': BASE + archive, 'etag': remote.etag,
                'files': len(entries), 'bytes': total, 'verification': 'ZIP CRC32 for every extracted file'}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RemoteZip(BASE + 'data_odometry_velodyne.zip')) as z:
        entries = [e for seq in SEQUENCES for e in selected(z, seq)]
        size = sum(e.file_size for e in entries)
        packed = sum(e.compress_size for e in entries)
        print(f'Selected scans: {len(entries)} files, {packed / 1e9:.2f} GB compressed, {size / 1e9:.2f} GB extracted', flush=True)
        if shutil.disk_usage(ROOT).free < size + 2 * 1024 ** 3:
            raise RuntimeError('Insufficient disk space including 2 GiB reserve')
    results = []
    for archive in ('data_odometry_calib.zip', 'data_odometry_velodyne.zip'):
        shards = 4 if archive == 'data_odometry_velodyne.zip' else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            jobs = [pool.submit(fetch, seq, archive, shard, shards)
                    for seq in SEQUENCES for shard in range(shards)]
            results.extend(job.result() for job in jobs)
    manifest_path = ROOT / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for seq in SEQUENCES:
        d = ROOT / 'sequences' / seq
        scans = sorted((d / 'velodyne').glob('*.bin'))
        labels = sorted((d / 'labels').glob('*.label'))
        poses = (d / 'poses.txt').read_text().splitlines()
        times = (d / 'times.txt').read_text().splitlines()
        assert len(scans) == len(labels) == len(poses) == len(times) > 0
        assert [p.stem for p in scans] == [p.stem for p in labels]
        for scan, label in zip(scans, labels):
            assert scan.stat().st_size % 16 == 0
            assert scan.stat().st_size // 16 == label.stat().st_size // 4, scan.name
        assert 'Tr:' in (d / 'calib.txt').read_text()
        manifest['sequences'][seq].update(scans=len(scans), timestamps=len(times), missing=[])
        print(f'VERIFIED {seq}: {len(scans)} matching scans, labels, poses and timestamps; per-frame point counts agree.', flush=True)
    manifest['kitti_archives'] = results
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    readme = ROOT / 'README.md'
    text = readme.read_text()
    text = text[:text.index('Status:')] + 'Status: COMPLETE. Sequences 09 and 10 contain verified scans, labels, calibration, timestamps, and SemanticKITTI SuMa poses. See download_manifest.json for provenance and checks.\n'
    readme.write_text(text)


if __name__ == '__main__':
    main()
