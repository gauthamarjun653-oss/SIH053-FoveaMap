#!/usr/bin/env python3
"""Download one labelled KITTI sequence, including scans, labels, poses and calibration.

Official HTTP range requests avoid downloading the entire 85 GB scan archive.
Existing files are checked against ZIP CRC32 before reuse. Run again to resume.
"""
import argparse
import concurrent.futures
import json
from pathlib import PurePosixPath
import shutil
import zipfile
from download_kitti_09_10 import BASE, ROOT, RemoteZip, crc, fetch, selected

LABEL_URL = 'https://semantic-kitti.org/assets/data_odometry_labels.zip'


def fetch_labels(seq):
    remote = RemoteZip(LABEL_URL)
    with zipfile.ZipFile(remote) as archive:
        entries = selected(archive, seq)
        if not entries:
            raise RuntimeError(f'No sequence {seq} labels in official archive')
        for entry in entries:
            parts = PurePosixPath(entry.filename).parts[1:]
            if '..' in parts or PurePosixPath(entry.filename).is_absolute():
                raise ValueError('Unsafe ZIP path')
            dest = ROOT.joinpath(*parts)
            if dest.exists():
                if dest.stat().st_size != entry.file_size or crc(dest) != entry.CRC:
                    raise RuntimeError(f'Existing file differs from archive: {dest}')
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                temporary = dest.with_suffix(dest.suffix + '.part')
                with archive.open(entry) as source, temporary.open('wb') as out:
                    shutil.copyfileobj(source, out)
                if temporary.stat().st_size != entry.file_size or crc(temporary) != entry.CRC:
                    raise RuntimeError(f'CRC mismatch: {temporary}')
                temporary.replace(dest)
        print(f'{seq}: labels/poses verified, {len(entries)} files', flush=True)
        return dict(sequence=seq, url=LABEL_URL, etag=remote.etag, files=len(entries),
                    bytes=sum(e.file_size for e in entries), verification='ZIP CRC32 for every extracted file')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', choices=[f'{i:02}' for i in range(11)])
    args = parser.parse_args()
    seq = args.sequence
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RemoteZip(BASE + 'data_odometry_velodyne.zip')) as archive:
        entries = selected(archive, seq)
        if not entries:
            raise RuntimeError('Requested sequence absent from official archive')
        total = sum(e.file_size for e in entries)
        expected = len(entries)
        # Labels are four bytes per 16-byte scan point; reserve extra working space.
        if shutil.disk_usage(ROOT).free < total * 1.25 + 2 * 1024**3:
            raise RuntimeError('Insufficient free disk space including 2 GiB reserve')
        print(f'Sequence {seq}: {expected} scans, {total / 1e9:.2f} GB scans + labels/metadata', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(fetch, seq, 'data_odometry_velodyne.zip', shard, 4) for shard in range(4)]
        jobs += [pool.submit(fetch, seq, 'data_odometry_calib.zip'), pool.submit(fetch_labels, seq)]
        results = [job.result() for job in jobs]
    directory = ROOT / 'sequences' / seq
    scans = sorted((directory/'velodyne').glob('*.bin'))
    labels = sorted((directory/'labels').glob('*.label'))
    poses = (directory/'poses.txt').read_text().splitlines()
    times = (directory/'times.txt').read_text().splitlines()
    if not (len(scans) == len(labels) == len(poses) == len(times) == expected):
        raise RuntimeError('Scan/label/pose/timestamp count mismatch')
    if [p.stem for p in scans] != [p.stem for p in labels]:
        raise RuntimeError('Scan/label filename mismatch')
    for scan, label in zip(scans, labels):
        if scan.stat().st_size % 16 or label.stat().st_size % 4 or scan.stat().st_size // 16 != label.stat().st_size // 4:
            raise RuntimeError(f'Point/label count mismatch: {scan}')
    if 'Tr:' not in (directory/'calib.txt').read_text():
        raise RuntimeError('Missing LiDAR calibration transform')
    manifest_path = ROOT / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault('sequences', {})[seq] = dict(scans=len(scans), labels=len(labels),
        poses=len(poses), timestamps=len(times), label_bytes=sum(p.stat().st_size for p in labels),
        missing=[], downloads=results)
    temporary = manifest_path.with_suffix('.json.part')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(manifest_path)
    readme = ROOT/'README.md'
    text = readme.read_text() if readme.exists() else '# SemanticKITTI local dataset\n\n'
    text = text.split('Status:')[0]
    complete = ', '.join(sorted(k for k,v in manifest['sequences'].items() if v.get('missing') == []))
    readme.write_text(text + f'Status: COMPLETE for sequences {complete}. Scans, labels, calibration, timestamps and SemanticKITTI SuMa poses verified. See download_manifest.json.\n')
    print(f'VERIFIED {seq}: {expected} matching scans, labels, poses and timestamps; every point has a label.', flush=True)


if __name__ == '__main__':
    main()
