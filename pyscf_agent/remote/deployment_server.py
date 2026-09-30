from __future__ import annotations

import argparse
import base64
import configparser
import json
import os
import re
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..runtime_identity import RuntimeIdentity


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _section_blocks(text: str) -> List[Tuple[Optional[str], str]]:
    matches = list(re.finditer(r'(?m)^\[([^\]]+)\]\s*$', text))
    if not matches:
        return [(None, text)]
    blocks: List[Tuple[Optional[str], str]] = []
    if matches[0].start() > 0:
        blocks.append((None, text[:matches[0].start()]))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append((match.group(1).strip(), text[match.start():end]))
    return blocks


def _render_section(name: str, values: Dict[str, str]) -> str:
    lines = ['[{0}]'.format(name)]
    for key, value in values.items():
        normalized = str(value or '')
        if '\n' not in normalized:
            lines.append('{0} = {1}'.format(key, normalized))
            continue
        lines.append('{0} ='.format(key))
        lines.extend('    {0}'.format(item) for item in normalized.splitlines())
    return '\n'.join(lines) + '\n\n'


def configure_remote_release(
    *,
    server_config: str,
    base_profile: str,
    server_profile: str,
    environment_id: str,
    release_root: str,
    base_python: str,
    identity: RuntimeIdentity,
) -> Dict[str, str]:
    config_path = Path(server_config).expanduser().resolve()
    release = Path(release_root).expanduser().resolve()
    source = release / 'source'
    if not config_path.is_file():
        raise ValueError('Server Slurm configuration was not found: {0}'.format(config_path))
    if not (source / 'pyproject.toml').is_file():
        raise ValueError('Remote release source is incomplete: {0}'.format(source))

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(str(config_path), encoding='utf-8')
    base_section = 'server:{0}'.format(base_profile)
    if not parser.has_section(base_section):
        raise ValueError('Base server profile was not found: {0}'.format(base_profile))

    identity_path = release / 'runtime-identity.json'
    identity_path.write_text(
        json.dumps(identity.to_dict(), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    wrapper = release / 'bin' / 'python'
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        '#!/usr/bin/env bash\n'
        + 'export PYTHONPATH=' + shlex.quote(str(source))
        + '${PYTHONPATH:+:${PYTHONPATH}}\n'
        + 'export PYSCF_AGENT_RUNTIME_IDENTITY=' + shlex.quote(str(identity_path)) + '\n'
        + 'exec ' + shlex.quote(base_python) + ' "$@"\n',
        encoding='utf-8',
    )
    wrapper.chmod(0o750)

    server_values = dict(parser.items(base_section, raw=True))
    base_work_root = Path(server_values['work_root']).expanduser()
    server_values.update({
        'project_root': str(source),
        'python_executable': str(wrapper),
        'work_root': str(base_work_root / 'environments' / environment_id),
    })
    generated = [_render_section('server:{0}'.format(server_profile), server_values)]
    profile_prefix = 'profile:{0}:'.format(base_profile).lower()
    for section in parser.sections():
        if section.lower().startswith(profile_prefix):
            suffix = section[len(profile_prefix):]
            generated.append(_render_section(
                'profile:{0}:{1}'.format(server_profile, suffix),
                dict(parser.items(section, raw=True)),
            ))

    target_names = {'server:{0}'.format(server_profile).lower()}
    target_prefix = 'profile:{0}:'.format(server_profile).lower()
    preserved = [
        block
        for name, block in _section_blocks(config_path.read_text(encoding='utf-8'))
        if name is None
        or (
            name.lower() not in target_names
            and not name.lower().startswith(target_prefix)
        )
    ]
    updated = ''.join(preserved).rstrip() + '\n\n' + ''.join(generated)
    backup = config_path.with_name(
        config_path.name + '.bak-' + identity.release_id
    )
    if not backup.exists():
        shutil.copy2(str(config_path), str(backup))
    temporary = config_path.with_name(config_path.name + '.tmp-' + identity.release_id)
    temporary.write_text(updated, encoding='utf-8')
    os.chmod(str(temporary), config_path.stat().st_mode & 0o777)
    os.replace(str(temporary), str(config_path))

    return {
        'schema': 'pyscf-agent.remote-release.v1',
        'environment_id': environment_id,
        'release_id': identity.release_id,
        'release_root': str(release),
        'source_root': str(source),
        'python_wrapper': str(wrapper),
        'runtime_identity_file': str(identity_path),
        'server_profile': server_profile,
        'server_config_backup': str(backup),
        'configured_at': _timestamp(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Configure one immutable remote release.')
    parser.add_argument('--server-config', required=True)
    parser.add_argument('--base-profile', required=True)
    parser.add_argument('--server-profile', required=True)
    parser.add_argument('--environment-id', required=True)
    parser.add_argument('--release-root', required=True)
    parser.add_argument('--base-python', required=True)
    parser.add_argument('--identity-base64', required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(base64.b64decode(args.identity_base64).decode('utf-8'))
    result = configure_remote_release(
        server_config=args.server_config,
        base_profile=args.base_profile,
        server_profile=args.server_profile,
        environment_id=args.environment_id,
        release_root=args.release_root,
        base_python=args.base_python,
        identity=RuntimeIdentity.from_dict(payload),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = ['configure_remote_release', 'main']
