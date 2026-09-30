"""Static figures of qualified DMET candidates; no solver or phase inference."""

from pathlib import Path

from pyscf_agent.artifacts import default_artifact_repository


def write_phase_plots(analysis, directory):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    directory = Path(directory)
    families = sorted({point['family'] for point in analysis['points']})
    colors = {
        'afm': '#3576b3',
        'cdw': '#d06431',
        'mixed': '#9063ac',
        'near_unordered': '#449d7b',
        'ambiguous': '#d4b144',
        'unknown': '#888888',
    }
    artifacts = []
    for family in families:
        points = [p for p in analysis['points'] if p['family'] == family]
        if not any(p['winner'] for p in points):
            continue
        fig = Figure(figsize=(7, 5), layout='constrained')
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        for branch, color in colors.items():
            subset = [
                p for p in points if p['winner'] and p['winner']['branch'] == branch
            ]
            if subset:
                ax.scatter(
                    [p['coordinates']['U'] for p in subset],
                    [p['coordinates']['V'] for p in subset],
                    color=color,
                    label=branch,
                    s=65,
                )
        coexist = [p for p in points if p['possible_hysteresis']]
        if coexist:
            ax.scatter(
                [p['coordinates']['U'] for p in coexist],
                [p['coordinates']['V'] for p in coexist],
                facecolors='none',
                edgecolors='black',
                s=150,
                linewidths=1.4,
                label='Distinct converged states (possible hysteresis)',
            )
        missing = [p for p in points if not p['winner']]
        if missing:
            ax.scatter(
                [p['coordinates']['U'] for p in missing],
                [p['coordinates']['V'] for p in missing],
                marker='x',
                color='gray',
                label='No qualified result',
            )
        ax.set(
            xlabel='U',
            ylabel='V',
            title='Lowest qualified DMET energy at each sampled point',
        )
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ('dmet-phase-map-' + family[:12] + '.png')
        fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=0.15)
        artifacts.append(
            default_artifact_repository().register_existing(
                path,
                kind='postprocess-plot',
                mime_type='image/png',
                description='DMET state map; rings mark possible hysteresis',
            )
        )
        for u in sorted({p['coordinates']['U'] for p in points}):
            line = sorted(
                (p for p in points if p['coordinates']['U'] == u),
                key=lambda p: p['coordinates']['V'],
            )
            fig = Figure(figsize=(7, 5), layout='constrained')
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            for branch, color in colors.items():
                # NaNs break lines across missing branch samples.
                energies = [
                    (p['branch_minima'].get(branch) or {}).get(
                        'energy_per_site', float('nan')
                    )
                    for p in line
                ]
                if any(branch in p['branch_minima'] for p in line):
                    ax.plot(
                        [p['coordinates']['V'] for p in line],
                        energies,
                        'o-',
                        color=color,
                        label=branch,
                    )
            ax.set(
                xlabel='V',
                ylabel='Energy per site (' + points[0]['energy_unit'] + ')',
                title=f'DMET branch energies, U={u:g}',
            )
            if ax.lines:
                ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
            path = directory / ('dmet-branch-energies-' + family[:12] + f'-U{u:g}.png')
            fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=0.15)
            artifacts.append(
                default_artifact_repository().register_existing(
                    path,
                    kind='postprocess-plot',
                    mime_type='image/png',
                    description='Converged branch energy curves; gaps remain unsampled',
                )
            )
    return [artifact for artifact in artifacts if artifact]
