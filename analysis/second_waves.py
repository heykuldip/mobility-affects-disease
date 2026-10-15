"""Does each run's epidemic die out after its first wave, or survive to a second one?

For every run of each model, from the run's compact summary (summary_ticks.csv: agents in each state per tick), this
reports the first-wave peak, the lowest number of exposed + infectious agents after it (the trough), whether infection
died out (exposed + infectious = 0) and when, whether it ever restarted after dying out (it shouldn't: the model has no
imported cases), the second-wave peak, and how many susceptibles were left at the trough, at their lowest, and at the
end (waning immunity should refill them).

Standard library only, Python 3.6 or later. Usage:
    python3 analysis/second_waves.py <sweep folder> [--models random_advan nhts_advan] [--out file.csv]
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

MODELS = ['nhts_advan', 'nhts_distance', 'nhts_random', 'random_places', 'random_advan']
STATES = ['Susceptible', 'Exposed', 'Infectious', 'Recovered']
TICKS_PER_DAY = 24
FIRST_WAVE_END_DAY = 25          # as in the notebook: the first-wave peak is sought within this many days


def runs_of(tree, model):
    """The notebook's rule: run_NN folders (not run_NN.previous_*) with a run_info.json or an agent log."""
    folder = tree / model / 'runs'
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob('run_*') if p.name[4:].isdigit()
                  and ((p / 'run_info.json').exists() or (p / 'agent_log0.csv').exists()))


def read_summary(path):
    ticks, rows = [], {s: [] for s in STATES}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        missing = [c for c in ['tick'] + STATES if c not in reader.fieldnames]
        if missing:
            raise ValueError('{}: missing columns {}'.format(path, missing))
        for r in reader:
            ticks.append(int(float(r['tick'])))
            for s in STATES:
                rows[s].append(int(float(r[s])))
    order = sorted(range(len(ticks)), key=ticks.__getitem__)       # in tick order, whatever the file's order
    return [ticks[i] for i in order], {s: [v[i] for i in order] for s, v in rows.items()}


def analyse(ticks, st):
    S, E, I, R = (st[s] for s in STATES)
    active = [e + i for e, i in zip(E, I)]
    population = S[0] + E[0] + I[0] + R[0]
    window = [k for k, t in enumerate(ticks) if t <= FIRST_WAVE_END_DAY * TICKS_PER_DAY]
    p = max(window, key=lambda k: I[k])                             # first-wave peak (first tick at the maximum)
    out = {'ticks': len(ticks), 'last_day': round(ticks[-1] / TICKS_PER_DAY, 2), 'population': population,
           'peak1_infectious': I[p], 'peak1_day': round(ticks[p] / TICKS_PER_DAY, 2)}
    # The first wave has fallen once infectious agents drop below 1% of its peak. Any second wave peaks after that,
    # and the trough is the fewest exposed + infectious between the two peaks (or to the end, if nothing rises).
    fall = next((k for k in range(p + 1, len(ticks)) if I[k] < 0.01 * I[p]), None)
    if fall is None:
        out['note'] = 'the first wave never fell below 1% of its peak'
        return out
    p2 = max(range(fall, len(ticks)), key=lambda k: I[k])
    rises = I[p2] > I[fall]
    window = range(p + 1, (p2 if rises else len(ticks) - 1) + 1)
    t = min(window, key=lambda k: active[k])
    zero = next((k for k in window if active[k] == 0), None)
    s_min = min(range(len(ticks)), key=lambda k: S[k])
    out.update({
        'trough_active': active[t], 'trough_day': round(ticks[t] / TICKS_PER_DAY, 2),
        'died_out': zero is not None,
        'died_out_day': round(ticks[zero] / TICKS_PER_DAY, 2) if zero is not None else '',
        'restarted_after_dying_out': zero is not None and any(active[k] > 0 for k in range(zero + 1, len(ticks))),
        'peak2_infectious': I[p2] if rises else 0,
        'peak2_day': round(ticks[p2] / TICKS_PER_DAY, 2) if rises else '',
        'susceptible_at_trough': S[t], 'susceptible_min': S[s_min],
        'susceptible_min_share': round(S[s_min] / population, 4) if population else '',
        'susceptible_min_day': round(ticks[s_min] / TICKS_PER_DAY, 2),
        'susceptible_end': S[-1], 'recovered_end': R[-1],
        'negative_counts': any(x < 0 for st_ in (S, E, I, R) for x in st_),
    })
    return out


def summarise(model, rows):
    done = [r for r in rows if 'trough_active' in r]
    if not done:
        return '{}: no runs analysed'.format(model)
    med = lambda key: statistics.median(r[key] for r in done)
    died = [r for r in done if r['died_out']]
    second = [r for r in done if r['peak2_infectious'] >= 1000]
    line = ('{}: {} runs; died out after the first wave in {}; restarted after dying out in {}; second wave '
            '(1,000+ infectious) in {}. Median trough {} exposed + infectious (min {}, max {}); median lowest '
            'susceptibles {:.1%} of the population; median second-wave peak {}').format(
        model, len(done), len(died), sum(r['restarted_after_dying_out'] for r in done), len(second),
        med('trough_active'), min(r['trough_active'] for r in done), max(r['trough_active'] for r in done),
        med('susceptible_min_share'), med('peak2_infectious'))
    if died:
        line += '; died out on day {} (median)'.format(statistics.median(r['died_out_day'] for r in died))
    bad = [r for r in done if r['negative_counts']]
    if bad:
        line += '. WARNING: negative counts in {} runs'.format(len(bad))
    short = [r for r in done if r['ticks'] < max(x['ticks'] for x in done)]
    if short:
        line += '. WARNING: {} runs have fewer ticks than the longest; check they finished'.format(len(short))
    return line


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('tree', help='the sweep folder made by scripts/run_sweep.sh')
    ap.add_argument('--models', nargs='*', default=MODELS)
    ap.add_argument('--out', default='second_waves.csv')
    args = ap.parse_args()
    tree = Path(args.tree)
    all_rows, problems = [], 0
    for model in args.models:
        rows = []
        for run in runs_of(tree, model):
            path = run / 'summary_ticks.csv'
            if not path.exists():
                print('{}/{}: no summary_ticks.csv'.format(model, run.name), file=sys.stderr)
                problems += 1
                continue
            try:
                r = analyse(*read_summary(path))
            except Exception as e:                                     # report and carry on with the other runs
                print('{}/{}: {}'.format(model, run.name, e), file=sys.stderr)
                problems += 1
                continue
            r = dict({'model': model, 'run': run.name}, **r)
            rows.append(r)
        print(summarise(model, rows))
        all_rows += rows
    if all_rows:
        keys = []
        for r in all_rows:
            keys += [k for k in r if k not in keys]
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print('wrote {} ({} runs){}'.format(args.out, len(all_rows),
                                             '; {} problems, listed above'.format(problems) if problems else ''))
    return 1 if problems and not all_rows else 0


if __name__ == '__main__':
    sys.exit(main())
