"""Contact network and mobility measures of one run (Tables 3 and 4 of the paper).

    python analysis/networks.py <run folder> [--chunk-rows N] [--apl-epsilon E] [--apl-max N] [--ncol] [--force]

Reads <run folder>/agent_log0.csv a chunk at a time, so memory stays bounded however large the log. Contacts become
undirected edges between agents, each kept once over the whole run. Then, on the giant component: degrees, Louvain
modularity and communities, and the average path length, estimated from breadth-first searches from randomly chosen
agents until its 95% interval is within --apl-epsilon of the mean. Writes network.json and degrees.npy into the run
folder (and the giant component as graph.ncol with --ncol). Seeds come from the run's base seed, so results are
reproducible.

In the same pass over the log, the mobility measures: each row's position is matched to a building by its coordinates;
a visit is a row whose building differs from the same agent's previous row, so each agent's first location counts as
one; unique places are the buildings that appear in the log at all; and the mean dwell time is the mean length, in
ticks (hours), of the runs of consecutive rows at one building, which equals rows / visits. Also the same by building
type, and each agent's number of distinct buildings. Writes mobility.json.
"""
import argparse
import json
import os
import random
import resource
import shutil
import sys
import time

import numpy as np
import pandas as pd
import igraph as ig

SHIFT = np.uint64(1 << 21)          # edge key = smaller id * 2^21 + larger id; agent ids must be below 2,097,152
BUCKETS = 64

p = argparse.ArgumentParser()
p.add_argument('run')
p.add_argument('--chunk-rows', type=int, default=10_000_000)
p.add_argument('--apl-epsilon', type=float, default=0.01, help='stop when the 95%% half-width is this share of the mean (1%%, as in the paper)')
p.add_argument('--apl-min', type=int, default=30)
p.add_argument('--apl-max', type=int, default=1000)
p.add_argument('--ncol', action='store_true', help='also write the giant component as graph.ncol (large)')
p.add_argument('--force', action='store_true', help='recompute even if network.json exists')
a = p.parse_args()

out_json = os.path.join(a.run, 'network.json')
mob_json = os.path.join(a.run, 'mobility.json')
if os.path.exists(out_json) and not a.force:
    sys.exit(f'{out_json} exists; use --force to recompute')
log_path = os.path.join(a.run, 'agent_log0.csv')
started = os.path.join(a.run, 'network.started')       # marks the run as in progress, for submit_networks.sh
job = (f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}" if 'SLURM_ARRAY_JOB_ID' in os.environ
       else os.environ.get('SLURM_JOB_ID', 'local'))
open(started, 'w').write(f'{job}\n{time.strftime("%Y-%m-%dT%H:%M:%S")}\n')
info_path = os.path.join(a.run, 'run_info.json')
seed = json.load(open(info_path))['base_seed'] if os.path.exists(info_path) else 1
timings, t0 = {}, time.time()

# Buildings, for matching each row's coordinates to a building, named as in the paper
cfg_path = next((p for p in [os.path.join(a.run, 'config.yaml'), os.path.join(a.run, '..', '..', 'config', 'data_config.yaml')]
                 if os.path.exists(p)), None) or sys.exit(f'no config.yaml in {a.run}: it names the buildings file')
import yaml
cfg = yaml.safe_load(open(cfg_path))
model_dir = os.path.join(a.run, '..', '..')
bld = pd.read_csv(os.path.join(model_dir, cfg['input_buildings']), usecols=['bldg_id', 'bldg_type', 'x_centroid', 'y_centroid'])
TYPES = {'residential': 'Home', 'workplace': 'Work', 'schools': 'School', 'hospitals': 'Hospital', 'errands': 'Errand',
         'recreation': 'Recreation', 'restaurants': 'Restaurant'}
type_names = sorted(set(TYPES.get(t, t) for t in bld['bldg_type']))
bld_type = np.array([type_names.index(TYPES.get(t, t)) for t in bld['bldg_type']], dtype=np.int8)
bkey = np.trunc(bld['x_centroid']).astype('int64') * 10_000_000 + np.trunc(bld['y_centroid']).astype('int64')
border = np.argsort(bkey.to_numpy(), kind='stable')
bkeys_sorted = bkey.to_numpy()[border]
n_types, home_type = len(type_names), type_names.index('Home') if 'Home' in type_names else -1
last_b = np.full(1, -1, dtype=np.int64)        # each agent's building at its previous row (-1: not seen yet)
cur_len = np.zeros(1, dtype=np.int64)          # length of the agent's current visit, in ticks
cur_type = np.zeros(1, dtype=np.int8)
visited = np.zeros(len(bld), dtype=bool)
visits_by_type = np.zeros(n_types, dtype=np.int64)
dwell_sum = np.zeros(n_types, dtype=np.int64)
dwell_n = np.zeros(n_types, dtype=np.int64)
pairs, pairs_pending, unmatched = [], 0, 0      # (agent, building) pairs, deduplicated as they accumulate


def grow(n):
    global last_b, cur_len, cur_type
    if n > len(last_b):
        extra = n - len(last_b)
        last_b = np.concatenate([last_b, np.full(extra, -1, dtype=np.int64)])
        cur_len = np.concatenate([cur_len, np.zeros(extra, dtype=np.int64)])
        cur_type = np.concatenate([cur_type, np.zeros(extra, dtype=np.int8)])


def stamp(name):
    timings[name] = round(max(0.0, time.time() - t0 - sum(timings.values())), 1)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    print(f'{name}: {timings[name]:,.0f} s; peak memory so far {peak:,.1f} GB', flush=True)


# 1. Every contact, as an undirected key, written to bucket files after removing repeats within each chunk
tmp = os.path.join(a.run, '_edges_tmp')
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
files = [open(os.path.join(tmp, f'{b:02d}.bin'), 'wb') for b in range(BUCKETS)]
rows = 0
for chunk in pd.read_csv(log_path, usecols=['tick', 'agent_id', 'x', 'y', 'agents_contacted'],
                         dtype={'tick': 'int64', 'agent_id': 'int64', 'x': 'float64', 'y': 'float64', 'agents_contacted': str},
                         chunksize=a.chunk_rows):
    rows += len(chunk)
    # Mobility: buildings, visits and dwell times, tick by tick (each agent appears once per tick)
    key = np.trunc(chunk['x'].to_numpy()).astype('int64') * 10_000_000 + np.trunc(chunk['y'].to_numpy()).astype('int64')
    pos = np.clip(np.searchsorted(bkeys_sorted, key), 0, len(bkeys_sorted) - 1)
    hit = bkeys_sorted[pos] == key
    unmatched += int((~hit).sum())
    bidx = np.where(hit, border[pos], -2)
    ticks, aid = chunk['tick'].to_numpy(), chunk['agent_id'].to_numpy()
    grow(int(aid.max()) + 1)
    order = np.argsort(ticks, kind='stable')
    starts = np.flatnonzero(np.r_[True, np.diff(ticks[order]) != 0])
    for lo_i, hi_i in zip(starts, np.r_[starts[1:], len(order)]):
        sel = order[lo_i:hi_i]
        ids, b = aid[sel], bidx[sel]
        new = last_b[ids] != b
        ended = new & (last_b[ids] >= 0)
        np.add.at(dwell_sum, cur_type[ids[ended]], cur_len[ids[ended]])
        np.add.at(dwell_n, cur_type[ids[ended]], 1)
        t_now = np.where(b >= 0, bld_type[np.maximum(b, 0)], 0).astype(np.int8)
        visits_by_type += np.bincount(t_now[new & (b >= 0)], minlength=n_types)
        cur_len[ids] = np.where(new, 1, cur_len[ids] + 1)
        cur_type[ids] = np.where(new, t_now, cur_type[ids])
        last_b[ids] = b
        visited[b[b >= 0]] = True
        nb_ = new & (b >= 0)
        if nb_.any():
            pairs.append(np.unique(ids[nb_].astype(np.int64) * (1 << 20) + b[nb_]))
            pairs_pending += len(pairs[-1])
    if pairs_pending > 20_000_000:
        pairs, pairs_pending = [np.unique(np.concatenate(pairs))], 0
    lists = chunk['agents_contacted'].str.slice(1, -1)
    keep = lists.str.len() > 0
    lists, owners = lists[keep], chunk['agent_id'].to_numpy()[keep.to_numpy()]
    if len(lists) == 0:
        continue
    counts = lists.str.count(',').to_numpy() + 1
    others = np.array(','.join(lists.tolist()).split(','), dtype=np.int64)
    me = np.repeat(owners, counts)
    lo, hi = np.minimum(me, others).astype(np.uint64), np.maximum(me, others).astype(np.uint64)
    if hi.max() >= SHIFT:
        sys.exit(f'agent id {int(hi.max())} is too large for the edge key')
    keys = np.unique(lo * SHIFT + hi)
    bucket = (keys % np.uint64(BUCKETS)).astype(np.int64)
    order = np.argsort(bucket, kind='stable')
    keys, bucket = keys[order], bucket[order]
    edges_at = np.searchsorted(bucket, np.arange(BUCKETS + 1))
    for b in range(BUCKETS):
        if edges_at[b + 1] > edges_at[b]:
            keys[edges_at[b]:edges_at[b + 1]].tofile(files[b])
    print(f'    {rows / 1e6:,.0f} million rows read', flush=True)
for f in files:
    f.close()
stamp('read log')

# Mobility, finished: close each agent's last visit, then the totals
seen = last_b >= 0
np.add.at(dwell_sum, cur_type[seen], cur_len[seen])
np.add.at(dwell_n, cur_type[seen], 1)
pair_keys = np.unique(np.concatenate(pairs)) if pairs else np.zeros(0, dtype=np.int64)
del pairs
pair_agent, pair_bldg = pair_keys >> 20, pair_keys & ((1 << 20) - 1)
per_agent = np.bincount(pair_agent, minlength=len(last_b))[seen]
non_home = bld_type[pair_bldg] != home_type
per_agent_out = np.bincount(pair_agent[non_home], minlength=len(last_b))[seen]
visits = int(visits_by_type.sum())
n_agents = int(seen.sum())
mobility = {
    'run': os.path.abspath(a.run), 'rows': rows, 'agents': n_agents, 'unmatched_rows': unmatched,
    'visits': visits, 'moves': visits - n_agents, 'mean_dwell_hours': rows / visits if visits else float('nan'),
    'unique_buildings': int(visited.sum()), 'buildings': int(len(bld)),
    'unique_non_home_buildings': int(visited[bld_type != home_type].sum()),
    'by_type': {name: {'buildings': int((bld_type == k).sum()), 'unique_visited': int(visited[bld_type == k].sum()),
                       'visits': int(visits_by_type[k]),
                       'mean_dwell_hours': float(dwell_sum[k] / dwell_n[k]) if dwell_n[k] else float('nan')}
                for k, name in enumerate(type_names)},
    'distinct_buildings_per_agent': {'mean': float(per_agent.mean()), 'median': float(np.median(per_agent)),
                                     'p90': float(np.percentile(per_agent, 90))},
    'distinct_non_home_buildings_per_agent': {'mean': float(per_agent_out.mean()), 'median': float(np.median(per_agent_out)),
                                              'p90': float(np.percentile(per_agent_out, 90))},
}
json.dump(mobility, open(mob_json, 'w'), indent=1)
print(json.dumps({k: v for k, v in mobility.items() if k not in ('run', 'by_type')}))
del last_b, cur_len, cur_type, visited, pair_keys, pair_agent, pair_bldg, per_agent, per_agent_out

# 2. Each edge once over the whole run. Kept lean, since a full run has hundreds of millions of edges: each bucket is
# deduplicated in place on disk, then written straight into one int32 edge array. Agent ids serve as vertex ids
# directly, so no remapping is needed; agents with no contacts become isolated vertices, which are left out below.
total = 0
for b in range(BUCKETS):
    path = os.path.join(tmp, f'{b:02d}.bin')
    u = np.unique(np.fromfile(path, dtype=np.uint64))
    u.tofile(path)
    total += len(u)
    del u
edges = np.empty((total, 2), dtype=np.int32)
at = 0
for b in range(BUCKETS):
    u = np.fromfile(os.path.join(tmp, f'{b:02d}.bin'), dtype=np.uint64)
    edges[at:at + len(u), 0] = u // SHIFT
    edges[at:at + len(u), 1] = u % SHIFT
    at += len(u)
    del u
shutil.rmtree(tmp)
n_vertices = int(edges.max()) + 1 if total else 0
print(f'{total:,} unique edges among {n_vertices:,} agent ids', flush=True)
stamp('unique edges')

# 3. The graph and its giant component (python-igraph passes a NumPy edge array to its C core without Python objects)
g = ig.Graph(n=n_vertices, edges=edges, directed=False)
del edges
if a.ncol:
    g.vs['agent'] = list(range(n_vertices))       # agent ids, carried into the giant component for graph.ncol
isolated = int((np.asarray(g.degree()) == 0).sum())
nodes = n_vertices - isolated                    # agents with at least one contact, as in the paper's graph
comps = g.connected_components() if hasattr(g, 'connected_components') else g.clusters()
giant = comps.giant()
n_components = len(comps) - isolated             # isolated agents aren't vertices of the paper's graph
del comps
stamp('graph')

# 4. Degrees, modularity and communities on the giant component (as in the paper)
deg = np.array(giant.degree(), dtype=np.int32)
np.save(os.path.join(a.run, 'degrees.npy'), deg)
random.seed(seed)
ig.set_random_number_generator(random)
communities = giant.community_multilevel()
stamp('modularity')

# 5. Average path length: breadth-first searches from random agents until the 95% interval is narrow enough
rng = np.random.default_rng(seed)
n = giant.vcount()
sources = rng.permutation(n)[:min(a.apl_max, n)]
means = []
half = float('nan')
for s in sources:
    d = np.asarray(giant.distances(source=[int(s)])[0], dtype=float)
    d = d[(d > 0) & np.isfinite(d)]
    if len(d):
        means.append(d.mean())
    if len(means) >= a.apl_min:
        m = np.mean(means)
        half = 1.96 * np.std(means, ddof=1) / np.sqrt(len(means))
        if half / m <= a.apl_epsilon:
            break
apl = float(np.mean(means)) if means else float('nan')
if len(means) > 1 and not np.isfinite(half):
    half = 1.96 * np.std(means, ddof=1) / np.sqrt(len(means))
stamp('path length')

if a.ncol:
    giant.write_ncol(os.path.join(a.run, 'graph.ncol'), names='agent', weights=None)
    stamp('write ncol')

result = {
    'run': os.path.abspath(a.run), 'base_seed': seed, 'log_rows': rows,
    'nodes': nodes, 'edges': g.ecount(), 'components': n_components,
    'giant_nodes': n, 'giant_edges': giant.ecount(), 'giant_share': round(n / nodes, 6),
    'mean_degree': float(deg.mean()), 'median_degree': float(np.median(deg)),
    'p99_degree': float(np.percentile(deg, 99)), 'max_degree': int(deg.max()),
    'modularity': float(communities.modularity), 'communities': len(communities),
    'apl': apl, 'apl_low': apl - half, 'apl_high': apl + half, 'apl_sources': len(means),
    'seconds': timings, 'peak_memory_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2),
}
json.dump(result, open(out_json, 'w'), indent=1)
os.remove(started)
print(json.dumps({k: v for k, v in result.items() if k not in ('run', 'seconds')}))
