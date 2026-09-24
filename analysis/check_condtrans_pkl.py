import pickle, numpy as np
old = pickle.load(open('analysis/agent_ml_data_episode_fourstate_union_stratval.pkl', 'rb'))
new = pickle.load(open('analysis/agent_ml_data_episode_condtrans_union_stratval.pkl', 'rb'))
print('old top_ns:', len(old['top_ns']), 'fdim', old['fdim'], '| new top_ns:', len(new['top_ns']), 'fdim', new['fdim'])
assert old['top_ns'] == new['top_ns'], 'ns mismatch'
for ns in old['top_ns']:
    o, n = old['data'][ns], new['data'][ns]
    print(ns, 'old tr/val:', len(o['Xtr']), len(o['Xval']), '| new tr/val:', len(n['Xtr']), len(n['Xval']),
          '| te state identical:', bool(np.allclose(o['Xte'], n['Xte'][:, :o['Xte'].shape[1]])),
          '| Yte identical:', bool(np.array_equal(o['Yte'], n['Yte'])))
uc = new['meta']['union_concepts']
ok_derive = True
for ns in new['top_ns']:
    n = new['data'][ns]
    sel = [uc.index(c) for c in new['ns_vocab'][ns]]
    for split, yb in [('tr', 'Ytr'), ('val', 'Yval'), ('te', 'Yte')]:
        y4 = n[f'Y4{split}'][:, sel]
        ok_derive &= np.array_equal(((y4 == 0) | (y4 == 2)).astype(np.float32), n[yb])
print('binary labels derivable from Y4 on all ns/splits:', ok_derive)
ns = new['top_ns'][0]
n = new['data'][ns]
print('Y4tr shape:', n['Y4tr'].shape, 'values:', np.unique(n['Y4tr']), 'mask sum:', int(n['label_mask'].sum()))
print('groups sizes:', {k: len(v) for k, v in new['groups_by_ns'][ns].items()})
print('te_keys match old:', new['te_keys'][ns] == old['te_keys'][ns])
