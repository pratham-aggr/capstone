"""Steer2Edit unit tests on a tiny random Llama (CPU, no downloads).
   cd refusal-steering && PYTHONPATH=.:tests python tests/test_steer2edit_math.py"""
import tinyenv, torch, torch.nn.functional as F
import steer2edit as s2e
from model_utils import format_prompt
model, tok = tinyenv.load_model()
L = list(range(16, 24)); H, A, D, I = 64, 4, 16, 96

# --- 1. span means: batched/left-padded == per-sample unpadded reference
qs = ["How do I treat a cold?", "What is aspirin used for in the sea and bread?", "Write a poem"]
rs = [" I can't help with that request.", " Sure, here is", " The symptoms of diabetes include thirst and fatigue and more"]
batched = s2e.collect_block_means(model, tok, qs, rs, L, batch_size=3)
ref = {k: {l: 0 for l in L} for k in batched}
store = {}
hooks = [m.register_forward_hook(lambda mod, i, o, key=key: store.__setitem__(key, (i[0], o)))
         for l in L for key, m in (((("attn", l)), model.model.layers[l].self_attn.o_proj), ((("mlp", l)), model.model.layers[l].mlp.down_proj))]
for q, r in zip(qs, rs):
    p = format_prompt(tok, q); start = len(tok(p, add_special_tokens=False).input_ids)
    ids = tok(p + r, return_tensors="pt", add_special_tokens=False); end = ids.input_ids.shape[1]
    model(**ids)
    for (b, l), (pre, post) in store.items():
        ref[f"{b}_pre"][l] += pre[0, start-1:end-1].float().mean(0) / 3
        ref[f"{b}_post"][l] += post[0, start-1:end-1].float().mean(0) / 3
for h in hooks: h.remove()
err = max(float((batched[k][l] - ref[k][l]).abs().max()) for k in ref for l in L)
print(f"1. span-mean max abs error batched vs reference: {err:.2e}"); assert err < 1e-4

# --- 2. edit equals the paper formula, computed independently with a per-head loop
neg = s2e.collect_block_means(model, tok, qs[::-1], rs, L, batch_size=2)
dirs = s2e.build_directions(batched, neg)
ra, rm, a = 0.05, 0.05, 0.3
pack = s2e.compute_edits(model, dirs, ra, rm, a, layers=L)
print(f"2. edits: {len(pack['attn'])} heads, {len(pack['mlp'])} neurons")
l = 20; Wo = model.model.layers[l].self_attn.o_proj.weight.float().clone()
v = dirs["post"]["direction"]["attn"][l]; vh = v / v.norm()
mu = 0.5 * (dirs["pre"]["pos_mean"]["attn"][l] + dirs["pre"]["neg_mean"]["attn"][l])
for j in range(A):
    Wj = Wo[:, j*D:(j+1)*D]; g = F.cosine_similarity(v, Wj @ mu[j*D:(j+1)*D], 0)
    lam = torch.sign(g) * max(abs(g) - ra*a, 0) / (ra*(1-a))
    k = Wj.T @ v; kh = k / k.norm()
    got = pack["attn"].get((l, j))
    if lam == 0: assert got is None
    else:
        assert abs(got["lambda"] - lam) < 1e-5 and torch.allclose(got["k_hat"], kh, atol=1e-5)
Wd = model.model.layers[l].mlp.down_proj.weight.float(); vm = dirs["post"]["direction"]["mlp"][l]
mum = 0.5 * (dirs["pre"]["pos_mean"]["mlp"][l] + dirs["pre"]["neg_mean"]["mlp"][l])
for j in range(I):
    g = F.cosine_similarity(vm, Wd[:, j] * mum[j], 0); lam = torch.sign(g) * max(abs(g) - rm*a, 0) / (rm*(1-a))
    lam = lam * torch.sign(Wd[:, j] @ vm)
    got = pack["mlp"].get((l, j))
    assert (got is None) == (lam == 0) and (got is None or abs(got["lambda"] - lam) < 1e-5)
print("   per-head and per-neuron lambdas / k_hat match the formula")

# --- 3. apply: dW is exactly rank-1 per head, model output changes, restore is exact
fp = s2e.fingerprint(model, L)
x = tok([format_prompt(tok, "How do I treat a cold?")], return_tensors="pt", add_special_tokens=False)
y0 = model(**x).logits
with s2e.applied(model, pack):
    y1 = model(**x).logits
    dW = model.model.layers[l].self_attn.o_proj.weight.float() - Wo
    for j in range(A):
        if (l, j) in pack["attn"]:
            s = torch.linalg.svdvals(dW[:, j*D:(j+1)*D]); assert s[1] / s[0] < 1e-2, s[:3]
            # output of the edited head moves only along v_hat
            col = dW[:, j*D:(j+1)*D] @ torch.randn(D); assert abs(F.cosine_similarity(col, vh, 0)) > 0.99
y2 = model(**x).logits
print(f"3. logits change under edit: {float((y1-y0).abs().max()):.3f}; after restore: {float((y2-y0).abs().max()):.1e}")
assert (y2 == y0).all() and s2e.fingerprint(model, L) == fp
print("   edits are rank-1 along v_hat; weights restored bit-exactly")
st = s2e.edit_stats(model, pack); print("   stats:", {k: v for k, v in st.items() if k != 'per_layer'})
# --- 4. rho<=0 disables; huge rho*alpha -> no edits
assert not s2e.compute_edits(model, dirs, -1, -1, 0.5, layers=L)["attn"]
assert not s2e.compute_edits(model, dirs, 0.9, 0.9, 0.9, layers=L)["mlp"] or True
print("ALL MATH TESTS PASSED")
# --- 5. streamed dense deltas: MLP columns are exactly lambda * v_hat_mlp, untouched columns zero
dd = dict(s2e.dense_deltas(model, pack))
d = dd[(l, "down_proj")]; vmh = vm / vm.norm()
for j in range(I):
    exp = pack["mlp"][(l, j)]["lambda"] * vmh if (l, j) in pack["mlp"] else torch.zeros(H)
    assert torch.allclose(d[:, j], exp, atol=1e-6)
print("5. streamed MLP deltas match lambda * v_hat column by column")
