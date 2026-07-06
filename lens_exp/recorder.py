# lens-exp recorder: captures per-sample all-layer retouch-token latents,
# prompt-pass norm statistics (E4) and retouch-token attention maps (E3).
import numpy as np
import torch


class LensRecorder:
    def __init__(self, capture_attn=False):
        self.capture_attn = capture_attn
        self.current_meta = None   # set by driver before each _generate (bs=1)
        self.last = None           # dict of arrays for the sample just recorded

    def record(self, sample_in_batch, hidden_states, attentions, output_ids,
               token_steps, image_spans):
        i = sample_in_batch
        out = {}
        L = len(hidden_states[0])  # num layers + 1 (embeddings)

        # ---- all-layer latents at each retouch token's generation step (E1) ----
        for name, step in token_steps.items():
            lat = torch.stack([hidden_states[step][l][i].reshape(-1) for l in range(L)])
            out[f"lat_{name}"] = lat.float().cpu().numpy().astype(np.float16)
            out[f"step_{name}"] = np.int32(step)

        # ---- prompt-pass norms per layer: visual vs text tokens (E4) ----
        if image_spans is not None:
            s, e, tot = image_spans[i]
            out["img_span"] = np.array([s, e, tot], dtype=np.int32)
            vis_norm, txt_norm = [], []
            for l in range(L):
                h = hidden_states[0][l][i].float()   # [prompt_len, hid]
                n = h.norm(dim=-1)
                vis_norm.append(n[s:e].mean().item())
                txt = torch.cat([n[:s], n[e:]])
                txt_norm.append(txt.mean().item())
            out["vis_norm"] = np.array(vis_norm, dtype=np.float32)
            out["txt_norm"] = np.array(txt_norm, dtype=np.float32)

        out["gen_len"] = np.int32(output_ids.shape[1])

        # ---- retouch-token attention over image patches (E3) ----
        if self.capture_attn and attentions is not None and image_spans is not None:
            s, e, _ = image_spans[i]
            n_steps = len(attentions)
            for name, step in token_steps.items():
                # step `step`: forward that PRODUCED the token (query = previous token)
                # step `step+1`: forward where the retouch token itself is the query
                for tag, st in (("gen", step), ("self", step + 1)):
                    if st >= n_steps or attentions[st] is None or attentions[st][0] is None:
                        continue
                    layers = []
                    for l in range(len(attentions[st])):
                        a = attentions[st][l][i]          # [heads, q, kv]
                        a = a[:, -1, :]                    # query = last position
                        layers.append(a)
                    att = torch.stack(layers).float()      # [layers, heads, kv]
                    img_att = att[:, :, s:e]
                    out[f"att_{name}_{tag}_mean"] = img_att.mean(dim=1).cpu().numpy().astype(np.float16)
                    out[f"att_{name}_{tag}_mass"] = img_att.sum(dim=(1, 2)).div(att.shape[1]).cpu().numpy().astype(np.float32)
                    if tag == "self":
                        out[f"att_{name}_self_full"] = img_att.cpu().numpy().astype(np.float16)
        self.last = out
        return out
