# Datasets — provenance & attribution

This audit analyzes **derived judgments** (closure scores, structural/aesthetic metrics)
over reasoning traces sampled from the public Hugging Face datasets below. **No raw dataset
text is redistributed in this repo** — `outputs/*.jsonl` contains only per-trace judgments
keyed by a content hash (`uid`). To re-run the pipeline, fetch the sources from Hugging Face;
each retains its original upstream license.

| Logical name | Hugging Face source | Teacher | Method | n judged |
|---|---|---|---|---:|
| `glm` | `Jackrong/GLM-5.1-Reasoning-1M-Cleaned` | GLM-5.1 | native-capture | 60 |
| `deepseek` | `Jackrong/DeepSeek-V4-Distill-8000x` | DeepSeek-V4-Distill | native-capture | 60 |
| `kimi` | `Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned` | KIMI-K2.5 | native-capture | 40 |
| `qwen` | `Jackrong/Qwen3.5-reasoning-700x` | Qwen3.5-27B | native-capture | 40 |
| `claude46_ti` | `Jackrong/Claude-opus-4.6-TraceInversion-9000x` | claude-opus-4.6 | trace-inversion (reconstruction) | 40 |
| `claude47_ti` | `Jackrong/Claude-opus-4.7-TraceInversion-5000x` | claude-opus-4.7 | trace-inversion (reconstruction) | 40 |
| `gemini` | `Roman1111111/gemini-3.1-pro-hard-high-reasoning` | gemini-3.1-pro | native-capture | 40 |
| `nohurry_opus` | `nohurry/Opus-4.6-Reasoning-3000x-filtered` | claude-opus-4.6 | human-filtered "high quality" | 40 |
| `angrygiraffe` | `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k` | claude-opus-4.6/4.7 | synthetic / roleplay | 40 |
| `roman_claude` | `Roman1111111/claude-opus-4.6-10000x` | claude-opus-4.6 | filtered / direct (no `<think>`) | 40 |

**Source families:** *Jackrong* (the seven `Jackrong/*` sets), *Roman* (`Roman1111111/*`),
and the *Opus*-teacher distillations (`nohurry`, `angrygiraffe`, the Claude trace-inversions,
`roman_claude`). The `glm` and `deepseek` sets share an identical question set, enabling the
length-matched natural experiment.

**Distillation-method taxonomy used throughout:**
- **native-capture** — the teacher's actual `<think>` stream is recorded verbatim.
- **reconstruction / trace-inversion** — a `<think>` is synthesized *after the fact* from a
  final answer (the loop is rebuilt, not observed).
- **filtered / direct** — answers selected or emitted with little/no exposed reasoning.

**Ethics / scope.** Traces are solitary math/NL problem-solving; refusal and safety content
was already cleaned from these sources upstream (hence value-anchoring is out of scope here).
The adapters that load each source live in [`src/distill_audit/adapters.py`](src/distill_audit/adapters.py).
