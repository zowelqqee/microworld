"""Render each comparison metric in a separate PNG and SVG figure."""
from __future__ import annotations
import argparse, json, os, tempfile
from pathlib import Path

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('--summary', required=True); parser.add_argument('--output', required=True); args=parser.parse_args(argv)
    # CI/sandbox home directories may be read-only; keep matplotlib's cache
    # outside the benchmark output and force a non-interactive backend.
    cache = Path(tempfile.gettempdir()) / "microworld-matplotlib-cache"; cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rows=json.loads(Path(args.summary).read_text())['rows']; output=Path(args.output); output.mkdir(parents=True, exist_ok=True)
    metrics={'complete_latency':['latency_p50_ms','latency_p95_ms','latency_p99_ms'], 'answer_accuracy':['answer_accuracy'], 'negative_audit_accuracy':['negative_accuracy'], 'unsupported_claim_rate':['unsupported_claim_rate'], 'predicate_adherence':['predicate_adherence'], 'artifact_size':['artifact_size_mib'], 'startup_load_time':['startup_ms'], 'extra_runtime_memory':['extra_memory_mib'], 'qwen_ttft_vs_complete':['ttft_p50_ms','latency_p50_ms'], 'quality_latency_scatter':['answer_accuracy','latency_p50_ms']}
    for title, fields in metrics.items():
        fig, ax=plt.subplots(figsize=(8,4)); labels=[f"{row['system']}\n{row['category']}" for row in rows]
        if title == 'quality_latency_scatter':
            for row in rows: ax.scatter(row.get(fields[1]) or 0, row.get(fields[0]) or 0, label=f"{row['system']} / {row['category']}")
            ax.set_xlabel('latency p50 ms'); ax.set_ylabel('answer accuracy'); ax.legend(fontsize=6)
        else:
            for field in fields: ax.bar(range(len(rows)), [row.get(field) or 0 for row in rows], alpha=.7, label=field)
            ax.set_xticks(range(len(rows)), labels, rotation=45, ha='right'); ax.legend()
        ax.set_title(title.replace('_',' ')); fig.tight_layout()
        for suffix in ('png','svg'): fig.savefig(output / f'{title}.{suffix}')
        plt.close(fig)
    return 0
if __name__ == '__main__': raise SystemExit(main())
