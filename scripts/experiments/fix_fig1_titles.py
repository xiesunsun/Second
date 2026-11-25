from pathlib import Path
p = Path("/home/sunxie/paper-experiments-20261124/plot_tevc_fig1.py")
lines = p.read_text().splitlines()
out = []
for line in lines:
    if "title=" in line and "(a)" in line:
        line = "           title=r\"(a) $K = \\lfloor 0.15p \\rfloor$: Precision\", ylim=(75.5, 89.2))"
    elif "title=" in line and "(b)" in line:
        line = "           title=r\"(b) $K = \\lfloor 0.15p \\rfloor$: Subset Size\", ylim=(0, 560))"
    elif "title=" in line and "(c)" in line:
        line = "           title=r\"(c) $K = 8$: Balanced Accuracy\", ylim=(50.0, 74.0))"
    out.append(line)
p.write_text("\n".join(out) + "\n")
