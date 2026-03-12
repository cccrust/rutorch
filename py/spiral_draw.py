import pandas as pd
import matplotlib.pyplot as plt

pts = pd.read_csv("out/spiral_points.csv")
grid = pd.read_csv("out/spiral_grid.csv")

plt.figure(figsize=(6,6))
plt.scatter(pts.x, pts.y, c=pts.label, s=10, cmap="coolwarm", alpha=0.9)

# 用 p1 畫等高線（0.5 是 decision boundary）
plt.tricontour(grid.x, grid.y, grid.p1, levels=[0.5], colors="black", linewidths=1)
plt.title("Spiral Decision Boundary")
plt.show()
