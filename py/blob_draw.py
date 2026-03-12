import pandas as pd
import matplotlib.pyplot as plt

pts = pd.read_csv("out/blob_points.csv")
grid = pd.read_csv("out/blob_grid.csv")

plt.figure(figsize=(6,6))
plt.scatter(pts.x, pts.y, c=pts.label, s=10, cmap="coolwarm", alpha=0.9)

# 畫多類別的邊界：取 argmax 當類別
probs = grid[["p0", "p1", "p2"]].values
pred = probs.argmax(axis=1)
plt.tricontour(grid.x, grid.y, pred, levels=[0.5, 1.5], colors="black", linewidths=1)

plt.title("Blob Decision Boundary")
plt.show()
