import numpy as np
import matplotlib.pyplot as plt

# Veri kümesini yükle
df = pd.read_csv("dataset.csv")

# Veriyi ayır ve etiketleri encode et
circles = df[df["Pattern Type"] == "Circle"][["x", "y"]].values
crosses = df[df["Pattern Type"] == "Cross-sign"][["x", "y"]].values


# Basit lineer ayrım için sınır çizgisini hesapla
# Bu örnekte, karar sınırı için basit bir lineer regresyon uygulanıyor
def fit_linear_separator(points1, points2):
    # Tüm veriyi birleştir ve etiketle
    X = np.vstack((points1, points2))
    y = np.array([0] * len(points1) + [1] * len(points2))

    # Normal denklem ile katsayıları hesapla: w = (X.T * X)^-1 * X.T * y
    X_design = np.c_[np.ones(X.shape[0]), X]  # Sabit terim için 1 ekle
    w = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    return w


# Modeli öğren
w = fit_linear_separator(circles, crosses)


# Karar sınırını çiz
def plot_decision_boundary(w, points1, points2):
    plt.scatter(points1[:, 0], points1[:, 1], color="blue", label="Circle")
    plt.scatter(points2[:, 0], points2[:, 1], color="red", label="Cross-sign")

    # Karar sınırını hesapla: w0 + w1*x + w2*y = 0
    x_vals = np.linspace(0, max(df["x"]), 100)
    y_vals = -(w[0] + w[1] * x_vals) / w[2]

    plt.plot(x_vals, y_vals, color="green", label="Decision Boundary")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Decision Boundary")
    plt.show()


plot_decision_boundary(w, circles, crosses)
