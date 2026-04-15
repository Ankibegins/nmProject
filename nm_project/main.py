import base64
import io
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template_string, request

# Use non-interactive backend for server-side image generation.
matplotlib.use("Agg")

app = Flask(__name__)


def load_data(filepath):
    """Load dataset, extracting price (target y) and area (feature X)."""
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1, usecols=(0, 1))
    y = data[:, 0:1]
    x_raw = data[:, 1:2]
    return x_raw, y


def compute_loss(x, y, theta):
    """Compute Mean Squared Error (MSE) loss."""
    n = len(y)
    predictions = x.dot(theta)
    mse = (1 / n) * np.sum((y - predictions) ** 2)
    return mse


def compute_gradient(x, y, theta):
    """Compute gradient of MSE loss."""
    n = len(y)
    predictions = x.dot(theta)
    gradient = -(2 / n) * x.T.dot(y - predictions)
    return gradient


def compute_hessian(x):
    """Compute Hessian matrix of MSE loss."""
    n = len(x)
    hessian = (2 / n) * x.T.dot(x)
    return hessian


def train_newton_model(iterations=15):
    """Train linear regression model using Newton's method."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset", "Housing.csv")
    x_raw, y = load_data(dataset_path)

    n = len(y)
    x = np.hstack((np.ones((n, 1)), x_raw))
    theta = np.zeros((2, 1))
    losses = [compute_loss(x, y, theta)]

    h_inv = np.linalg.inv(compute_hessian(x))
    for _ in range(iterations):
        grad = compute_gradient(x, y, theta)
        theta = theta - h_inv.dot(grad)
        losses.append(compute_loss(x, y, theta))

    return x_raw, y, theta, losses


def predict_price(area, theta):
    """Predict price for given area."""
    return theta[0, 0] + theta[1, 0] * area


def predict_area(price, theta):
    """Predict area for given price."""
    if theta[1, 0] == 0:
        return 0
    return (price - theta[0, 0]) / theta[1, 0]


def generate_plots(x_raw, y, theta, losses):
    """Generate model plots and return as base64 PNG."""
    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # Top-left: Area -> Price graph
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.scatter(x_raw, y, color="royalblue", alpha=0.6, label="Actual Data")
    x_line = np.linspace(x_raw.min(), x_raw.max(), 100).reshape(-1, 1)
    x_with_bias = np.hstack((np.ones((100, 1)), x_line))
    y_line = x_with_bias.dot(theta)
    ax1.plot(x_line, y_line, color="crimson", linewidth=2, label="Predicted Line")
    ax1.set_title("Area vs Price")
    ax1.set_xlabel("Area (sq. ft.)")
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Top-right: Price -> Area graph (inverse relation)
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.scatter(y, x_raw, color="teal", alpha=0.6, label="Actual Data")
    y_line_inverse = np.linspace(y.min(), y.max(), 100).reshape(-1, 1)
    x_line_inverse = (y_line_inverse - theta[0, 0]) / theta[1, 0]
    ax2.plot(y_line_inverse, x_line_inverse, color="darkorange", linewidth=2, label="Predicted Line")
    ax2.set_title("Price vs Area")
    ax2.set_xlabel("Price")
    ax2.set_ylabel("Area (sq. ft.)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # Bottom (full width): Error convergence
    ax3 = fig.add_subplot(grid[1, :])
    ax3.plot(range(len(losses)), losses, marker="o", color="green", linewidth=2)
    ax3.set_title("Newton Loss Reduction Method (Error Convergence)")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Mean Squared Error (MSE)")
    ax3.set_yscale("log")
    ax3.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close(fig)
    return image_base64


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Housing Price-Area Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1100px;
            margin: 20px auto;
            padding: 0 16px;
        }
        .card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            background: #fafafa;
        }
        .row {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
        }
        input[type="number"] {
            padding: 8px;
            width: 220px;
        }
        button {
            padding: 8px 14px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            color: white;
            background: #2563eb;
        }
        .result {
            font-weight: bold;
            margin-top: 10px;
        }
        img {
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: white;
        }
    </style>
</head>
<body>
    <h1>Housing Predictor Web App</h1>

    <div class="card">
        <h2>Graphs</h2>
        <img src="data:image/png;base64,{{ plot_image }}" alt="Prediction and loss convergence graphs">
    </div>

</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    x_raw, y, theta, losses = train_newton_model(iterations=15)

    plot_image = generate_plots(x_raw, y, theta, losses)
    return render_template_string(
        HTML_TEMPLATE,
        plot_image=plot_image,
    )


if __name__ == "__main__":
    app.run(debug=True)
