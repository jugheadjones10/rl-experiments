import matplotlib.pyplot as plt
import pandas as pd


# Maybe move this to utils directory in torch-ac package
def plot_return(log_path, title):
    df = pd.read_csv(log_path)

    # Create the plot with a specific style and size
    # plt.style.use("seaborn")
    plt.figure(figsize=(12, 6))

    # Plot the main line (return_mean)
    plt.plot(
        df["update"], df["return_mean"], color="blue", label="Mean Return", linewidth=2
    )

    # Create the shaded area for standard deviation
    plt.fill_between(
        df["update"],
        df["return_mean"] - df["return_std"],
        df["return_mean"] + df["return_std"],
        color="blue",
        alpha=0.2,
        label="±1 Standard Deviation",
    )

    # Customize the plot
    plt.xlabel("Updates", fontsize=12)
    plt.ylabel("Return", fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Add some padding to the y-axis limits to make the plot look better
    plt.margins(y=0.1)

    # Adjust layout to prevent label clipping
    plt.tight_layout()

    # Show the plot
    plt.show()
