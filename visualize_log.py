import pandas as pd
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, help="Log file.")
    args = parser.parse_args()

    df = pd.read_csv(args.log_file)

    plt.figure()
    df.plot(x="frames", y="mean_win_rate")
    # df["mean_episode_len"].plot(x=df["frames"], secondary_y=True, color="g")
    plt.show()
    plt.savefig("plot_log.png")
