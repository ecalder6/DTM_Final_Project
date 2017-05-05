import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    args = get_args()
    f = open(args.file_name, "rb")
    npzfile = np.load(f)

    s = 0
    graph_dimensions = []

    while s < args.latent_size:
        graph_dimensions.append(s)
        s += args.step_size

    print(graph_dimensions)

    graph_vals = {}
    for i in range(args.num_sentences):
        graph_vals[i] = dict(zip(graph_dimensions, [[] for _ in range(len(graph_dimensions))]))

    for i in range(len(npzfile.files)):
        z = npzfile['arr_' + str(i)]
        print(z.shape)
        for k in graph_vals:
            for d in graph_vals[k]:
                graph_vals[k][d].append(z[k, d])

    for k in graph_vals:
        plt.figure(k)
        plt.ylabel('Z value')
        plt.xlabel('Iterations (increment by 10)')
        plt.title('VAE + Mutual Info sentence auto encoding Z values for ' + str(len(graph_dimensions)) + ' dimensions')
        for d in graph_vals[k]:
            plt.plot(graph_vals[k][d])
    plt.show()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_size', default=128, type=int)
    parser.add_argument('--num_sentences', default=10, type=int)
    parser.add_argument('--step_size', default=60, type=int)
    parser.add_argument('--file_name', default='../data/analytics/z', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Sample command for movie:
    #   python train.py --checkpoint_path=../movie_lstm_checkpoint/ --use_mutual=False --use_vae=False --use_highway=False --task=movie
    main()
