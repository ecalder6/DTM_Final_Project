from Converter import Converter
import argparse

'''
Driver program for tokenizing input and convert to tfrecords.
'''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--input_filename', default='raw/twitter.txt', type=str)
    parser.add_argument('--output_filename', default='train/twitter.tfrecords', type=str)
    parser.add_argument('--meta_file', default='metadata/twitter_metadata', type=str)
    parser.add_argument('--data_type', default='twitter', type=str)
    parser.add_argument('--num_lines', default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Example command for movie:
    #   python data.py --input_filename=raw/movie.txt --output_filename=train/movie.tfrecords --meta_file=metadata/movie_metadata --data_type=movie

    converter = Converter(input_filename = args.data_dir + args.input_filename, output_filename = args.data_dir + args.output_filename, 
                            meta_file = args.data_dir + args.meta_file, data_type = args.data_type, lines = args.num_lines)
    converter.process_data()
    print("Done")

if __name__ == "__main__":
    main()
