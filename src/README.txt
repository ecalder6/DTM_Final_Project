Jason Yim and Eric Calder


File structure:
~/
    data/
        analytics/
        metadata/
        model/
        outputs/
        raw/
        train/
    src/
        char_data.py
        Converter.py
        data.py
        graphz.py
        LSTMVAE.py
        Reader.py
        train.py
        run.cmd
        README.txt

How to run the code:
    0. make sure you have tensorflow installed (GPU version strongly recommended).
    1. make sure either the twitter dataset or the movie dataset is in ~/data/raw.
    2. in ~/src/, run python data.py --input_filename=raw/movie.txt --output_filename=train/movie.tfrecords --meta_file=metadata/movie_metadata --data_type=movie
        2.1. for twitter, replace movie with twitter in command.
    3. Train the model using run.cmd (either running it or running one of the command in it)
