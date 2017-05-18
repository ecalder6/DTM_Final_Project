@ECHO OFF
ECHO "twitter_seq2seq"
python train.py --cov_path=analytics/twitter_seq2seq --use_mutual=False --use_vae=False --use_highway=False --kl_anneal=False --use_checkpoint=False --task=twitter --iterations=2000 --output_file=../data/outputs/twitter_seq2seq.txt

ECHO "twitter_vae"
python train.py --cov_path=analytics/twitter_vae --use_mutual=False --use_vae=True --use_highway=False --kl_anneal=False --use_checkpoint=True --task=twitter --iterations=2000 --output_file=../data/outputs/twitter_vae.txt

ECHO "twitter_vae_anneal"
python train.py --cov_path=analytics/twitter_vae_anneal --use_mutual=False --use_vae=True --use_highway=False --kl_anneal=True --use_checkpoint=False --task=twitter --iterations=2000 --output_file=../data/outputs/twitter_vae.txt

ECHO "twitter_vae_highway"
python train.py --cov_path=analytics/twitter_vae_highway --use_mutual=False --use_vae=True --use_highway=True --kl_anneal=False --use_checkpoint=True --task=twitter --iterations=5000 --output_file=../data/outputs/twitter_vae_highway.txt

ECHO "twitter_vae_highway_anneal"
python train.py --cov_path=analytics/twitter_vae_highway_annealz --use_mutual=False --use_vae=True --use_highway=True --kl_anneal=True --use_checkpoint=False --task=twitter --iterations=500 --update_every=10 --save_z=True --output_file=../data/outputs/twitter_vae_highway_annealz.txt
python train.py --cov_path=analytics/twitter_vae_highway_mutual_anneal --use_mutual=True --use_vae=True --use_highway=True --kl_anneal=True --use_checkpoint=False --task=twitter --iterations=2000 --update_every=100 --save_z=False --output_file=../data/outputs/twitter_vae_highway_mutual_anneal.txt

ECHO "twitter_vae_mutual"
python train.py --cov_path=analytics/twitter_vae_mutual1 --use_mutual=True --use_vae=True --use_highway=False --kl_anneal=False --use_checkpoint=False --task=twitter --iterations=2000 --output_file=../data/outputs/twitter_vae_mutual1.txt
python train.py --cov_path=analytics/twitter_vae_mutual2 --use_mutual=True --use_vae=True --use_highway=False --kl_anneal=False --use_checkpoint=False --task=twitter --iterations=2000 --output_file=../data/outputs/twitter_vae_mutual2.txt
python train.py --cov_path=analytics/twitter_vae_mutual3 --use_mutual=True --use_vae=True --use_highway=False --kl_anneal=False --use_checkpoint=False --task=twitter --iterations=2000 --output_file=../data/outputs/twitter_vae_mutual3.txt

ECHO "twitter_vae_mutual_anneal"
python train.py --cov_path=analytics/twitter_vae_mutual_anneal --use_mutual=True --use_vae=True --use_highway=False --kl_anneal=True --use_checkpoint=False --task=twitter --iterations=2000 --output_file=../data/outputs/twitter_vae_mutual_anneal.txt

ECHO "twitter_vae_mutual_highway"
python train.py --cov_path=analytics/twitter_vae_mutual_highway --use_mutual=True --use_vae=True --use_highway=True --kl_anneal=False --use_checkpoint=False --task=twitter --iterations=2000 --output_file=../data/outputs/twitter_vae_mutual_highway.txt

ECHO "twitter_vae_mutual_highway_anneal"
python train.py --cov_path=analytics/twitter_vae_mutual_highway_anneal --use_mutual=True --use_vae=True --use_highway=True --kl_anneal=True --use_checkpoint=False --task=twitter --iterations=2000 --output_file=../data/outputs/twitter_vae_mutual_highway_anneal.txt

ECHO "movie_seq2seq"
python train.py --cov_path=analytics/movie_seq2seq --use_mutual=False --use_vae=False --use_highway=False --kl_anneal=False --use_checkpoint=False --task=movie --iterations=2000 --output_file=../data/outputs/movie_seq2seq.txt

ECHO "movie_vae"
python train.py --cov_path=analytics/movie_vae --use_mutual=False --use_vae=True --use_highway=False --kl_anneal=False --use_checkpoint=False --task=movie --iterations=2000 --output_file=../data/outputs/movie_vae.txt

ECHO "twitter_movie_vae_annealseq2seq"
python train.py --cov_path=analytics/movie_vae_anneal --use_mutual=False --use_vae=True --use_highway=False --kl_anneal=True --use_checkpoint=False --task=movie --iterations=2000 --output_file=../data/outputs/movie_vae.txt

ECHO "movie_vae_highway"
python train.py --cov_path=analytics/movie_vae_highway --use_mutual=False --use_vae=True --use_highway=True --kl_anneal=False --use_checkpoint=False --task=movie --iterations=2000 --output_file=../data/outputs/movie_vae_highway.txt

ECHO "movie_vae_highway_anneal"
python train.py --cov_path=analytics/movie_vae_highway_anneal --use_mutual=False --use_vae=True --use_highway=True --kl_anneal=True --use_checkpoint=False --task=movie --iterations=2000 --output_file=../data/outputs/movie_vae_highway_anneal.txt

ECHO "movie_vae_mutual"
python train.py --cov_path=analytics/movie_vae_mutual --use_mutual=True --use_vae=True --use_highway=False --kl_anneal=False --use_checkpoint=False --task=movie --iterations=2000 --output_file=../data/outputs/movie_vae_mutual.txt

ECHO "movie_vae_mutual_anneal"
python train.py --cov_path=analytics/movie_vae_mutual_anneal --use_mutual=True --use_vae=True --use_highway=False --kl_anneal=True --use_checkpoint=False --task=movie --iterations=2000 --output_file=../data/outputs/movie_vae_mutual_anneal.txt

ECHO "movie_vae_mutual_highway"
python train.py --cov_path=analytics/movie_vae_mutual_highway --use_mutual=True --use_vae=True --use_highway=True --kl_anneal=False --use_checkpoint=False --task=movie --iterations=2000 --output_file=../data/outputs/movie_vae_mutual_highway.txt

ECHO "movie_vae_mutual_highway_anneal"
python train.py --cov_path=analytics/movie_vae_mutual_highway_anneal --use_mutual=True --use_vae=True --use_highway=True --kl_anneal=True --use_checkpoint=False --task=movie --iterations=2000 --output_file=../data/outputs/movie_vae_mutual_highway_anneal.txt