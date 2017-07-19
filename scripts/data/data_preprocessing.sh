./getEmbedding.py
./getGloveEmbedding.py 50
./mkTrainDevdata.py
./mkTestdata.py
./gen_oov_set.py train
./gen_oov_set.py dev
./gen_oov_set.py test
./paddingZero.py train
./paddingZero.py dev
./paddingZero.py test
./gen_tfrecords.py train
./gen_tfrecords.py dev
./gen_tfrecords.py test

