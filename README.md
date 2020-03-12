Data path contains the FMA file and Go file.

Use FindNDR.py you can create all ndr data.

You can create PCSMax test data by using findPCSMax.py

CreateAllDataBert.py can create all the concepts' embedding vectors.

By the way, if you want use bert you can use bert-as-service:

first pip the package you need:

pip install bert-serving-server  # server
pip install bert-serving-client  # client, independent of `bert-serving-server`

then use command:
bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4 
