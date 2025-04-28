REAL_TOTAL_CLIENTS=5

TOTAL_CLIENTS=100
CLIENT_ID=-1
NUM_SAMPLES=512
LAYER_SIZE=2048
IGNORE_WARMUP=True

cp docker-compose_template.yaml docker-compose.yaml
for i in $(seq 0 $((REAL_TOTAL_CLIENTS-1)))
do
  cat client_module_template.yaml | sed "s/<ID>/$CLIENT_ID/g" | sed "s/<REAL_CLIENT_ID>/$i/g" | sed "s/<TOTAL>/$TOTAL_CLIENTS/g" | sed "s/<NUM_SAMPLES>/$NUM_SAMPLES/g" | sed "s/<LAYER_SIZE>/$LAYER_SIZE/g" | sed "s/<IGNORE_WARMUP>/$IGNORE_WARMUP/g" >> docker-compose.yaml
done
