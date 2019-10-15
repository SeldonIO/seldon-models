MODEL_NAME=movie_sentiment_roberta
TAG=0.1

.PHONY: download
download:
	curl

.PHONY: build
build:
	docker build . -t seldonio/${MODEL_NAME}:${TAG}

.PHONY: run_docker
run_docker:
	docker run --name ${MODEL_NAME} -d -p 5000:5000 seldonio/${MODEL_NAME}:${TAG}

.PHONY: test_docker
test_docker:
	curl 0.0.0.0:5000/predict -d '{"data":{"ndarray": ["like its bizarre heroine , it irrigates our souls ."]}}' -H "Content-Type: application/json"

.PHONY: stop_docker
stop_docker:
	docker stop ${MODEL_NAME} && docker rm ${MODEL_NAME}