help: ## show help message.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build:  # build dev docker image.
	docker-compose -f docker-compose.dev.yml build labeledmatrix

rebuild:  # Force rebuild the image
	rm -rf build/
	rm -rf dist/
	rm -rf *egg*/
	docker-compose -f docker-compose.dev.yml build --no-cache labeledmatrix

DOCKER_CMD := docker-compose -f docker-compose.dev.yml run --rm labeledmatrix

shell: build
	$(DOCKER_CMD) /bin/bash

test: build
	$(DOCKER_CMD) pytest --cov cyperf

lint: build
	$(DOCKER_CMD) pylint labeledmatrix --rcfile=setup.cfg

build_lock: # build docker image for pipenv lock
	docker-compose -f docker-compose.dev.yml build lock

DOCKER_LOCK_CMD := docker-compose -f docker-compose.dev.yml run --rm lock

lock: build_lock
	$(DOCKER_LOCK_CMD) pipenv lock --verbose
