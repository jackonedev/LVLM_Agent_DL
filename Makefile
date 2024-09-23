format:
	isort --profile=black --skip .venv --skip .env_aux . &&\
	autopep8 --in-place ./*.py &&\
	black --line-length 88 . --exclude '(\.venv|\.env_aux)'

lint:
	pylint *.py
