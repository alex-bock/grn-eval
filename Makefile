clean:
	find . -path "*/__pycache__/*" -delete
	find . -type d -name "__pycache__" -empty -delete
	rm -rf .pytest_cache
	rm .coverage

lint:
	flake8 ./grn_eval/ ./scripts/

test:
	coverage run --source ./grn_eval/ -m --omit="*/tests/*" pytest ./tests/ && coverage report -m