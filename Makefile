.PHONY: docs

init:
	pip3 install -r requirements.txt

test:
	# This runs all of the tests. To run an individual test, run py.test with
	# the -k flag, like "py.test -k test_path_is_not_double_encoded"
	find -name '*.pyc' -delete
	find -name "__pycache__" |xargs rm -r {}\; 2>/dev/null
	py.test tests

coverage:
	find -name '*.pyc' -delete
	find -name "__pycache__" |xargs rm -r {}\; 2>/dev/null
	py.test --verbose --cov-report term --cov=requests tests

publishPrepare:
	rm -fr build dist .egg PyMimircache.egg-info 2>/dev/null 
	python3 setup.py sdist bdist_wheel


publishTest: publishPrepare
	# python3 setup.py sdist upload -r pypitest
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
	rm -fr build dist .egg PyMimircache.egg-info


publish: publishPrepare
	# python3 setup.py register
	# python3 setup.py sdist upload -r pypi
	twine upload dist/*
	# python setup.py bdist_wheel --universal upload
	rm -fr build dist .egg PyMimircache.egg-info

build:
	python3 setup.py build_ext -i

install:
	pip3 install -r requirements.txt
	python3 setup.py install


docs:
	cd docs && make html
	@echo "\033[95m\n\nBuild successful! View the docs homepage at docs/_build/html/index.html.\n\033[0m"