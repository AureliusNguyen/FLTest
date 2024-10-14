#poetry run python fl_testing/scripts/main.py framework=flare

# poetry run python fl_testing/scripts/main.py framework=pfl

pytest fl_testing/tests/ --html=results/pytests_report.html --self-contained-html 

