"""
Plug-in to configure pytest to manually track pass/fail results.

Results will be sent to testing_logger for review.

https://stackoverflow.com/questions/57917863/returning-only-number-of-tests-passed-and-number-of-tests-failed
"""

import logging
import pytest

testing_logger = logging.getLogger('test_logger')


@pytest.mark.tryfirst
def pytest_configure(config):
    """
    Fixture for pytest configuration, to call TestCounter class immediately

    Args:
        config: Pytest plugin
    """
    config.pluginmanager.register(TestCounter(), 'test_counter')


class TestCounter:
    """
    Class to configure Pytest to keep track of number of tests passed/failed.
    """
    def __init__(self):
        """
        Instantiate TestCounter class with number of pass/fail tests set to 0.
        """
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        """
        Function to assess Pytest report after every test is run.

        Args:
            report (Pytest Plugin): Pytest return plugin to report test pass or fail.
        """
        if report.when != 'call':
            return
        if report.passed:
            testing_logger.info("=====SUCCESS: Test passed.=====")
            self.passed += 1
        elif report.failed:
            testing_logger.error("=====ERROR: Test failed.=====")
            self.failed += 1

    def pytest_sessionfinish(self):
        """
        Function to report total tests passed and failed upon exit.
        """
        testing_logger.info(
            "%s tests passed and %s tests failed.", str(
                self.passed), str(
                self.failed))
        print(self.passed, self.failed, sep=',')
