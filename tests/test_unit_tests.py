import unittest


loader = unittest.TestLoader()
suite = loader.discover(start_dir="tests/unit_tests", pattern="test_*.py")
runner = unittest.TextTestRunner()
result = runner.run(suite)
