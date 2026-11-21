from hailo_sdk_client import ClientRunner
import sys

runner = ClientRunner(har=str(sys.argv[1])) # path to har file
print(runner.get_hn_dict())