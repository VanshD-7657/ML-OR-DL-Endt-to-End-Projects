import sys
from networksecurity.logging.logger import logger
class NetworkSecurityException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        self.error_message = error_message
        _,_, exc_tb = error_detail.exc_info()
        
        self.line_number = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error occured in script: {self.file_name} at line number: {self.line_number} with error message: {self.error_message} "


if __name__ == "__main__":
    try:
        logger.info("Testing custom exception")
        a = 1/0
        print("This will not be printed")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

     
    