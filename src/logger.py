import logging
import os
from datetime import datetime

LOG_FILE = f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
logs_path=os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    format="[%(asctime)s] %(filename)s:%(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
    ]
)

# to suppress flask logs and warning
logging.getLogger("werkzeug").setLevel(logging.ERROR)